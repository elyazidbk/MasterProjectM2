import numpy as np
import pandas as pd
from engine.orderClass import OHLCVOrder
from engine.debug import logger

class OHLCVBacktester:
    def __init__(self, data, strategy, fee_perc=0.0005, sp500_membership=None, rebalance_frequency=20):
        self.data = data
        self.strategy = strategy
        self.fee_perc = fee_perc
        self.trades = []
        self.equity_curve = []
        self.returns = []
        self.sp500_membership = sp500_membership  # DataFrame: index=date, columns=symbol, True/False
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_counter = 0

    def run(self, initial_cash=1_000_000, max_daily_deploy_percent=0.02):
        print(f"\n=== Running backtest for strategy: {getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy'))} ===")
        asset_symbols = sorted(list(set([col.split('_')[0] for col in self.data.columns if col.endswith('_Adjusted Close')])))
        n_assets = len(asset_symbols)
        asset_states = {
            symbol: {
                'cash': 0,
                'position': 0,
                'portfolio_value': [],
                'trades': [],
                'equity': [],
                'realized_pnl': [],
                'unrealized_pnl': [],
                'cumulative_realized_pnl': 0,
                'cumulative_realized_pnl_history': [],
                'positions': [],
                'equity_allocation': [],
                'cash_allocation': [],
                'skipped_trades': 0,
                'last_buy_price': None,
                'rolling_vol': None
            } for symbol in asset_symbols
        }
        asset_signals = {}
        asset_indicators = {}
        cash_buffer = 0.05 * initial_cash
        last_rebalance_vol = 0  # Initialize to 0 for baseline
        # Compute 20-day rolling volatility for each asset
        for symbol in asset_symbols:
            price_col = f'{symbol}_Adjusted Close'
            if price_col in self.data.columns:
                prices = self.data[price_col].pct_change()
                rolling_vol = prices.rolling(window=20).std()
                asset_states[symbol]['rolling_vol'] = rolling_vol
            else:
                asset_states[symbol]['rolling_vol'] = pd.Series([np.nan]*len(self.data), index=self.data.index)
        for symbol in asset_symbols:
            cols = [c for c in self.data.columns if c.startswith(symbol+'_')]
            asset_df = self.data[cols].copy()
            asset_df.columns = [c.replace(symbol+'_', '') for c in cols]
            asset_df = asset_df.reset_index(drop=True)
            asset_df['ticker'] = symbol
            asset_df['date'] = self.data.index[:len(asset_df)]
            if 'ticker' not in asset_df.columns or 'date' not in asset_df.columns:
                raise ValueError("asset_df must contain 'ticker' and 'date' columns for ML strategies.")
            sig, ind = self.strategy.generate_signals(asset_df)
            if isinstance(sig, list):
                sig = pd.Series(sig)
            sig = sig.shift(1).fillna(0)
            asset_signals[symbol] = sig
            asset_indicators[symbol] = ind
            asset_states[symbol]['indicators'] = ind
            asset_states[symbol]['signals'] = sig
        portfolio_cash = initial_cash
        portfolio_equity = [initial_cash]
        n_rows = len(self.data)
        for idx, (i, row) in enumerate(self.data.iterrows()):
            date = row.name if self.data.index.name else idx
            # --- Compute total portfolio value at this date ---
            total_portfolio_value = portfolio_cash
            for symbol in asset_symbols:
                price_col = f'{symbol}_Adjusted Close'
                if price_col in row and not pd.isna(row[price_col]) and row[price_col] != 0:
                    total_portfolio_value += asset_states[symbol]['position'] * row[price_col]
            # --- Compute value-weighted average rolling_vol ---
            total_value = 0
            weighted_vol = 0
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if price_col in row and not pd.isna(row[price_col]) and row[price_col] != 0:
                    vol = None
                    try:
                        vol = state['rolling_vol'].get(date, np.nan)
                    except Exception:
                        vol = np.nan
                    posval = state['position'] * row[price_col]
                    if not pd.isna(vol) and vol > 0 and posval > 0:
                        total_value += posval
                        weighted_vol += posval * vol
            curr_weighted_vol = (weighted_vol / total_value) if total_value > 0 else 0
            rebalance_due_to_vol = False
            if last_rebalance_vol > 0:
                if abs(curr_weighted_vol - last_rebalance_vol) / last_rebalance_vol > 0.10:
                    rebalance_due_to_vol = True
            # --- Rebalance logic ---
            if self.rebalance_counter >= self.rebalance_frequency or rebalance_due_to_vol:
                try:
                    pos_assets = [s for s in asset_symbols if asset_states[s]['position'] > 0]
                    valid_assets = []
                    inv_vols = []
                    prices = {}
                    skip_flags = {s: False for s in pos_assets}
                    for s in pos_assets:
                        price_col = f'{s}_Adjusted Close'
                        if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                            if not skip_flags[s]:
                                asset_states[s]['skipped_trades'] += 1
                                skip_flags[s] = True
                            continue
                        try:
                            vol = asset_states[s]['rolling_vol'].get(date, np.nan)
                        except Exception:
                            vol = np.nan
                        if pd.isna(vol) or vol == 0:
                            if not skip_flags[s]:
                                asset_states[s]['skipped_trades'] += 1
                                skip_flags[s] = True
                            continue
                        valid_assets.append(s)
                        inv_vols.append(1.0/vol)
                        prices[s] = row[price_col]
                    if not valid_assets:
                        self.rebalance_counter = 0
                        last_rebalance_vol = curr_weighted_vol  # update baseline even if no rebalance
                        continue
                    inv_vols = np.array(inv_vols)
                    norm_weights = inv_vols / inv_vols.sum()
                    # Use total_portfolio_value computed above
                    total_value = total_portfolio_value
                    for j, s in enumerate(valid_assets):
                        state = asset_states[s]
                        price = prices[s]
                        current_value = state['position'] * price
                        target_value = norm_weights[j] * total_value
                        delta_value = target_value - current_value
                        if abs(delta_value) < 0.005 * total_value:
                            continue
                        side = 'buy' if delta_value > 0 else 'sell'
                        available_cash = max(0, portfolio_cash - cash_buffer)
                        qty_change = int(np.floor(abs(delta_value) / (price * (1+self.fee_perc))))
                        if qty_change < 1:
                            if not skip_flags[s]:
                                state['skipped_trades'] += 1
                                skip_flags[s] = True
                            continue
                        if side == 'sell':
                            qty_change = min(qty_change, state['position'])
                        fee = qty_change * price * self.fee_perc
                        try:
                            if side == 'buy':
                                if (qty_change * price + fee) > available_cash:
                                    if not skip_flags[s]:
                                        state['skipped_trades'] += 1
                                        skip_flags[s] = True
                                    continue
                                state['position'] += qty_change
                                state['cash'] -= (qty_change * price + fee)
                                portfolio_cash -= (qty_change * price + fee)
                            else:
                                state['position'] -= qty_change
                                state['cash'] += (qty_change * price - fee)
                                portfolio_cash += (qty_change * price - fee)
                            state['trades'].append(OHLCVOrder(date, price, qty_change, side, fee, None, trade_type="rebalance"))
                            state['cash_allocation'].append(state['cash'])
                        except Exception:
                            if not skip_flags[s]:
                                state['skipped_trades'] += 1
                                skip_flags[s] = True
                    self.rebalance_counter = 0
                    last_rebalance_vol = curr_weighted_vol  # update baseline after any rebalance
                except Exception as e:
                    self.rebalance_counter = 0
                    last_rebalance_vol = curr_weighted_vol
            else:
                self.rebalance_counter += 1
            # --- Signal sells ---
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if self.sp500_membership is not None:
                    if date not in self.sp500_membership.index or symbol not in self.sp500_membership.columns or not self.sp500_membership.loc[date, symbol]:
                        continue
                if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                    continue
                price = row[price_col]
                signal = asset_signals[symbol][idx]
                indicator = asset_indicators[symbol][idx]
                if signal == -1 and state['position'] > 0:
                    quantity = state['position']
                    fee = price * self.fee_perc * quantity
                    if state['last_buy_price'] is not None:
                        trade_pnl = (price * quantity) - (state['last_buy_price'] * quantity) - fee
                    else:
                        trade_pnl = 0
                    state['cumulative_realized_pnl'] += trade_pnl
                    state['realized_pnl'].append(trade_pnl)
                    state['trades'].append(OHLCVOrder(date, price, quantity, 'sell', fee, indicator, trade_type="signal"))
                    state['cash'] += (price * quantity) - fee
                    portfolio_cash += (price * quantity) - fee
                    state['position'] = 0
                    state['last_buy_price'] = None
                    state['cash_allocation'].append(state['cash'])
            # --- Signal buys ---
            buy_candidates = []
            buy_skip_flags = {s: False for s in asset_symbols}
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if self.sp500_membership is not None:
                    if date not in self.sp500_membership.index or symbol not in self.sp500_membership.columns or not self.sp500_membership.loc[date, symbol]:
                        continue
                if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                    continue
                price = row[price_col]
                signal = asset_signals[symbol][idx]
                indicator = asset_indicators[symbol][idx]
                try:
                    vol = state['rolling_vol'].get(date, np.nan)
                except Exception:
                    vol = np.nan
                if signal == 1 and state['position'] == 0 and not pd.isna(vol) and vol > 0:
                    volume_col = f'{symbol}_Volume'
                    volume = row[volume_col] if volume_col in row and not pd.isna(row[volume_col]) else 0
                    buy_candidates.append((symbol, price, indicator, volume, vol))
            if buy_candidates:
                buy_candidates.sort(key=lambda x: x[3], reverse=True)
                inv_vols = np.array([1.0/x[4] for x in buy_candidates])
                norm_weights = inv_vols / inv_vols.sum()
                for k, (symbol, price, indicator, volume, vol) in enumerate(buy_candidates):
                    state = asset_states[symbol]
                    available_cash = max(0, portfolio_cash - cash_buffer)
                    allocation = min(norm_weights[k] * available_cash, available_cash, portfolio_cash * max_daily_deploy_percent)
                    skip_flag = False
                    if allocation < price * (1 + self.fee_perc):
                        if not buy_skip_flags[symbol]:
                            state['skipped_trades'] += 1
                            buy_skip_flags[symbol] = True
                        continue
                    max_quantity = int(np.floor(allocation / (price * (1 + self.fee_perc))))
                    max_quantity = max(max_quantity, 1)
                    max_volume = int(volume * 0.1) if volume > 0 else float('inf')
                    quantity = min(max_quantity, max_volume)
                    if quantity < 1:
                        if not buy_skip_flags[symbol]:
                            state['skipped_trades'] += 1
                            buy_skip_flags[symbol] = True
                        continue
                    total_cost = price * quantity + price * self.fee_perc * quantity
                    if total_cost > available_cash:
                        if not buy_skip_flags[symbol]:
                            state['skipped_trades'] += 1
                            buy_skip_flags[symbol] = True
                        continue
                    fee = price * self.fee_perc * quantity
                    state['trades'].append(OHLCVOrder(date, price, quantity, 'buy', fee, indicator, trade_type="signal"))
                    state['position'] += quantity
                    state['cash'] -= (price * quantity) + fee
                    portfolio_cash -= (price * quantity) + fee
                    state['last_buy_price'] = price
                    state['cash_allocation'].append(state['cash'])
            # --- Portfolio/accounting update ---
            total_equity = portfolio_cash
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if self.sp500_membership is not None:
                    if date not in self.sp500_membership.index or symbol not in self.sp500_membership.columns or not self.sp500_membership.loc[date, symbol]:
                        state['unrealized_pnl'].append(0)
                        state['portfolio_value'].append(state['cash'])
                        state['cash_allocation'].append(state['cash'])
                        continue
                if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                    state['unrealized_pnl'].append(0)
                    state['portfolio_value'].append(state['cash'])
                    state['cash_allocation'].append(state['cash'])
                    continue
                price = row[price_col]
                state['positions'].append(state['position'])
                state['unrealized_pnl'].append(state['position'] * price)
                state['equity'].append(state['position'] * price)
                state['cash_allocation'].append(state['cash'])
                state['equity_allocation'].append(state['position'] * price)
                state['portfolio_value'].append(state['cash'] + state['position'] * price)
                total_equity += state['position'] * price
            portfolio_equity.append(total_equity)
            for symbol in asset_symbols:
                asset_states[symbol]['cumulative_realized_pnl_history'].append(
                    asset_states[symbol]['cumulative_realized_pnl']
                )
        self.asset_states = asset_states
        self.asset_symbols = asset_symbols
        self.portfolio_equity_curve = portfolio_equity
        self.returns = pd.Series(self.portfolio_equity_curve).pct_change().fillna(0)
        skipped_trades_dict = {s: asset_states[s]['skipped_trades'] for s in asset_symbols}
        signal_counts = {s: len([t for t in asset_states[s]['trades'] if hasattr(t, 'trade_type') and t.trade_type == 'signal']) for s in asset_symbols}
        rebalance_counts = {s: len([t for t in asset_states[s]['trades'] if hasattr(t, 'trade_type') and t.trade_type == 'rebalance']) for s in asset_symbols}
        logger.info(f"Non-executed trades per asset: {skipped_trades_dict}")
        logger.info(f"Backtest complete. Signal trades: {signal_counts}")
        logger.info(f"Backtest complete. Rebalance trades: {rebalance_counts}")
        return asset_states, self.portfolio_equity_curve
