import re
import numpy as np
import pandas as pd
from ohlcv_order import OHLCVOrder
from debug import logger

class OHLCVBacktester:
    def __init__(self, data, strategy, fee_perc=0.0005):
        self.data = data
        self.strategy = strategy
        self.fee_perc = fee_perc
        self.trades = []
        self.equity_curve = []
        self.returns = []

    def run(self, initial_cash=1_000_000):
        print(f"\n=== Running backtest for strategy: {getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy'))} ===")
        # Identify asset symbols from columns (e.g., AAPL_Adjusted Close, MSFT_Adjusted Close)
        asset_symbols = sorted(list(set([col.split('_')[0] for col in self.data.columns if col.endswith('_Adjusted Close')])))
        n_assets = len(asset_symbols)

        # Initialize per-asset state
        asset_states = {
            symbol: {
                'cash': 0,
                'position': 0,
                'portfolio_value': [],  # cash + position*price at each step
                'trades': [],
                'equity': [],
                'realized_pnl': [],
                'unrealized_pnl': [],
                'cumulative_realized_pnl': 0,
                'cumulative_realized_pnl_history': [],  # track history per row
                'positions': [],
                'equity_allocation': [],
                'cash_allocation': [],
                'last_buy_price': None  # Track the last buy price
            } for symbol in asset_symbols
        }

        # Prepare per-asset signals/indicators
        asset_signals = {}
        asset_indicators = {}
        for symbol in asset_symbols:
            # Extract per-asset DataFrame
            cols = [c for c in self.data.columns if c.startswith(symbol+'_')]
            asset_df = self.data[cols].copy()
            asset_df.columns = [c.replace(symbol+'_', '') for c in cols]
            asset_df = asset_df.reset_index(drop=True)
            # Generate signals/indicators for this asset
            sig, ind = self.strategy.generate_signals(asset_df)

            # Ensure signal is a pandas Series
            if isinstance(sig, list):
                sig = pd.Series(sig)

            # Enforce T+1 execution delay
            sig = sig.shift(1).fillna(0)
            asset_signals[symbol] = sig
            asset_indicators[symbol] = ind

            # Store indicators in asset_states for later analysis
            asset_states[symbol]['indicators'] = ind
            asset_states[symbol]['signals'] = sig  # Store signals for analysis

        # Use a single portfolio cash pool
        portfolio_cash = initial_cash
        portfolio_equity = [initial_cash]

        for idx, (i, row) in enumerate(self.data.iterrows()):
            # 1. Process all sells first to free up cash
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                price = row[price_col]
                signal = asset_signals[symbol][idx]
                indicator = asset_indicators[symbol][idx]
                date = row.name if self.data.index.name else idx
                # Sell logic
                if signal == -1 and state['position'] > 0:
                    quantity = state['position']
                    fee = price * self.fee_perc * quantity
                    if state['last_buy_price'] is not None:
                        trade_pnl = (price * quantity) - (state['last_buy_price'] * quantity) - fee
                    else:
                        trade_pnl = 0  # Default to 0 if no last buy price is available
                    state['cumulative_realized_pnl'] += trade_pnl
                    state['realized_pnl'].append(trade_pnl)  # record per-trade PnL
                    # append sell order
                    state['trades'].append(OHLCVOrder(date, price, quantity, 'sell', fee, indicator))
                    # update per-asset cash on sell
                    state['cash'] += (price * quantity) - fee
                    portfolio_cash += (price * quantity) - fee
                    state['position'] = 0
                    state['last_buy_price'] = None  # Reset last buy price after selling

            # 2. Prepare buy candidates (signal==1 and no position)
            buy_candidates = []
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                    continue
                price = row[price_col]
                signal = asset_signals[symbol][idx]
                indicator = asset_indicators[symbol][idx]
                if signal == 1 and state['position'] == 0:
                    volume_col = f'{symbol}_Volume'
                    volume = row[volume_col] if volume_col in row and not pd.isna(row[volume_col]) else 0
                    buy_candidates.append((symbol, price, indicator, volume))

            # 3. Sort buy candidates by volume descending
            buy_candidates.sort(key=lambda x: x[3], reverse=True)

            # 4. Allocate up to 2% of available cash per buy, until cash exhausted
            for symbol, price, indicator, volume in buy_candidates:
                state = asset_states[symbol]
                allocation = min(portfolio_cash, portfolio_cash * 0.02)
                if allocation < price * (1 + self.fee_perc):
                    continue  # Not enough cash for even 1 share
                max_quantity = int(allocation // (price * (1 + self.fee_perc)))
                max_volume = int(volume * 0.1) if volume > 0 else float('inf')
                quantity = min(max_quantity, max_volume)
                if quantity > 0 and (price * quantity + price * self.fee_perc * quantity) <= portfolio_cash:
                    fee = price * self.fee_perc * quantity
                    state['trades'].append(OHLCVOrder(date, price, quantity, 'buy', fee, indicator))
                    state['position'] += quantity
                    # update per-asset cash on buy
                    state['cash'] -= (price * quantity) + fee
                    portfolio_cash -= (price * quantity) + fee
                    state['last_buy_price'] = price  # Update last buy price

            # 5. Update state for all assets (do not double-count cash)
            total_equity = portfolio_cash
            for symbol in asset_symbols:
                state = asset_states[symbol]
                price_col = f'{symbol}_Adjusted Close'
                if price_col not in row or pd.isna(row[price_col]) or row[price_col] == 0:
                    state['unrealized_pnl'].append(0)  # Append 0 if price is invalid
                    # record portfolio value even if price invalid
                    state['portfolio_value'].append(state['cash'])
                    continue
                price = row[price_col]
                state['positions'].append(state['position'])
                state['unrealized_pnl'].append(state['position'] * price)  # Update unrealized PnL
                state['equity'].append(state['position'] * price)
                state['cash_allocation'].append(0)
                state['equity_allocation'].append(state['position'] * price)
                # append true portfolio value for this asset
                state['portfolio_value'].append(state['cash'] + state['position'] * price)
                total_equity += state['position'] * price
            portfolio_equity.append(total_equity)

            # Append cumulative realized PnL history after processing this row
            for symbol in asset_symbols:
                asset_states[symbol]['cumulative_realized_pnl_history'].append(
                    asset_states[symbol]['cumulative_realized_pnl']
                )

        self.asset_states = asset_states
        self.asset_symbols = asset_symbols
        self.portfolio_equity_curve = portfolio_equity
        self.returns = pd.Series(self.portfolio_equity_curve).pct_change().fillna(0)
        logger.info(f"Backtest complete. Trades: {[len(asset_states[s]['trades']) for s in asset_symbols]}")
        return asset_states, self.portfolio_equity_curve
