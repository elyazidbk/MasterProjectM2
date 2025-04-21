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
        # Identify asset symbols from columns (e.g., AAPL_Close, MSFT_Close)
        asset_symbols = sorted(list(set([col.split('_')[0] for col in self.data.columns if col.endswith('_Close')])))
        n_assets = len(asset_symbols)
        cash_per_asset = initial_cash / n_assets

        # Initialize per-asset state
        asset_states = {
            symbol: {
                'cash': cash_per_asset,
                'position': 0,
                'trades': [],
                'equity': [cash_per_asset],
                'realized_pnl': [],
                'unrealized_pnl': [],
                'cumulative_realized_pnl': 0,
                'positions': [],
                'equity_allocation': [],
                'cash_allocation': []
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
            asset_signals[symbol] = sig
            asset_indicators[symbol] = ind
            # Store indicators in asset_states for later analysis
            asset_states[symbol]['indicators'] = ind

        # Main backtest loop
        for idx, (i, row) in enumerate(self.data.iterrows()):
            total_equity = 0
            for symbol in asset_symbols:
                state = asset_states[symbol]
                # Use Adjusted Close if available, otherwise fallback to Close
                price_col = f'{symbol}_Adj Close' if f'{symbol}_Adj Close' in self.data.columns else f'{symbol}_Close'
                if price_col not in row or pd.isna(row[price_col]):
                    state['realized_pnl'].append(state['cumulative_realized_pnl'])
                    state['positions'].append(state['position'])
                    state['unrealized_pnl'].append(state['position'] * 0)
                    state['equity'].append(state['cash'])
                    state['cash_allocation'].append(state['cash'])
                    state['equity_allocation'].append(0)
                    continue
                price = row[price_col]
                signal = asset_signals[symbol][idx]
                indicator = asset_indicators[symbol][idx]
                date = row.name if self.data.index.name else idx

                if signal == 1 and state['position'] == 0:
                    max_invest = state['cash']
                    cash_quantity = int(max_invest // (price * (1 + self.fee_perc)))
                    volume_col = f'{symbol}_Volume'
                    if volume_col in row and not pd.isna(row[volume_col]):
                        max_volume = int(row[volume_col] * 0.1)
                    else:
                        max_volume = float('inf')  # If volume is missing, only cash constraint applies
                    quantity = min(cash_quantity, max_volume)
                    if quantity > 0:
                        fee = price * self.fee_perc * quantity
                        state['trades'].append(OHLCVOrder(date, price, quantity, 'buy', fee, indicator))
                        state['position'] += quantity
                        state['cash'] -= (price * quantity) + fee
                    state['realized_pnl'].append(state['cumulative_realized_pnl'])
                elif signal == -1 and state['position'] > 0:
                    fee = price * self.fee_perc * state['position']
                    state['trades'].append(OHLCVOrder(date, price, state['position'], 'sell', fee, indicator))
                    last_buy_price = state['trades'][-2].price if len(state['trades']) >= 2 else price
                    trade_pnl = (price * state['position']) - (last_buy_price * state['position']) - fee
                    state['cumulative_realized_pnl'] += trade_pnl
                    state['realized_pnl'].append(state['cumulative_realized_pnl'])
                    state['cash'] += (price * state['position']) - fee
                    state['position'] = 0
                else:
                    state['realized_pnl'].append(state['cumulative_realized_pnl'])
                state['positions'].append(state['position'])
                state['unrealized_pnl'].append(state['position'] * price)
                state['equity'].append(state['cash'] + state['position'] * price)
                state['cash_allocation'].append(state['cash'])
                state['equity_allocation'].append(state['position'] * price)
                total_equity += state['cash'] + state['position'] * price
            # Optionally, store total_equity for portfolio curve
            # ...

        # Aggregate results
        self.asset_states = asset_states
        self.asset_symbols = asset_symbols
        self.portfolio_equity_curve = [sum(asset_states[s]['equity'][i] for s in asset_symbols) for i in range(len(self.data)+1)]
        self.returns = pd.Series(self.portfolio_equity_curve).pct_change().fillna(0)
        logger.info(f"Backtest complete. Trades: {[len(asset_states[s]['trades']) for s in asset_symbols]}")
        return asset_states, self.portfolio_equity_curve
