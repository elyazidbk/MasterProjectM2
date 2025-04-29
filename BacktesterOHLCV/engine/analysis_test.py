import numpy as np
import pandas as pd
import plotly.graph_objects as go
from debug import logger

class AssetTradeAnalysis:
    def __init__(self, asset_states, data, strategy=None):
        self.asset_states = asset_states
        self.data = data
        self.strategy = strategy

    def plot_asset_trades(self, symbol):
        state = self.asset_states[symbol]
        price_col = f'{symbol}_Adjusted Close'
        if price_col not in self.data.columns:
            print(f"No price data for {symbol}.")
            return
        prices = self.data[price_col].values
        trades = state['trades']
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for t in trades:
            if t.side == 'buy':
                buy_x.append(t.date)
                buy_y.append(t.price)
            elif t.side == 'sell':
                sell_x.append(t.date)
                sell_y.append(t.price)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(prices)], y=prices, mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'))
        fig.update_layout(title=f'{symbol} Price with Trades', xaxis_title='Date', yaxis_title='Price')
        fig.show()

    def plot_asset_pnl_with_trades(self, symbol):
        state = self.asset_states[symbol]
        value = np.array(state.get('portfolio_value', []))
        if len(value) == 0:
            print(f"No portfolio value to plot for {symbol}.")
            return
        pnl_curve = value - value[0]
        trades = state['trades']
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for t in trades:
            if isinstance(t.date, (int, np.integer)):
                idx = t.date
            elif t.date in self.data.index:
                idx = self.data.index.get_loc(t.date)
            else:
                idx = None
            if idx is not None and idx < len(pnl_curve):
                if t.side == 'buy':
                    buy_x.append(t.date)
                    buy_y.append(pnl_curve[idx])
                elif t.side == 'sell':
                    sell_x.append(t.date)
                    sell_y.append(pnl_curve[idx])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(pnl_curve)], y=pnl_curve, mode='lines', name='PnL'))
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'))
        fig.update_layout(title=f'{symbol} PnL with Trades', xaxis_title='Date', yaxis_title='PnL')
        fig.show()

    def plot_asset_position(self, symbol):
        state = self.asset_states[symbol]
        positions = state.get('positions', [])
        if not positions:
            print(f"No position data for {symbol}.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(positions)], y=positions, mode='lines', name='Position'))
        fig.update_layout(title=f'{symbol} Position Over Time', xaxis_title='Date', yaxis_title='Position Size')
        fig.show()