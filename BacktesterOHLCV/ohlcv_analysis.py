import numpy as np
import pandas as pd
import plotly.graph_objects as go
from debug import logger

class OHLCVAnalysis:
    def __init__(self, equity_curve, trades, data, realized_pnl, unrealized_pnl, positions, cash_allocation, equity_allocation):
        self.equity_curve = equity_curve
        self.trades = trades
        self.data = data
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.positions = positions
        self.cash_allocation = cash_allocation  # Use cash allocation time series
        self.equity_allocation = equity_allocation

    def compute_metrics(self):
        if not self.equity_curve or self.equity_curve[0] == 0 or len(self.equity_curve) < 2:
            logger.warning("Equity curve is empty, too short, or starts at zero. Cannot compute return.")
            total_return = 0
            max_drawdown = 0
            sharpe = 0
        else:
            returns = pd.Series(self.equity_curve).pct_change().fillna(0)
            total_return = self.equity_curve[-1] / self.equity_curve[0] - 1
            max_drawdown = self._max_drawdown(self.equity_curve)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            logger.info(f"Return: {total_return:.2%}, Max Drawdown: {max_drawdown:.2%}, Sharpe: {sharpe:.2f}")
        return {'return': total_return, 'max_drawdown': max_drawdown, 'sharpe': sharpe}

    def _max_drawdown(self, curve):
        curve = np.array(curve)
        roll_max = np.maximum.accumulate(curve)
        drawdown = (curve - roll_max) / roll_max
        return drawdown.min()

    def plot(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[self.data.columns[0]], y=self.equity_curve, mode='lines', name='Equity Curve'))

        # Add buy trades
        buy_trades = [trade for trade in self.trades if trade.side == 'buy']
        buy_dates = [trade.date for trade in buy_trades]
        buy_equity = [self.equity_curve[self.data[self.data.columns[0]].tolist().index(trade.date)] for trade in buy_trades]
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_equity,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ))

        # Add sell trades
        sell_trades = [trade for trade in self.trades if trade.side == 'sell']
        sell_dates = [trade.date for trade in sell_trades]
        sell_equity = [self.equity_curve[self.data[self.data.columns[0]].tolist().index(trade.date)] for trade in sell_trades]
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_equity,
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ))

        fig.update_layout(title='Equity Curve with Trades', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_with_signals(self):
        import plotly.graph_objects as go

        # Create a figure for stock price and signals
        fig = go.Figure()

        # Add stock price line
        fig.add_trace(go.Scatter(
            x=self.data[self.data.columns[0]],
            y=self.data['Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='blue')
        ))

        # Add buy signals
        buy_trades = [trade for trade in self.trades if trade.side == 'buy']
        buy_dates = [trade.date for trade in buy_trades]
        buy_prices = [trade.price for trade in buy_trades]
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ))

        # Add sell signals
        sell_trades = [trade for trade in self.trades if trade.side == 'sell']
        sell_dates = [trade.date for trade in sell_trades]
        sell_prices = [trade.price for trade in sell_trades]
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ))

        # Update layout
        fig.update_layout(
            title='Stock Price with Buy/Sell Signals',
            xaxis_title='Date',
            yaxis_title='Price'
        )

        fig.show()

    def plot_pnl(self):
        import plotly.graph_objects as go

        # Create a figure for PnL
        fig = go.Figure()

        # Calculate total PnL
        total_pnl = [realized + unrealized for realized, unrealized in zip(self.realized_pnl, self.unrealized_pnl)]

        # Add total PnL line
        fig.add_trace(go.Scatter(
            x=self.data[self.data.columns[0]],
            y=total_pnl,
            mode='lines',
            name='Total PnL',
            line=dict(color='blue')
        ))

        # Update layout
        fig.update_layout(
            title='Total PnL',
            xaxis_title='Date',
            yaxis_title='PnL'
        )

        fig.show()

    def plot_positions(self):
        import plotly.graph_objects as go

        # Create a figure for positions
        fig = go.Figure()

        # Add position line
        fig.add_trace(go.Scatter(
            x=self.data[self.data.columns[0]],
            y=self.positions,
            mode='lines',
            name='Positions',
            line=dict(color='purple')
        ))

        # Update layout
        fig.update_layout(
            title='Positions Over Time',
            xaxis_title='Date',
            yaxis_title='Position Size'
        )

        fig.show()

    def plot_allocations(self):
        import plotly.graph_objects as go

        # Create a figure for allocations
        fig = go.Figure()

        # Add equity allocation line
        fig.add_trace(go.Scatter(
            x=self.data[self.data.columns[0]],
            y=self.equity_allocation,
            mode='lines',
            name='Equity Allocation',
            line=dict(color='green')
        ))

        # Add cash allocation line
        fig.add_trace(go.Scatter(
            x=self.data[self.data.columns[0]],
            y=self.cash_allocation,  # Use cash allocation time series
            mode='lines',
            name='Cash Allocation',
            line=dict(color='blue')
        ))

        # Update layout
        fig.update_layout(
            title='Cash and Equity Allocations Over Time',
            xaxis_title='Date',
            yaxis_title='Allocation'
        )

        fig.show()
