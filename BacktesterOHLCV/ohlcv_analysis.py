import numpy as np
import pandas as pd
import plotly.graph_objects as go
from debug import logger

class OHLCVAnalysis:
    def __init__(self, asset_states, portfolio_equity_curve, data, strategy=None):
        self.asset_states = asset_states  # dict: symbol -> state dict
        self.portfolio_equity_curve = portfolio_equity_curve  # list
        self.data = data
        self.strategy = strategy

    def compute_metrics(self):
        metrics = {}
        # Portfolio metrics
        metrics['portfolio'] = self._compute_curve_metrics(self.portfolio_equity_curve)
        # Per-asset metrics
        for symbol, state in self.asset_states.items():
            metrics[symbol] = self._compute_curve_metrics(state['equity'])
        return metrics

    def _compute_curve_metrics(self, curve):
        if not curve or curve[0] == 0 or len(curve) < 2:
            logger.warning("Equity curve is empty, too short, or starts at zero. Cannot compute return.")
            total_return = 0
            max_drawdown = 0
            sharpe = 0
        else:
            returns = pd.Series(curve).pct_change().fillna(0)
            total_return = curve[-1] / curve[0] - 1
            max_drawdown = self._max_drawdown(curve)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        return {'return': total_return, 'max_drawdown': max_drawdown, 'sharpe': sharpe}

    def _max_drawdown(self, curve):
        curve = np.array(curve)
        roll_max = np.maximum.accumulate(curve)
        drawdown = (curve - roll_max) / roll_max
        return drawdown.min()

    def plot_portfolio_equity(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.portfolio_equity_curve, mode='lines', name='Portfolio Equity'))
        fig.update_layout(title='Portfolio Equity Curve', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_asset_equity(self, symbol):
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=state['equity'], mode='lines', name=f'{symbol} Equity'))
        fig.update_layout(title=f'{symbol} Equity Curve', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_asset_trades(self, symbol):
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        trades = state['trades']
        # Get price series for the asset
        price_col = f'{symbol}_Close'
        if price_col not in self.data.columns:
            raise ValueError(f"Price column {price_col} not found in data.")
        price_series = self.data[price_col]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=price_series, mode='lines', name=f'{symbol} Price'))
        # Add buy/sell markers
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        buy_dates = [t.date for t in buy_trades]
        sell_dates = [t.date for t in sell_trades]
        buy_prices = [price_series[date] if date in price_series.index else None for date in buy_dates]
        sell_prices = [price_series[date] if date in price_series.index else None for date in sell_dates]
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'))
        fig.update_layout(title=f'{symbol} Price with Trades', xaxis_title='Date', yaxis_title='Price')
        fig.show()

    def plot_all_assets(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for symbol, state in self.asset_states.items():
            fig.add_trace(go.Scatter(x=self.data.index, y=state['equity'], mode='lines', name=f'{symbol}'))
        fig.update_layout(title='All Asset Equity Curves', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_portfolio_pnl(self):
        # PnL = equity - initial equity
        pnl = np.array(self.portfolio_equity_curve) - self.portfolio_equity_curve[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=pnl, mode='lines', name='Portfolio PnL'))
        fig.update_layout(title='Portfolio PnL', xaxis_title='Date', yaxis_title='PnL')
        fig.show()

    def plot_portfolio_allocations(self):
        # Sum cash and equity allocation across all assets at each time step
        n = len(self.portfolio_equity_curve)
        cash = np.zeros(n)
        equity = np.zeros(n)
        for state in self.asset_states.values():
            # Pad allocations if needed (in case of missing data for some assets)
            cash += np.pad(state['cash_allocation'], (0, n - len(state['cash_allocation'])), 'constant')
            equity += np.pad(state['equity_allocation'], (0, n - len(state['equity_allocation'])), 'constant')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=cash, mode='lines', name='Total Cash Allocation'))
        fig.add_trace(go.Scatter(x=self.data.index, y=equity, mode='lines', name='Total Equity Allocation'))
        fig.update_layout(title='Portfolio Cash and Equity Allocation', xaxis_title='Date', yaxis_title='Value')
        fig.show()

    def plot_portfolio_allocations_pct(self):
        # Sum cash and equity allocation across all assets at each time step
        n = len(self.portfolio_equity_curve)
        cash = np.zeros(n)
        equity = np.zeros(n)
        for state in self.asset_states.values():
            cash += np.pad(state['cash_allocation'], (0, n - len(state['cash_allocation'])), 'constant')
            equity += np.pad(state['equity_allocation'], (0, n - len(state['equity_allocation'])), 'constant')
        total = cash + equity
        # Avoid division by zero
        total[total == 0] = 1
        cash_pct = cash / total * 100
        equity_pct = equity / total * 100
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=cash_pct, mode='lines', name='Cash Allocation (%)'))
        fig.add_trace(go.Scatter(x=self.data.index, y=equity_pct, mode='lines', name='Equity Allocation (%)'))
        fig.update_layout(title='Portfolio Cash and Equity Allocation (%)', xaxis_title='Date', yaxis_title='Allocation (%)', yaxis=dict(range=[0, 100]))
        fig.show()

    def plot_asset_allocations(self, symbol):
        """Plot cash and equity allocation for a single asset."""
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        n = len(state['equity'])
        cash = np.array(state['cash_allocation'])
        equity = np.array(state['equity_allocation'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=cash, mode='lines', name=f'{symbol} Cash Allocation'))
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=equity, mode='lines', name=f'{symbol} Equity Allocation'))
        fig.update_layout(title=f'{symbol} Cash and Equity Allocation', xaxis_title='Date', yaxis_title='Value')
        fig.show()

    def plot_asset_equity_allocation_pct(self, symbol):
        """
        Plot the percentage of equity allocation for a single asset over time.
        """
        state = self.asset_states[symbol]
        n = len(state['equity'])
        cash = np.array(state['cash_allocation'])
        equity = np.array(state['equity_allocation'])
        total = cash + equity
        # Avoid division by zero
        total[total == 0] = 1
        cash_pct = cash / total * 100
        equity_pct = equity / total * 100
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=cash_pct, mode='lines', name=f'{symbol} Cash Allocation (%)'))
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=equity_pct, mode='lines', name=f'{symbol} Equity Allocation (%)'))
        fig.update_layout(title=f'{symbol} Cash and Equity Allocation (%)', xaxis_title='Date', yaxis_title='Allocation (%)', yaxis=dict(range=[0, 100]))
        fig.show()

    def plot_pnl_decomposition(self, top_n=5):
        """Plot bar chart of top and bottom N assets by total PnL (final equity - initial equity)."""
        import plotly.graph_objects as go
        pnls = {}
        for symbol, state in self.asset_states.items():
            if len(state['equity']) > 0:
                pnls[symbol] = state['equity'][-1] - state['equity'][0]
            else:
                pnls[symbol] = 0
        sorted_pnls = sorted(pnls.items(), key=lambda x: x[1], reverse=True)
        top = sorted_pnls[:top_n]
        bottom = sorted_pnls[-top_n:][::-1]
        labels = [s for s, _ in top] + [s for s, _ in bottom]
        values = [v for _, v in top] + [v for _, v in bottom]
        colors = ['green']*top_n + ['red']*top_n
        fig = go.Figure([go.Bar(x=labels, y=values, marker_color=colors)])
        fig.update_layout(title=f'Top {top_n} and Bottom {top_n} Asset PnL Contributions', xaxis_title='Asset', yaxis_title='Total PnL')
        fig.show()

    def print_portfolio_metrics(self, full=False):
        metrics = self.compute_metrics()['portfolio']
        pnls = []
        for state in self.asset_states.values():
            equity = state.get('equity', [])
            if len(equity) > 0:
                pnls.append(equity[-1] - equity[0])
        num_assets = len(self.asset_states)
        if pnls:
            avg_pnl = np.mean(pnls)
            var_pnl = np.var(pnls)
        else:
            avg_pnl = 0
            var_pnl = 0
        strat_name = getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy')) if self.strategy else 'Unknown Strategy'
        print(f"\nPortfolio metrics for strategy: {strat_name}")
        print(f"  Number of assets: {num_assets}")
        print(f"  Average Asset PnL: {avg_pnl:.2f}")
        print(f"  Variance of Asset PnL: {var_pnl:.2f}")
        if full:
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Total Return: {metrics['return']:.2%}")

    def plot_average_position_size(self):
        """Plot the average position size per asset over time."""
        n = len(self.portfolio_equity_curve)
        n_assets = len(self.asset_states)
        # Collect all positions per asset (pad with zeros if needed)
        positions_matrix = []
        for state in self.asset_states.values():
            pos = np.array(state['positions'])
            if len(pos) < n:
                pos = np.pad(pos, (0, n - len(pos)), 'constant')
            positions_matrix.append(pos)
        positions_matrix = np.array(positions_matrix)
        avg_position = positions_matrix.mean(axis=0)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=avg_position, mode='lines', name='Average Position Size'))
        fig.update_layout(title='Average Position Size per Asset Over Time', xaxis_title='Date', yaxis_title='Average Position Size')
        fig.show()

    @staticmethod
    def plot_rsi_indicator(analysis_or_strategy, asset_states=None, symbol=None, data=None):
        """
        Plot the RSI indicator evolution for a given asset, with buy/sell thresholds.
        Accepts either (strategy, asset_states, symbol, data) or (analysis, symbol).
        """
        import plotly.graph_objects as go
        # If called with an analysis object
        if asset_states is None and symbol is not None and hasattr(analysis_or_strategy, 'asset_states'):
            analysis = analysis_or_strategy
            strategy = analysis.strategy
            asset_states = analysis.asset_states
            data = analysis.data
        else:
            strategy = analysis_or_strategy
        state = asset_states[symbol]
        indicators = state.get('indicators', [])
        if indicators and isinstance(indicators[0], (tuple, list)):
            rsi_values = [v[0] for v in indicators]
        else:
            rsi_values = indicators
        buy_threshold = getattr(strategy, 'buyThreshold', 30)
        sell_threshold = getattr(strategy, 'sellThreshold', 70)
        strat_name = getattr(strategy, 'name', getattr(strategy, 'strat', 'Unknown Strategy'))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[:len(rsi_values)], y=rsi_values, mode='lines', name='RSI'))
        fig.add_trace(go.Scatter(x=data.index[:len(rsi_values)], y=[buy_threshold]*len(rsi_values), mode='lines', name='Buy Threshold', line=dict(dash='dash', color='green')))
        fig.add_trace(go.Scatter(x=data.index[:len(rsi_values)], y=[sell_threshold]*len(rsi_values), mode='lines', name='Sell Threshold', line=dict(dash='dash', color='red')))
        title = f'RSI Indicator Evolution for {symbol} ({strat_name})'
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='RSI')
        fig.show()

    @staticmethod
    def plot_moving_averages(analysis_or_strategy, asset_states=None, symbol=None, data=None):
        """
        Plot the short and long moving averages for a given asset.
        Accepts either (strategy, asset_states, symbol, data) or (analysis, symbol).
        """
        import plotly.graph_objects as go
        # If called with an analysis object
        if asset_states is None and symbol is not None and hasattr(analysis_or_strategy, 'asset_states'):
            analysis = analysis_or_strategy
            strategy = analysis.strategy
            asset_states = analysis.asset_states
            data = analysis.data
        else:
            strategy = analysis_or_strategy
        state = asset_states[symbol]
        indicators = state.get('indicators', [])
        if indicators and isinstance(indicators[0], (tuple, list)) and len(indicators[0]) >= 2:
            short_ma = [v[0] for v in indicators]
            long_ma = [v[1] for v in indicators]
        else:
            short_ma = indicators
            long_ma = [None]*len(indicators)
        strat_name = getattr(strategy, 'name', getattr(strategy, 'strat', 'Unknown Strategy'))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[:len(short_ma)], y=short_ma, mode='lines', name='Short MA'))
        fig.add_trace(go.Scatter(x=data.index[:len(long_ma)], y=long_ma, mode='lines', name='Long MA'))
        title = f'Moving Averages Evolution for {symbol} ({strat_name})'
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
        fig.show()

    @staticmethod
    def plot_multiple_portfolio_pnl(analyses_or_curves, data=None):
        """
        Plot the PnL of multiple strategies on the same plot.
        Accepts either a list of (portfolio_equity_curve, strategy) tuples and data,
        or a list of OHLCVAnalysis instances.
        If a list of OHLCVAnalysis is passed, data is not required.
        """
        import plotly.graph_objects as go
        fig = go.Figure()
        # If passed a list of analyses
        if all(hasattr(a, 'portfolio_equity_curve') for a in analyses_or_curves):
            for analysis in analyses_or_curves:
                pnl = np.array(analysis.portfolio_equity_curve) - analysis.portfolio_equity_curve[0]
                strat_name = getattr(analysis.strategy, 'name', getattr(analysis.strategy, 'strat', 'Unknown Strategy')) if analysis.strategy else 'Unknown Strategy'
                fig.add_trace(go.Scatter(x=analysis.data.index, y=pnl, mode='lines', name=strat_name))
        else:
            if data is None:
                raise ValueError("If passing (curve, strategy) tuples, you must also pass data (the DataFrame with the index)")
            for portfolio_equity_curve, strategy in analyses_or_curves:
                pnl = np.array(portfolio_equity_curve) - portfolio_equity_curve[0]
                strat_name = getattr(strategy, 'name', getattr(strategy, 'strat', 'Unknown Strategy'))
                fig.add_trace(go.Scatter(x=data.index, y=pnl, mode='lines', name=strat_name))
        fig.update_layout(title='Portfolio PnL Comparison', xaxis_title='Date', yaxis_title='PnL')
        fig.show()

    def plot_asset_pnl_with_trades(self, symbol):
        """Plot the PnL for a single asset with buy/sell trade markers."""
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        trades = state['trades']
        equity = np.array(state['equity'])
        pnl = equity - equity[0]
        n = len(equity)
        price_col = f'{symbol}_Close'
        if price_col not in self.data.columns:
            raise ValueError(f"Price column {price_col} not found in data.")
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        buy_dates = [t.date for t in buy_trades]
        sell_dates = [t.date for t in sell_trades]
        buy_pnl = [pnl[self.data.index.get_loc(t.date)] if t.date in self.data.index else None for t in buy_trades]
        sell_pnl = [pnl[self.data.index.get_loc(t.date)] if t.date in self.data.index else None for t in sell_trades]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=pnl, mode='lines', name=f'{symbol} PnL'))
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_pnl, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_pnl, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'))
        fig.update_layout(title=f'{symbol} PnL with Trades', xaxis_title='Date', yaxis_title='PnL')
        fig.show()

    def print_asset_report(self, symbol):
        """
        Print a report for a single asset including PnL, drawdown, number of signals, and number of trades executed.
        """
        state = self.asset_states[symbol]
        equity = np.array(state['equity'])
        if len(equity) == 0:
            print(f"No equity data for {symbol}.")
            return
        pnl = equity[-1] - equity[0]
        # Drawdown
        roll_max = np.maximum.accumulate(equity)
        drawdown = (equity - roll_max) / roll_max
        max_drawdown = drawdown.min()
        # Number of signals (assume 'indicators' contains signals or use trades as proxy)
        num_signals = len(state.get('signals', [])) if 'signals' in state else len(state.get('indicators', []))
        # Number of trades executed
        num_trades = len(state['trades'])
        strat_name = getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy')) if self.strategy else 'Unknown Strategy'
        print(f"\nReport for {symbol} ({strat_name}):")
        print(f"  PnL: {pnl:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Number of signals: {num_signals}")
        print(f"  Number of trades executed: {num_trades}")
