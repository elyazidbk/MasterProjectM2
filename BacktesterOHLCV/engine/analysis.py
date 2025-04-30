import numpy as np
import pandas as pd
import plotly.graph_objects as go
from engine.debug import logger
from scipy import stats


class OHLCVAnalysis:
    def __init__(self, asset_states, portfolio_equity_curve, data, strategy=None):
        self.asset_states = asset_states
        self.portfolio_equity_curve = portfolio_equity_curve
        self.data = data
        self.strategy = strategy

    def compute_metrics(self):
        metrics = {}
        metrics['portfolio'] = self._compute_curve_metrics(self.portfolio_equity_curve)
        for symbol, state in self.asset_states.items():
            equity = np.array(state.get('portfolio_value', state.get('equity', [])))
            if np.count_nonzero(equity) < 2:
                continue
            nonzero_idx = np.argmax(equity != 0) if np.any(equity != 0) else 0
            trimmed_equity = equity[nonzero_idx:] if len(equity) > 0 else equity
            metrics[symbol] = self._compute_curve_metrics(trimmed_equity)
        return metrics

    def _compute_curve_metrics(self, curve):
        curve = np.array(curve)
        valid_curve = curve[curve > 1]
        if len(valid_curve) < 2 or np.all(valid_curve == 0):
            logger.warning("Equity curve is empty, too short, or all zero. Cannot compute return.")
            return {
                'return': 0,
                'max_drawdown': 0,
                'sharpe': 0,
                'annualized_return': 0
            }
        returns = pd.Series(valid_curve).pct_change().fillna(0)
        total_return = valid_curve[-1] / valid_curve[0] - 1 if valid_curve[0] != 0 else 0
        max_roll = np.maximum.accumulate(valid_curve)
        drawdown = (valid_curve - max_roll) / max_roll
        max_drawdown = drawdown.min()
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        num_years = len(valid_curve) / 252
        annualized_return = (valid_curve[-1] / valid_curve[0])**(1/num_years) - 1 if num_years > 0 else 0

        return {
            'return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'annualized_return': annualized_return
        }

    def _max_drawdown(self, curve):
        curve = np.array(curve)
        if len(curve) == 0 or np.all(curve == 0):
            return 0
        roll_max = np.maximum.accumulate(curve)
        roll_max[roll_max == 0] = 1
        drawdown = (curve - roll_max) / roll_max
        return drawdown.min()

    def print_portfolio_metrics(self, full=False):
        metrics = self.compute_metrics()['portfolio']
        pnls = []
        for state in self.asset_states.values():
            equity = state.get('equity', [])
            if len(equity) > 0:
                pnls.append(equity[-1] - equity[0])
        pnls = np.array(pnls)
        num_assets = len(self.asset_states)
        avg_pnl = np.mean(pnls) if len(pnls) > 0 else 0
        var_pnl = np.var(pnls) if len(pnls) > 0 else 0
        strat_name = getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy')) if self.strategy else 'Unknown Strategy'
        print(f"\nPortfolio metrics for strategy: {strat_name}")
        print(f"  Number of assets: {num_assets}")
        print(f"  Average Asset PnL: {avg_pnl:.2f}")
        print(f"  Variance of Asset PnL: {var_pnl:.2f}")
        if full:
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Total Return: {metrics['return']:.2%}")
            print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
            std_pnl = np.std(pnls)
            skew = stats.skew(pnls)
            kurt = stats.kurtosis(pnls)
            outliers = pnls[np.abs(stats.zscore(pnls)) > 2.5]
            pos_pnls = pnls[pnls > 0]
            neg_pnls = pnls[pnls <= 0]
            print(f"  PnL Std Dev: {std_pnl:.2f}")
            print(f"  Skewness: {skew:.2f}")
            print(f"  Kurtosis: {kurt:.2f}")
            print(f"  Positive PnLs: {len(pos_pnls)} | Avg: {np.mean(pos_pnls):.2f}")
            print(f"  Negative PnLs: {len(neg_pnls)} | Avg: {np.mean(neg_pnls):.2f}")
            print(f"  Outliers (|z| > 2.5): {len(outliers)}")

    def print_asset_report(self, symbol):
        state = self.asset_states[symbol]
        portfolio_value = np.array(state.get('portfolio_value', []))
        # Filter to only meaningful capital periods
        portfolio_value = portfolio_value[portfolio_value > 1]
        if len(portfolio_value) < 2:
            print(f"No usable portfolio value data for {symbol}.")
            return
        pnl = portfolio_value[-1] - portfolio_value[0]
        roll_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - roll_max) / np.where(roll_max != 0, roll_max, 1)
        max_dd_pct = drawdown.min()
        max_dd_abs = np.max(roll_max - portfolio_value)

        signals = state.get('signals', [])
        num_signals = sum(1 for s in signals if s in (1, -1))
        num_trades = len(state['trades'])
        strat_name = getattr(self.strategy, 'name', getattr(self.strategy, 'strat', 'Unknown Strategy')) if self.strategy else 'Unknown Strategy'

        print(f"\nReport for {symbol} ({strat_name}):")
        print(f"  PnL: {pnl:.2f}")
        print(f"  Max Drawdown: {max_dd_abs:.2f} (abs), {max_dd_pct:.2%} (pct)")
        print(f"  Number of signals: {num_signals}")
        print(f"  Number of trades executed: {num_trades}")

    def plot_portfolio_equity(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(self.portfolio_equity_curve)], y=self.portfolio_equity_curve, mode='lines', name='Portfolio Equity'))
        fig.update_layout(title='Portfolio Equity Curve', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_portfolio_allocations(self):
        n = len(self.portfolio_equity_curve)
        cash = np.zeros(n)
        equity = np.zeros(n)
        for state in self.asset_states.values():
            equity += np.pad(state['equity_allocation'], (0, n - len(state['equity_allocation'])), 'constant')
        # Portfolio cash is the difference between total portfolio equity and sum of all asset equities
        for i in range(n):
            cash[i] = self.portfolio_equity_curve[i] - equity[i]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=cash, mode='lines', name='Total Cash Allocation'))
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=equity, mode='lines', name='Total Equity Allocation'))
        fig.update_layout(title='Portfolio Cash and Equity Allocation', xaxis_title='Date', yaxis_title='Value')
        fig.show()

    def plot_portfolio_allocations_pct(self):
        n = len(self.portfolio_equity_curve)
        cash = np.zeros(n)
        equity = np.zeros(n)
        for state in self.asset_states.values():
            equity += np.pad(state['equity_allocation'], (0, n - len(state['equity_allocation'])), 'constant')
        for i in range(n):
            cash[i] = self.portfolio_equity_curve[i] - equity[i]
        total = cash + equity
        total[total == 0] = 1
        cash_pct = cash / total * 100
        equity_pct = equity / total * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=cash_pct, mode='lines', name='Cash Allocation (%)'))
        fig.add_trace(go.Scatter(x=self.data.index[:n], y=equity_pct, mode='lines', name='Equity Allocation (%)'))
        fig.update_layout(title='Portfolio Cash and Equity Allocation (%)', xaxis_title='Date', yaxis_title='Allocation (%)', yaxis=dict(range=[0, 100]))
        fig.show()

    @staticmethod
    def plot_multiple_portfolio_pnl(analyses):
        import plotly.graph_objects as go
        fig = go.Figure()
        for analysis in analyses:
            strat_name = getattr(analysis.strategy, 'name', getattr(analysis.strategy, 'strat', 'Unknown Strategy')) if analysis.strategy else 'Unknown Strategy'
            fig.add_trace(go.Scatter(x=analysis.data.index[:len(analysis.portfolio_equity_curve)], y=analysis.portfolio_equity_curve, mode='lines', name=strat_name))
        fig.update_layout(title='Portfolio PnL Comparison', xaxis_title='Date', yaxis_title='Portfolio Equity')
        fig.show()

    def plot_average_position_size(self):
        import plotly.graph_objects as go
        avg_positions = {}
        for symbol, state in self.asset_states.items():
            positions = np.array(state['positions'])
            avg_positions[symbol] = np.mean(positions) if len(positions) > 0 else 0
        symbols = list(avg_positions.keys())
        avg_vals = list(avg_positions.values())
        fig = go.Figure([go.Bar(x=symbols, y=avg_vals)])
        fig.update_layout(title='Average Position Size per Asset', xaxis_title='Asset', yaxis_title='Average Position Size')
        fig.show()

    def plot_asset_trades(self, symbol):
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        price_col = f'{symbol}_Adjusted Close'
        if price_col not in self.data.columns:
            print(f"No price data for {symbol}.")
            return
        prices = self.data[price_col].values
        trades = state['trades']
        # Separate by trade_type and side
        buy_signal_x, buy_signal_y, buy_reb_x, buy_reb_y = [], [], [], []
        sell_signal_x, sell_signal_y, sell_reb_x, sell_reb_y = [], [], [], []
        for t in trades:
            if t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'signal':
                buy_signal_x.append(t.date)
                buy_signal_y.append(t.price)
            elif t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                buy_reb_x.append(t.date)
                buy_reb_y.append(t.price)
            elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'signal':
                sell_signal_x.append(t.date)
                sell_signal_y.append(t.price)
            elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                sell_reb_x.append(t.date)
                sell_reb_y.append(t.price)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(prices)], y=prices, mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=buy_signal_x, y=buy_signal_y, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy (Signal)'))
        fig.add_trace(go.Scatter(x=buy_reb_x, y=buy_reb_y, mode='markers', marker=dict(color='lightblue', symbol='triangle-up', size=10), name='Buy (Rebalance)'))
        fig.add_trace(go.Scatter(x=sell_signal_x, y=sell_signal_y, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell (Signal)'))
        fig.add_trace(go.Scatter(x=sell_reb_x, y=sell_reb_y, mode='markers', marker=dict(color='orange', symbol='triangle-down', size=10), name='Sell (Rebalance)'))
        fig.update_layout(title=f'{symbol} Price with Trades by Type', xaxis_title='Date', yaxis_title='Price')
        fig.show()

    def plot_asset_pnl_with_trades(self, symbol):
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        value = np.array(state.get('portfolio_value', []))
        if len(value) == 0:
            print(f"No portfolio value to plot for {symbol}.")
            return
        pnl_curve = value - value[0]
        trades = state['trades']
        # Separate by trade_type and side
        buy_signal_x, buy_signal_y, buy_reb_x, buy_reb_y = [], [], [], []
        sell_signal_x, sell_signal_y, sell_reb_x, sell_reb_y = [], [], [], []
        for t in trades:
            if isinstance(t.date, (int, np.integer)):
                idx = t.date
            elif t.date in self.data.index:
                idx = self.data.index.get_loc(t.date)
            else:
                idx = None
            if idx is not None and idx < len(pnl_curve):
                if t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'signal':
                    buy_signal_x.append(t.date)
                    buy_signal_y.append(pnl_curve[idx])
                elif t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                    buy_reb_x.append(t.date)
                    buy_reb_y.append(pnl_curve[idx])
                elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'signal':
                    sell_signal_x.append(t.date)
                    sell_signal_y.append(pnl_curve[idx])
                elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                    sell_reb_x.append(t.date)
                    sell_reb_y.append(pnl_curve[idx])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(pnl_curve)], y=pnl_curve, mode='lines', name='PnL'))
        fig.add_trace(go.Scatter(x=buy_signal_x, y=buy_signal_y, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy (Signal)'))
        fig.add_trace(go.Scatter(x=buy_reb_x, y=buy_reb_y, mode='markers', marker=dict(color='lightblue', symbol='triangle-up', size=10), name='Buy (Rebalance)'))
        fig.add_trace(go.Scatter(x=sell_signal_x, y=sell_signal_y, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell (Signal)'))
        fig.add_trace(go.Scatter(x=sell_reb_x, y=sell_reb_y, mode='markers', marker=dict(color='orange', symbol='triangle-down', size=10), name='Sell (Rebalance)'))
        fig.update_layout(title=f'{symbol} PnL with Trades by Type', xaxis_title='Date', yaxis_title='PnL')
        fig.show()

    def plot_asset_drawdown(self, symbol):
        import numpy as np
        import plotly.graph_objects as go
        state = self.asset_states.get(symbol)
        if state is None:
            print(f"No state found for {symbol}.")
            return
        value = np.array(state.get('portfolio_value', []))
        if value.size == 0:
            print(f"No portfolio value data for {symbol}.")
            return
        # compute rolling peak and percentage drawdown
        peak = np.maximum.accumulate(value)
        # avoid division by zero
        peak[peak == 0] = 1
        drawdown = (value - peak) / peak * 100
        dates = self.data.index[:len(value)]
        fig = go.Figure([go.Scatter(x=dates, y=drawdown, mode='lines', name='Drawdown (%)')])
        fig.update_layout(
            title=f'{symbol} Drawdown Curve (%)',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)'
        )
        fig.show()

    def plot_asset_allocations(self, symbol):
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        equity = state['equity_allocation']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(equity)], y=equity, mode='lines', name='Equity Allocation'))
        fig.update_layout(title=f'{symbol} Equity Allocation Over Time', xaxis_title='Date', yaxis_title='Equity Allocation')
        fig.show()

    def plot_top_contributors(self):
        import plotly.graph_objects as go

        # Extract cumulative realized PnL for all assets
        pnl_contributions = {
            asset: state['cumulative_realized_pnl']
            for asset, state in self.asset_states.items()
        }

        # Sort assets by PnL contribution
        sorted_contributors = sorted(pnl_contributions.items(), key=lambda x: x[1], reverse=True)

        # Top 5 positive contributors
        top_positive = sorted_contributors[:5]

        # Top 5 negative contributors
        top_negative = sorted_contributors[-5:]

        # Combine for plotting
        contributors = top_positive + top_negative
        assets, pnl_values = zip(*contributors)

        # Create bar plot
        fig = go.Figure([go.Bar(x=assets, y=pnl_values)])
        fig.update_layout(
            title=f'Top 5 Positive and Negative Contributors to PnL',
            xaxis_title='Assets',
            yaxis_title='Cumulative Realized PnL'
        )
        fig.show()

    def decompose_trades(self):
        # Decompose trades into signal and rebalance types for each asset
        decomposition = {}
        for symbol, state in self.asset_states.items():
            trades = state.get('trades', [])
            signal_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'signal']
            rebalance_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'rebalance']
            decomposition[symbol] = {
                'signal_trades': signal_trades,
                'rebalance_trades': rebalance_trades,
                'all_trades': trades
            }
        return decomposition

    def compute_trade_type_metrics(self):
        # Compute PnL, turnover, and count for each trade type
        results = {}
        for symbol, state in self.asset_states.items():
            trades = state.get('trades', [])
            signal_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'signal']
            rebalance_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'rebalance']
            def trade_pnl(trades):
                pnl = 0
                for t in trades:
                    if t.side == 'sell':
                        pnl += t.price * t.quantity - t.fee
                    elif t.side == 'buy':
                        pnl -= t.price * t.quantity + t.fee
                return pnl
            results[symbol] = {
                'signal': {
                    'pnl': trade_pnl(signal_trades),
                    'turnover': sum(abs(t.price * t.quantity) for t in signal_trades),
                    'count': len(signal_trades)
                },
                'rebalance': {
                    'pnl': trade_pnl(rebalance_trades),
                    'turnover': sum(abs(t.price * t.quantity) for t in rebalance_trades),
                    'count': len(rebalance_trades)
                },
                'all': {
                    'pnl': trade_pnl(trades),
                    'turnover': sum(abs(t.price * t.quantity) for t in trades),
                    'count': len(trades)
                }
            }
        return results

    def plot_trade_type_equity_curves(self, symbol):
        # Plot cumulative equity curves for signal and rebalance trades separately
        state = self.asset_states[symbol]
        trades = state.get('trades', [])
        signal_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'signal']
        rebalance_trades = [t for t in trades if getattr(t, 'trade_type', 'signal') == 'rebalance']
        dates = list(self.data.index)
        equity_signal = np.zeros(len(dates))
        equity_rebalance = np.zeros(len(dates))
        equity_agg = np.zeros(len(dates))
        pos_signal = 0
        pos_rebalance = 0
        cash_signal = 0
        cash_rebalance = 0
        cash_agg = 0
        for i, d in enumerate(dates):
            # Signal trades
            for t in signal_trades:
                if t.date == d:
                    if t.side == 'buy':
                        pos_signal += t.quantity
                        cash_signal -= t.price * t.quantity + t.fee
                    elif t.side == 'sell':
                        pos_signal -= t.quantity
                        cash_signal += t.price * t.quantity - t.fee
            # Rebalance trades
            for t in rebalance_trades:
                if t.date == d:
                    if t.side == 'buy':
                        pos_rebalance += t.quantity
                        cash_rebalance -= t.price * t.quantity + t.fee
                    elif t.side == 'sell':
                        pos_rebalance -= t.quantity
                        cash_rebalance += t.price * t.quantity - t.fee
            price_col = f'{symbol}_Adjusted Close'
            price = self.data[price_col][d] if price_col in self.data.columns and d in self.data.index else 0
            equity_signal[i] = cash_signal + pos_signal * price
            equity_rebalance[i] = cash_rebalance + pos_rebalance * price
            equity_agg[i] = equity_signal[i] + equity_rebalance[i]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity_signal, mode='lines', name='Signal Trades Equity', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=equity_rebalance, mode='lines', name='Rebalance Trades Equity', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=dates, y=equity_agg, mode='lines', name='Aggregate Equity', line=dict(color='black', dash='dash')))
        fig.update_layout(title=f'{symbol} Equity Curves by Trade Type', xaxis_title='Date', yaxis_title='Equity')
        fig.show()

    def plot_asset_trades_colored(self, symbol):
        # Plot trades with color by trade_type
        import plotly.graph_objects as go
        state = self.asset_states[symbol]
        price_col = f'{symbol}_Adjusted Close'
        if price_col not in self.data.columns:
            print(f"No price data for {symbol}.")
            return
        prices = self.data[price_col].values
        trades = state['trades']
        buy_signal_x, buy_signal_y, buy_reb_x, buy_reb_y = [], [], [], []
        sell_signal_x, sell_signal_y, sell_reb_x, sell_reb_y = [], [], [], []
        for t in trades:
            if t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'signal':
                buy_signal_x.append(t.date)
                buy_signal_y.append(t.price)
            elif t.side == 'buy' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                buy_reb_x.append(t.date)
                buy_reb_y.append(t.price)
            elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'signal':
                sell_signal_x.append(t.date)
                sell_signal_y.append(t.price)
            elif t.side == 'sell' and getattr(t, 'trade_type', 'signal') == 'rebalance':
                sell_reb_x.append(t.date)
                sell_reb_y.append(t.price)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[:len(prices)], y=prices, mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=buy_signal_x, y=buy_signal_y, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy (Signal)'))
        fig.add_trace(go.Scatter(x=buy_reb_x, y=buy_reb_y, mode='markers', marker=dict(color='lightblue', symbol='triangle-up', size=10), name='Buy (Rebalance)'))
        fig.add_trace(go.Scatter(x=sell_signal_x, y=sell_signal_y, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell (Signal)'))
        fig.add_trace(go.Scatter(x=sell_reb_x, y=sell_reb_y, mode='markers', marker=dict(color='orange', symbol='triangle-down', size=10), name='Sell (Rebalance)'))
        fig.update_layout(title=f'{symbol} Price with Trades by Type', xaxis_title='Date', yaxis_title='Price')
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

