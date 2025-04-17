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
        self.strategy.reset()
        signals, indicators = self.strategy.generate_signals(self.data)  # Get signals and indicators
        position = 0
        cash = initial_cash
        equity = [initial_cash]  # Start with initial cash
        realized_pnl = []  # Track realized PnL
        unrealized_pnl = []  # Track unrealized PnL
        cumulative_realized_pnl = 0  # Track cumulative realized PnL
        positions = []  # Track position size over time
        equity_allocation = []  # Track equity allocation over time
        cash_allocation = []  # Track cash allocation over time
        for i, row in self.data.iterrows():
            signal = signals[i]
            indicator = indicators[i]  # Get the indicator value for this step
            price = row['Close']
            date = row[self.data.columns[0]]

            if signal == 1 and position == 0:  # Buy all in
                quantity = int(cash // (price * (1 + self.fee_perc)))  # Account for fees
                fee = price * self.fee_perc * quantity
                self.trades.append(OHLCVOrder(date, price, quantity, 'buy', fee, indicator))
                position += quantity
                cash -= (price * quantity) + fee
                realized_pnl.append(cumulative_realized_pnl)  # No realized PnL on buy
                print(f"Step {i}, Date {date}: Trade Executed: BUY {quantity} units at {price}, Indicator={indicator}, Cash={cash}, Position={position}, Equity={cash + position * price}, Realized PnL={realized_pnl[-1]}")
            elif signal == -1 and position > 0:  # Sell all
                fee = price * self.fee_perc * position
                self.trades.append(OHLCVOrder(date, price, position, 'sell', fee, indicator))
                trade_pnl = (price * position) - (self.trades[-2].price * position) - fee
                cumulative_realized_pnl += trade_pnl
                realized_pnl.append(cumulative_realized_pnl)  # Realized PnL from the trade
                cash += (price * position) - fee
                print(f"Step {i}, Date {date}: Trade Executed: SELL {position} units at {price}, Indicator={indicator}, Cash={cash}, Position=0, Equity={cash}, Trade PnL={trade_pnl}, Realized PnL={realized_pnl[-1]}")
                position = 0
            else:
                realized_pnl.append(cumulative_realized_pnl)  # No realized PnL if no trade

            positions.append(position)  # Track the current position size
            unrealized_pnl.append(position * price)  # Unrealized PnL based on current position
            equity.append(cash + position * price)

            # Track allocations
            cash_allocation.append(cash)
            equity_allocation.append(position * price)

        self.equity_curve = equity
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.positions = positions  # Store positions in the backtester
        self.cash_allocation = cash_allocation  # Store cash allocation
        self.equity_allocation = equity_allocation  # Store equity allocation
        self.returns = pd.Series(equity).pct_change().fillna(0)
        logger.info(f"Backtest complete. Trades: {len(self.trades)}")
        return self.trades, self.equity_curve, self.unrealized_pnl, self.realized_pnl, self.positions, self.cash_allocation, self.equity_allocation
