import pandas as pd
import numpy as np
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import trading_strat
from debug import logger


class MovingAverageStrat(trading_strat):
    def __init__(self, name, short_window, long_window):
        super().__init__(name)
        self.short_window = short_window  # Number of periods for the short moving average
        self.long_window = long_window    # Number of periods for the long moving average
        self.prices = []                  # Store historical prices to calculate moving averages
    def calculate_moving_averages(self):
        # Ensure there are enough prices to calculate the moving averages
        if len(self.prices) >= self.long_window:
            short_ma = np.mean(self.prices[-self.short_window:])
            long_ma = np.mean(self.prices[-self.long_window:])
            return short_ma, long_ma
        return None, None
    def strategy(self, orderClass):
        # Append the latest market price to the price history
        current_price = orderClass.mid_price  # Assuming `orderClass` provides the mid-price
        self.prices.append(current_price)
        # Ensure we have enough data to calculate moving averages
        short_ma, long_ma = self.calculate_moving_averages()
        if short_ma is None or long_ma is None:
            return  # Not enough data yet
         
        # Implement Dual Moving Average Crossover Strategy
        if short_ma > long_ma:  # Golden Cross (Buy Signal)
            if self.inventory["quantity"] <= 5:  # Ensure no long position above 6
                price, quantity = current_price + 1, 1  # Buy one unit at slightly higher price
                orderClass.send_order(self, price, quantity)
                self.inventory["quantity"] += quantity
                self.orderID += 1
            else:
                orderToCancel = list(self.order_out.keys())
                for id in orderToCancel:
                    orderClass.cancel_order(self, id)

        elif short_ma < long_ma:  # Death Cross (Sell Signal)
            if self.inventory["quantity"] >= -5:  # Ensure no short position below 6
                price, quantity = current_price - 1, -1  # Sell one unit at slightly lower price
                orderClass.send_order(self, price, quantity)
                self.inventory["quantity"] += quantity
                self.orderID += 1
            else:
                orderToCancel = list(self.order_out.keys())
                for id in orderToCancel:
                    orderClass.cancel_order(self, id)
        # Update filled orders
        orderClass.filled_order(self)