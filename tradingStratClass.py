import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from orderClass import orders

class trading_strat(ABC):
    def __init__(self, strat_name):
        self.strat = strat_name
        self.historical_trade = []
        self.historical_pnl = []
        self.inventory = {"price" : 0 , "quantity" : 0}
        self.__class__.order_out = {}
        self.__class__.orderID = 0

    @abstractmethod
    def strategy():
        return
    

class basicStrat(trading_strat):

    def __init__(self, name):
        super().__init__(name)

    def strategy(self, orderClass):

        targetBuy = 59800
        targetSell = 60000

        
        if orderClass.bids>=targetBuy:
            if self.inventory["quantity"] <= 10:
                price, quantity, status = orderClass.bids+1, -1, 0
                orderClass.send_order(self, price, -quantity)
                self.inventory["quantity"] += -quantity
                self.orderID +=1
        elif orderClass.asks<=targetSell:
            if self.inventory["quantity"] >= -10:
                price, quantity, status = orderClass.asks-1, 1, 0
                orderClass.send_order(self, price, -quantity)
                self.inventory["quantity"] += -quantity
                self.orderID +=1
        
        orderClass.filled_order(self)

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
            if self.inventory["quantity"] <= 0:  # Ensure no long position already exists
                price, quantity = current_price + 1, 1  # Buy one unit at slightly higher price
                orderClass.send_order(self, price, quantity)
                self.inventory["quantity"] += quantity
                self.orderID += 1
        elif short_ma < long_ma:  # Death Cross (Sell Signal)
            if self.inventory["quantity"] > 0:  # Ensure no short position already exists
                price, quantity = current_price - 1, -1  # Sell one unit at slightly lower price
                orderClass.send_order(self, price, quantity)
                self.inventory["quantity"] += quantity
                self.orderID += 1

        # Update filled orders
        orderClass.filled_order(self)

