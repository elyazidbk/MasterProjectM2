import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import trading_strat
from BacktesterOHLCV.debug import logger

MAX_INVENT = 5

class movingAverageStrat(trading_strat):

    def __init__(self, name, short_window, long_window):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.prices = deque(maxlen=self.long_window) # Store only up to `long_window` prices
        self.short_sum = 0
        self.long_sum = 0
        self.historical_short_ma = []
        self.historical_long_ma = []

    def calculate_moving_averages(self):
        new_price = OBData.mid()
        self.prices.append(new_price)

        if len(self.prices) >= self.short_window:
            if len(self.prices) == self.short_window:
                self.short_sum += new_price - self.prices[0]
            else:
                self.short_sum += new_price - self.prices[-self.short_window]

            if len(self.prices) >= self.long_window:

                self.long_sum += new_price - self.prices[0] 

                # Calculate the moving averages
                short_ma = self.short_sum / self.short_window
                long_ma = self.long_sum / self.long_window
    
                # Append to historical data
                self.historical_short_ma.append(short_ma)
                self.historical_long_ma.append(long_ma)

                return short_ma, long_ma

            else:
                self.long_sum += new_price
                self.historical_long_ma.append(None)
                self.historical_short_ma.append(None)
                return None, None
            
        else:
            self.short_sum += new_price
            self.long_sum += new_price
            self.historical_long_ma.append(None)
            self.historical_short_ma.append(None)
            return None, None


        
    def strategy(self, orderClass):
        
        short_ma, long_ma = self.calculate_moving_averages()

        
        if short_ma is None or long_ma is None:
            pass
        else:
            # Append the latest market price to the price history
            current_price = OBData.mid() # Assuming `orderClass` provides the mid-price
            # Ensure we have enough data to calculate moving averages

            buyOrderOut = [id for id, trade in self.order_out.items() 
                        if trade[orders.orderIndex["quantity"]] > 0]

            sellOrderOut = [id for id, trade in self.order_out.items() 
                        if trade[orders.orderIndex["quantity"]] < 0]
        
            # Implement Dual Moving Average Crossover Strategy
            if self.inventory["quantity"]+len(buyOrderOut) <= MAX_INVENT:  # Ensure no long position above 6
                if short_ma > long_ma:  # Golden Cross (Buy Signal)
                    price, quantity = current_price, 1  # Buy one unit at slightly higher price
                    orderClass.send_order(self, price, quantity)
                    self.orderID += 1
            else:
                buyOrderToCancel = buyOrderOut[:MAX_INVENT-(self.inventory["quantity"]+len(buyOrderOut))]

                if len(buyOrderToCancel) > 0:
                    for id in buyOrderToCancel:
                        orderClass.cancel_order(self, id)
            
            if self.inventory["quantity"]-len(sellOrderOut) >= -MAX_INVENT:  # Ensure no short position below 6
                if short_ma < long_ma:  # Death Cross (Sell Signal)
                    price, quantity = current_price, -1  # Sell one unit at slightly lower price
                    orderClass.send_order(self, price, quantity)
                    self.orderID += 1


            else:
                sellOrderToCancel = sellOrderOut[:MAX_INVENT-(-self.inventory["quantity"]+len(sellOrderOut))]

                if len(sellOrderToCancel) >0:
                    for id in sellOrderToCancel:
                        orderClass.cancel_order(self, id)


        # Update filled orders
        orderClass.filled_order(self)