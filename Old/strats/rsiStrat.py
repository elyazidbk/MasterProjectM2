import pandas as pd
import numpy as np
from collections import deque
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import trading_strat
from BacktesterOHLCV.debug import logger

MAX_INVENT = 5

class rsiStrat(trading_strat):
    
    def __init__(self, name, window=1000, sellThreshold=70, buyThreshold=40, alpha=0.002):
        super().__init__(name)
        self.name = name
        self.windowLengt = window
        self.windowRSI = deque(maxlen=window)
        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha  # Smoothing factor
        self.historical_RSI = []

    
    def compute_RSI(self):

        self.windowRSI.append(OBData.mid())

        if len(self.windowRSI) > 1:
            delta = self.windowRSI[-1] - self.windowRSI[-2]
        else:
            delta = 0  # No change for the first element

        # Initialize rolling averages if not already done
        if not hasattr(self, "avg_gain"):
            self.avg_gain = 0
            self.avg_loss = 0

        # Compute current gain and loss
        gain = max(delta, 0)
        loss = max(-delta, 0)

        # Update rolling averages using exponential moving average (EMA)

        self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
        self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        # Avoid division by zero and compute RSI
        if self.avg_loss == 0:
            rsi = 100  # No losses mean RSI is maxed
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))

    
        if len(self.windowRSI) < self.windowLengt:
            self.historical_RSI.append(None)
            return None
        else:
            self.historical_RSI.append(rsi)
            return rsi

    def strategy(self, orderClass):

        rsi = self.compute_RSI()
        current_price = OBData.mid()

        if rsi is None:
            pass
        
        else:

            buyOrderOut = [id for id, trade in self.order_out.items() 
                        if trade[orders.orderIndex["quantity"]] > 0]

            sellOrderOut = [id for id, trade in self.order_out.items() 
                        if trade[orders.orderIndex["quantity"]] < 0]
        
            # Implement Dual Moving Average Crossover Strategy
            if self.inventory["quantity"]+len(buyOrderOut) <= MAX_INVENT:  # Ensure no long position above 6
                if rsi <= self.buyThreshold:  # Buy Signal
                    price, quantity = current_price, 1  # Buy one unit at slightly higher price
                    orderClass.send_order(self, price, quantity)
                    self.orderID += 1
            else:
                buyOrderToCancel = buyOrderOut[:MAX_INVENT-(self.inventory["quantity"]+len(buyOrderOut))]

                if len(buyOrderToCancel) > 0:
                    for id in buyOrderToCancel:
                        orderClass.cancel_order(self, id)
            
            if self.inventory["quantity"]-len(sellOrderOut) >= -MAX_INVENT:  # Ensure no short position below 6
                if rsi >= self.sellThreshold:  # Sell Signal
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