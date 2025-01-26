import pandas as pd
import numpy as np
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import trading_strat
from debug import logger

class basicStrat(trading_strat):

    def __init__(self, name):
        super().__init__(name)

    def strategy(self, orderClass):

        targetBuy = 59000
        targetSell = 60000

        
        if OBData.mid()<=targetBuy:
            # Best ask below Buy Target -> I buy
            if  self.inventory["quantity"] <= 5:
                price, quantity = targetBuy, 1
                orderClass.send_order(self, price, quantity)
                self.orderID +=1
            else:
                orderToCancel = list(self.order_out.keys())
                for id in orderToCancel:
                    orderClass.cancel_order(self, id)

        elif OBData.mid()>=targetSell:
            # Mid above Sell Target -> I sell
            if self.inventory["quantity"] >= -5:
                price, quantity = targetSell, -1
                orderClass.send_order(self, price, quantity)
                self.orderID +=1
            else:
                orderToCancel = list(self.order_out.keys())
                for id in orderToCancel:
                    orderClass.cancel_order(self, id)
        
        orderClass.filled_order(self) 