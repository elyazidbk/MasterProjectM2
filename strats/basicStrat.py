import pandas as pd
import numpy as np
from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.tradingStratClass import trading_strat
from debug import logger

MAX_INVENT = 5

class basicStrat(trading_strat):

    def __init__(self, name):
        super().__init__(name)

    def strategy(self, orderClass):

        targetBuy = 59000
        targetSell = 60000

        
        if OBData.mid()<=targetBuy:
            # Best ask below Buy Target -> I buy
            if  self.inventory["quantity"] <= MAX_INVENT:
                price, quantity = targetBuy, 1
                orderClass.send_order(self, price, quantity)
                self.orderID +=1
            else:
                buyOrderToCancel = [id for id, trade in self.order_out.items() 
                                    if trade[orders.orderIndex["quantity"]] > 0]
                if len(buyOrderToCancel) > 0:
                    for id in buyOrderToCancel:
                        orderClass.cancel_order(self, id)

        elif OBData.mid()>=targetSell:
            # Mid above Sell Target -> I sell
            if self.inventory["quantity"] >= -MAX_INVENT:
                price, quantity = targetSell, -1
                orderClass.send_order(self, price, quantity)
                self.orderID +=1
            else:
                sellOrderToCancel = [id for id, trade in self.order_out.items() 
                                    if trade[orders.orderIndex["quantity"]] < 0]
                if len(sellOrderToCancel) >0:
                    for id in sellOrderToCancel:
                        orderClass.cancel_order(self, id)

        
        orderClass.filled_order(self) 