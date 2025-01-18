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

