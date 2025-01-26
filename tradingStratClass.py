import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from orderClass import orders
from orderBookClass import OBData
from debug import logger

class trading_strat(ABC):
    def __init__(self, strat_name):
        self.strat = strat_name
        self.historical_trade = []
        self.historical_inventory = []
        self.historical_pnl = []
        self.historical_unrealPnL = []
        self.PnL = 0
        self.unrealPnL = 0
        self.inventory = {"price" : 0 , "quantity" : 0}
        self.__class__.order_out = {}
        self.__class__.orderID = 0

    def computePnL(self, orderID):

        avgPrice = self.inventory["price"]

        if self.inventory["quantity"] > 0:
            # Compute of last filled order negative
            order = self.order_out[orderID]
            if order[orders.orderIndex["quantity"]] < 0 : 
                if (
                    (np.sign(self.inventory["quantity"]+order[orders.orderIndex["quantity"]]) == np.sign(self.inventory["quantity"]))
                    or
                    (np.sign(self.inventory["quantity"]+order[orders.orderIndex["quantity"]]) == 0)
                    
                    ):
                    logger.info(f'order price : {order[orders.orderIndex["price"]]} - inventPrice : {avgPrice} - order qty: {order[orders.orderIndex["quantity"]]}')
                    self.PnL += (avgPrice-order[orders.orderIndex["price"]])*order[orders.orderIndex["quantity"]]
                    logger.info(f'PnL Generated: {self.PnL}')
                else: 
                    self.PnL += (avgPrice-order[orders.orderIndex["price"]])*(self.inventory["quantity"])

        elif self.inventory["quantity"] < 0:
            # Compute if last filled order positive
            order = self.order_out[orderID]
            if order[orders.orderIndex["quantity"]] > 0 : 
                if (
                    (np.sign(self.inventory["quantity"]+order[orders.orderIndex["quantity"]]) == np.sign(self.inventory["quantity"]))
                    or
                    (np.sign(self.inventory["quantity"]+order[orders.orderIndex["quantity"]]) == 0)
                    
                    ):
                    logger.info(f'order price : {order[orders.orderIndex["price"]]} - inventPrice : {avgPrice} - order qty: {order[orders.orderIndex["quantity"]]}')
                    self.PnL += (avgPrice-order[orders.orderIndex["price"]])*order[orders.orderIndex["quantity"]]
                    logger.info(f'PnL Generated: {self.PnL}')
                else: 
                    self.PnL += (avgPrice-order[orders.orderIndex["price"]])*(self.inventory["quantity"])


        return 

    def computeUnrealPnL(self):
        """
        Compute unrealized PnL - with only one asset available for now
        """
        avgPrice = self.inventory["price"]
        quantity = self.inventory["quantity"]
        
        self.unrealPnL = (avgPrice-OBData.mid())*-quantity

        if len(self.historical_pnl)>0:
            self.unrealPnL += self.historical_pnl[-1]

        return 
    
    def updateInventory(self, orderPrice: int, orderQuantity: int):

        if self.inventory["quantity"] == 0:
            self.inventory["price"] = orderPrice
        elif np.sign(self.inventory["quantity"] + orderQuantity) != np.sign(self.inventory["quantity"]):
            self.inventory["price"] = orderPrice
        elif np.sign(self.inventory["quantity"]) != np.sign(orderQuantity):
            pass
        elif np.sign(self.inventory["quantity"]) == np.sign(orderQuantity):
            self.inventory["price"] = (self.inventory["price"]*self.inventory["quantity"] 
                                                + orderPrice*orderQuantity) / (orderQuantity+self.inventory["quantity"]) 
            
        self.inventory["quantity"] += orderQuantity
    
    @abstractmethod
    def strategy():
        return
    

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
        
