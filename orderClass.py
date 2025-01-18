import pandas as pd
import numpy as np
from orderBookClass import OBData


class orders:
    def __init__(self, OBState: list):
        self.__class__.time = OBState[OBData.OBIndex["transactionTime"]]
        self.__class__.bids = OBState[OBData.OBIndex["bids"]]
        self.__class__.asks = OBState[OBData.OBIndex["asks"]]
        self.__class__.bids_v = OBState[OBData.OBIndex["bids_v"]]
        self.__class__.asks_v = OBState[OBData.OBIndex["asks_v"]]

        # self.__class__.orderID = 0

        self.__class__.orderIndex = {"sendTime":0, "price":1, "quantity":2, "executionTime":3, "status":4}
        self.__class__.fees = {"market": 0.0002, "limit": 0.0001}

    @classmethod
    def send_order(self,trading_strat: object,price: int, quantity:int):
        # In the future replace print by logs
        # For now only bba and bba_v -> not list 
        if ((quantity < 0) and (price>self.bids)) or ((quantity > 0) and (price < self.asks)):
            # Send limit order:
            trading_strat.order_out[trading_strat.orderID]= [self.time, price, quantity, None, 0] # sendTime, price, quantity, executionTime, status={"out":0, "filled":1, "cancelled":-1}
            fees = price*abs(quantity)*self.fees["limit"]
            # print(f"Limit order sent : {quantity} @ {price} - fees = {fees}")
            return {"orderID": trading_strat.orderID, "time": self.time, "price":price, "quantity": quantity, "fees": fees}

        elif (quantity < 0) and (price <= self.bids):
            # Send sell market order:
            if self.bids_v >= abs(quantity):
                trading_strat.historical_trade.append([trading_strat.orderID,self.time, price, quantity, self.time, 1]) # sendTime, price, quantity, executionTime, status={"out":0, "filled":1, "cancelled":-1}
                fees = price*abs(quantity)*self.fees["market"]
                # print(f"Market order sent : {quantity} @ {price} - fees = {fees}")
                return {"orderID": trading_strat.orderID, "time": self.time, "price":price, "quantity": quantity, "fees": fees}
            
            else: 
                trading_strat.historical_trade.append([trading_strat.orderID,self.time, price, -self.bids_v, self.time, 1]) # sendTime, price, quantity, executionTime, status={"out":0, "filled":1, "cancelled":-1}
                fees = price*self.bids_v*self.fees["market"]
                # print(f"Market order sent : {-self.bids_v} @ {price} - fees = {fees}")
                return {"orderID": trading_strat.orderID, "time": self.time, "price":price, "quantity": -self.bids_v, "fees": fees}

        elif (quantity > 0) and (price >= self.asks):
            # Send buy market order:
            if abs(self.asks_v) >= quantity:
                trading_strat.historical_trade.append([trading_strat.orderID,self.time, price, quantity, self.time,1]) # sendTime, price, quantity, executionTime, status={"out":0, "filled":1, "cancelled":-1}
                fees = price*quantity*self.fees["market"]
                # print(f"Market order sent : {quantity} @ {price} - fees = {fees}")
                return {"orderID": trading_strat.orderID, "time": self.time, "price":price, "quantity": quantity, "fees": fees}
            
            else: 
                trading_strat.historical_trade.append([trading_strat.orderID,self.time, price, -self.asks_v, self.time,1]) # sendTime, price, quantity, executionTime, status={"out":0, "filled":1, "cancelled":-1}
                fees = price*abs(self.asks_v)*self.fees["market"]
                # print(f"Market order sent : {-self.asks_v[0]} @ {price} - fees = {fees}")
                return {"orderID": trading_strat.orderID, "time": self.time, "price":price, "quantity": -self.asks_v, "fees": fees}
        
 

    def cancel_order(self, trading_strat: object, orderID: int):
        print("step",OBData.step,"order cancel : ", trading_strat.order_out[orderID])
        trading_strat.order_out.pop(orderID) 

    def filled_order(self, trading_strat):
        orderToCancel = []
        for orderID in trading_strat.order_out.keys():
            if (

                (trading_strat.order_out[orderID][self.orderIndex["quantity"]] < 0) and 
                (trading_strat.order_out[orderID][self.orderIndex["price"]] <= self.bids)
                
                ):
                # sell order filled
                trading_strat.order_out[orderID][self.orderIndex["executionTime"]] = self.time
                trading_strat.order_out[orderID][self.orderIndex["status"]] = 1


                if abs(trading_strat.order_out[orderID][self.orderIndex["quantity"]]) > self.bids_v:
                    # partial filled
                    # for now we handle partiel filled cancelling the order - maybe keep it in the future
                    trading_strat.order_out[orderID][self.orderIndex["quantity"]] = self.bids_v
                
                trading_strat.historical_trade.append(trading_strat.order_out[orderID].insert(0, orderID)) # SEE HOW TO MERGE orderID in the list
                # self.cancel_order(trading_strat, orderID)
                orderToCancel.append(orderID)

            elif (

                (trading_strat.order_out[orderID][self.orderIndex["quantity"]] > 0) and 
                (trading_strat.order_out[orderID][self.orderIndex["price"]] >= self.asks)
                
                ):

                # buy order filled
                trading_strat.order_out[orderID][self.orderIndex["executionTime"]] = self.time
                trading_strat.order_out[orderID][self.orderIndex["status"]] = 1

                if trading_strat.order_out[orderID][self.orderIndex["quantity"]] > abs(self.asks_v):
                    # partial filled
                    # for now we handle partiel filled cancelling the order - maybe keep it in the future
                    trading_strat.order_out[orderID][self.orderIndex["quantity"]] = self.asks_v
                
                trading_strat.historical_trade.append(trading_strat.order_out[orderID].insert(0, orderID)) # SEE HOW TO MERGE orderID in the list
                # self.cancel_order(trading_strat, orderID)
                orderToCancel.append(orderID)
        
        if len(orderToCancel) > 0:
            for id in orderToCancel:
                self.cancel_order(trading_strat, id)

