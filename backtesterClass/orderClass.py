import pandas as pd
import numpy as np
from .orderBookClass import OBData
from debug import logger

class orders:
    def __init__(self, OBState: list):
        self.__class__.time = OBState[OBData.OBIndex["eventTime"]]
        self.__class__.bids = OBState[OBData.OBIndex["bids"]]
        self.__class__.asks = OBState[OBData.OBIndex["asks"]]
        self.__class__.bids_v = OBState[OBData.OBIndex["bids_v"]]
        self.__class__.asks_v = OBState[OBData.OBIndex["asks_v"]]

        # self.__class__.orderID = 0

        self.__class__.orderIndex = {"orderId":0, "sendTime":1, "price":2, "quantity":3, "endTime":4, "status":5}
        self.__class__.fees = {"market": 0.0002, "limit": 0.0001}

    @classmethod
    def send_order(self,trading_strat: object,orderPrice: int, orderQuantity:int):
        # In the future replace print by logs
        # For now only bba and bba_v -> not list 
        if ((orderQuantity < 0) and (orderPrice>self.bids)) or ((orderQuantity > 0) and (orderPrice < self.asks)):
            # Send limit order:
            trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID,self.time, orderPrice, orderQuantity, None, 0] # sendTime, price, quantity, endTime, status={"out":0, "filled":1, "cancelled":-1}
            fees = orderPrice*abs(orderQuantity)*self.fees["limit"]
            # print(f"Limit order sent : {quantity} @ {price} - fees = {fees}")

        elif (orderQuantity < 0) and (orderPrice <= self.bids):
            # Send sell market order:
            orderPrice = self.bids # False if send limit order with a fixed price below the bb

            if self.bids_v >= abs(orderQuantity):
                trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID,self.time, orderPrice, orderQuantity,self.time, 1] # sendTime, price, quantity, endTime, status={"out":0, "filled":1, "cancelled":-1}
                trading_strat.historical_trade.append(trading_strat.order_out[trading_strat.orderID]) 
                fees = orderPrice*abs(orderQuantity)*self.fees["market"]
                # print(f"Market order sent : {quantity} @ {price} - fees = {fees}")
            
            else:
                orderQuantity = -self.bids_v
                trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID,self.time, orderPrice, -self.bids_v,self.time, 1] # sendTime, price, quantity, endTime, status={"out":0, "filled":1, "cancelled":-1}
                trading_strat.historical_trade.append(trading_strat.order_out[trading_strat.orderID])
                fees = orderPrice*self.bids_v*self.fees["market"]
                # print(f"Market order sent : {-self.bids_v} @ {price} - fees = {fees}")

        elif (orderQuantity > 0) and (orderPrice >= self.asks):
            # Send buy market order:
            orderPrice = self.asks # False if send limit order with a fixed price above the ba

            if abs(self.asks_v) >= orderQuantity:
                trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID,self.time, orderPrice, orderQuantity,self.time, 1] # sendTime, price, quantity, endTime, status={"out":0, "filled":1, "cancelled":-1}
                trading_strat.historical_trade.append(trading_strat.order_out[trading_strat.orderID])
                fees = orderPrice*orderQuantity*self.fees["market"]
                # print(f"Market order sent : {quantity} @ {price} - fees = {fees}")
            
            else: 
                orderQuantity = -self.asks_v
                trading_strat.order_out[trading_strat.orderID]= [trading_strat.orderID,self.time, orderPrice, -self.asks_v,self.time, 1] # sendTime, price, quantity, endTime, status={"out":0, "filled":1, "cancelled":-1}
                trading_strat.historical_trade.append(trading_strat.order_out[trading_strat.orderID])                
                fees = orderPrice*abs(self.asks_v)*self.fees["market"]
                # print(f"Market order sent : {-self.asks_v[0]} @ {price} - fees = {fees}")        
    @classmethod
    def cancel_order(self, trading_strat: object, orderID: int):
        if trading_strat.order_out[orderID][self.orderIndex["status"]] == 1 : 
            # Filled order to cancel
            pass
        else:
            # Unfilled order to cancel
            tradeCancelled = trading_strat.order_out[orderID]
            tradeCancelled[self.orderIndex["status"]] = -1
            tradeCancelled[self.orderIndex["endTime"]] = self.time
            trading_strat.historical_trade.append(tradeCancelled)
        
        trading_strat.order_out.pop(orderID)
        # logger.info(f"order {orderID} cancelled - orders out : {trading_strat.order_out}")

    @classmethod
    def filled_order(self, trading_strat):
        orderFilledToCancel = []
        
        for orderID in trading_strat.order_out.keys():

            orderQuantity = trading_strat.order_out[orderID][self.orderIndex["quantity"]]
            orderPrice = trading_strat.order_out[orderID][self.orderIndex["price"]] 
            orderStatus = trading_strat.order_out[orderID][self.orderIndex["status"]] 
            
            if orderStatus == 0 :
                if (orderQuantity < 0) and (orderPrice <= self.bids):
                    # sell order filled
                    trading_strat.order_out[orderID][self.orderIndex["endTime"]] = self.time
                    trading_strat.order_out[orderID][self.orderIndex["status"]] = 1

                    if abs(orderQuantity) > self.bids_v:
                        # partial filled
                        # for now we handle partiel filled cancelling the order - maybe keep it in the future
                        orderQuantity = -self.bids_v

                    trading_strat.computePnL(orderID)
                    trading_strat.updateInventory(orderPrice, orderQuantity)
                    tradeFilled = trading_strat.order_out[orderID]
                    trading_strat.historical_trade.append(tradeFilled)
                    orderFilledToCancel.append(orderID)
                
                elif (orderQuantity > 0) and (orderPrice >= self.asks):

                    # buy order filled
                    trading_strat.order_out[orderID][self.orderIndex["endTime"]] = self.time
                    trading_strat.order_out[orderID][self.orderIndex["status"]] = 1

                    if orderQuantity > abs(self.asks_v):
                        # partial filled
                        # for now we handle partiel filled cancelling the order - maybe keep it in the future
                        orderQuantity = -self.asks_v

                    trading_strat.computePnL(orderID)
                    trading_strat.updateInventory(orderPrice,orderQuantity)
                    tradeFilled = trading_strat.order_out[orderID]
                    trading_strat.historical_trade.append(tradeFilled)
                    orderFilledToCancel.append(orderID)
            
            elif orderStatus == 1:
                trading_strat.computePnL(orderID)
                trading_strat.updateInventory(orderPrice, orderQuantity)
                orderFilledToCancel.append(orderID)


        if len(orderFilledToCancel) > 0:
            for id in orderFilledToCancel:
                self.cancel_order(trading_strat, id)

        if len(trading_strat.historical_pnl)>0:
            trading_strat.PnL += trading_strat.historical_pnl[-1]


        
        
        trading_strat.historical_pnl.append(trading_strat.PnL) # add realized PnL to the historic
        trading_strat.computeUnrealPnL()
        trading_strat.historical_unrealPnL.append(trading_strat.unrealPnL) # add unrealized PnL to the historic
        trading_strat.historical_inventory.append(trading_strat.inventory["quantity"])

        # Reset :
        trading_strat.PnL = 0
        trading_strat.unrealPnL = 0


