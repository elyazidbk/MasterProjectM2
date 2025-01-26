import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .orderBookClass import OBData
from datetime import datetime
from debug import logger
from SQLite_Manager.sqlManager import SqlAlchemyDataBaseManager


class analysisClass:
    def __init__(self, autoTrader, path : str = None, dashboardName : str = None, dbName : str = None):
        self.autoTrader = autoTrader
        self.path = path
        self.dashboardName = dashboardName
        self.dbName = dbName

        self.time = OBData.OBData_[:,OBData.OBIndex["eventTime"]]
        self.time  = pd.to_datetime(self.time, unit='ms')
        self.bids = OBData.OBData_[:,OBData.OBIndex["bids"]]
        self.asks = OBData.OBData_[:,OBData.OBIndex["asks"]]
        self.mid = (self.bids+self.asks)/2

        self.historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["idx", "sendTime", "price", "quantity", "endTime", "status"])
        self.historicalTrades["sendTime"] = pd.to_datetime(self.historicalTrades["sendTime"], unit='ms')
        self.historicalTrades["endTime"] = pd.to_datetime(self.historicalTrades["endTime"], unit='ms')

        self.historicalInventory = pd.DataFrame({"time":self.time,"inventory":self.autoTrader.historical_inventory})
        self.historicalPnL = pd.DataFrame({"time":self.time,"Pnl":self.autoTrader.historical_pnl})
        self.historicalUnrealizedPnL = pd.DataFrame({"time":self.time,"unrealPnl":self.autoTrader.historical_unrealPnL})

        self.df_mid = pd.DataFrame({"time":self.time, "mids":self.mid})

    def dashboard(self, save = True, show = False):

        fig = make_subplots(specs=[[{"secondary_y": True}], [{}]], 
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        
        fig.update_layout(title_text="Execution Dashboard")
        


        
        fig.add_trace(go.Scatter(
                            x=self.df_mid.time,
                            y=self.df_mid.mids,
                            name="mid price"
                                )
                        ,row=1, col=1, secondary_y=False)
    
        fig.add_trace(go.Scatter(
                        x=self.historicalPnL.time,
                        y=self.historicalPnL.Pnl,
                        name="pnl"
                            )
                    ,row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
                        x=self.historicalUnrealizedPnL.time,
                        y=self.historicalUnrealizedPnL.unrealPnl,
                        name="UnrealPnl"
                            )
                    ,row=1, col=1, secondary_y=True)
        
        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[self.historicalTrades.quantity > 0].sendTime,  # Timestamps for buy signals
                y=self.historicalTrades[self.historicalTrades.quantity > 0].price,  # Prices at which buy signals occurred
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Sent"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[(self.historicalTrades.quantity > 0) & (self.historicalTrades.status==1)].endTime,  # Timestamps for buy signals
                y=self.historicalTrades[(self.historicalTrades.quantity > 0) & (self.historicalTrades.status==1)].price,  # Prices at which buy signals occurred
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Filled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[(self.historicalTrades.quantity > 0) & (self.historicalTrades.status==-1)].endTime,  # Timestamps for buy signals
                y=self.historicalTrades[(self.historicalTrades.quantity > 0) & (self.historicalTrades.status==-1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='blue', size=10, symbol='triangle-up'),
                name="Buy Cancelled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[(self.historicalTrades.quantity < 0)].sendTime,  # Timestamps for buy signals
                y=self.historicalTrades[(self.historicalTrades.quantity < 0)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name="Sell Sent"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[(self.historicalTrades.quantity < 0) & (self.historicalTrades.status==1)].endTime,  # Timestamps for buy signals
                y=self.historicalTrades[(self.historicalTrades.quantity < 0) & (self.historicalTrades.status==1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name="Sell Filled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[(self.historicalTrades.quantity < 0) & (self.historicalTrades.status==-1)].endTime,  # Timestamps for buy signals
                y=self.historicalTrades[(self.historicalTrades.quantity < 0) & (self.historicalTrades.status==-1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='blue', size=10, symbol='triangle-down'),
                name="Sell Cancelled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(go.Scatter(
                            x=self.historicalInventory.time,
                            y=self.historicalInventory.inventory,
                            name="inventory"
                                )
                        ,row=2, col=1)

        if show:
            fig.show()
        if save:
            if self.path == None:
                fig.write_html(f"{self.dashboardName}.html")
            else:
                fig.write_html(f"{self.path}/{self.dashboardName}.html")

    def dataBase(self):

        if self.path == None:
            db = SqlAlchemyDataBaseManager(f"{self.dashboardName}.db")
        else:
            db = SqlAlchemyDataBaseManager(f"{self.path}/{self.dbName}.db")

        db.update("historicalTrades",self.historicalTrades)
        db.update("historicalInventory",self.historicalInventory)
        db.update("historicalPnL",self.historicalPnL)
        db.update("historicalUnrealizedPnL",self.historicalUnrealizedPnL)