import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from datetime import datetime 
from debug import logger

from .orderBookClass import OBData

from strats.basicStrat import basicStrat
from strats.movingAverageStrat import movingAverageStrat
from strats.rsiStrat import rsiStrat
from strats.momentumStrat import momentumStrat
from strats.momentumOnlineLearn import momentumOnlineLearnStrat

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

        self.data = pd.DataFrame({"time":self.time,
                                  "mids":self.mid,
                                  "inventory":self.autoTrader.historical_inventory,
                                  "Pnl":self.autoTrader.historical_pnl,
                                  "unrealPnl":self.autoTrader.historical_unrealPnL})

        self.historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["idx", "sendTime", "price", "quantity", "endTime", "status"])
        self.historicalTrades["sendTime"] = pd.to_datetime(self.historicalTrades["sendTime"], unit='ms')
        self.historicalTrades["endTime"] = pd.to_datetime(self.historicalTrades["endTime"], unit='ms')
    
    def create_dashboard(self, save = True, show = False, streamlit=False):

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):
            # Intermediary dashboard to display rsi chart
            fig = make_subplots(specs=[[{"secondary_y": True}], [{}], [{}]], 
                                rows=3, cols=1,
                                shared_xaxes=True,
                                row_heights=[0.5, 0.2, 0.3],
                                vertical_spacing=0.02)
        else:
            
            fig = make_subplots(specs=[[{"secondary_y": True}], [{}]], 
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
        

        fig.update_layout(title_text=f"{self.autoTrader.strat} Execution Dashboard")
        

        fig.add_trace(go.Scatter(
                            x=self.data.time,
                            y=self.data.mids,
                            name="mid price",
                            marker=dict(color='darkgray')
                                )
                        ,row=1, col=1, secondary_y=False)
    
        fig.add_trace(go.Scatter(
                        x=self.data.time,
                        y=self.data.Pnl,
                        name="pnl",
                        marker=dict(color='red')
                            )
                    ,row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
                        x=self.data.time,
                        y=self.data.unrealPnl,
                        name="UnrealPnl",
                        marker=dict(color='deepskyblue')
                            )
                    ,row=1, col=1, secondary_y=True)
        
        fig.add_trace(
            go.Scatter(
                x=self.historicalTrades[self.historicalTrades.quantity > 0].sendTime,  # Timestamps for buy signals
                y=self.historicalTrades[self.historicalTrades.quantity > 0].price,  # Prices at which buy signals occurred
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Sent",
                visible='legendonly'
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
                name="Buy Cancelled",
                visible='legendonly'
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
                name="Sell Sent",
                visible='legendonly'
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
                name="Sell Cancelled",
                visible='legendonly'
            ),
            row=1,
            col=1
        )

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):
            fig.add_trace(go.Scatter(
                                x=self.data.time,
                                y=self.data.inventory,
                                name="inventory",
                                marker=dict(color='darkgray')
                                    )
                            ,row=3, col=1)
        else:
            fig.add_trace(go.Scatter(
                                x=self.data.time,
                                y=self.data.inventory,
                                name="inventory",
                                marker=dict(color='darkgray')
                                    )
                            ,row=2, col=1)           

        if isinstance(self.autoTrader, movingAverageStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):

            fig.add_trace(go.Scatter(
                                x=self.time,
                                y=self.autoTrader.historical_long_ma,
                                name="long_ma",
                                marker=dict(color='brown')
                                    )
                            ,row=1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                                x=self.time,
                                y=self.autoTrader.historical_short_ma,
                                name="short_ma",
                                marker=dict(color='seagreen')
                                    )
                            ,row=1, col=1, secondary_y=False)
            
        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat) or isinstance(self.autoTrader, momentumOnlineLearnStrat):

            fig.add_trace(go.Scatter(
                                x=self.time,
                                y=self.autoTrader.historical_RSI,
                                name="rsi",
                                marker=dict(color='darkgray')
                                    )
                            ,row=2, col=1, secondary_y=False)

            fig.add_shape(
                type="line",
                x0=min(self.time),  # Starting x-coordinate
                x1=max(self.time),  # Ending x-coordinate
                y0=self.autoTrader.sellThreshold, # y-coordinate for the line
                y1=self.autoTrader.sellThreshold,
                line=dict(color="red", width=2, dash="dash"),
                xref="x2",  # Refers to the x-axis of the second row
                yref="y3"   # Refers to the y-axis of the second row
            )

            fig.add_shape(
                type="line",
                x0=min(self.time),
                x1=max(self.time),
                y0=self.autoTrader.buyThreshold,
                y1=self.autoTrader.buyThreshold,
                line=dict(color="green", width=2, dash="dash"),
                xref="x2",
                yref="y3"
            )            

            fig.update_layout(
                legend_orientation="h",
                xaxis3=dict(
                    rangeslider=dict(
                        visible=True,
                        bgcolor="darkgray",  # Set the background color of the slider
                        thickness=0.03  # Set the thickness of the range slider
                    ),
                    showgrid=True,
                ),

                legend=dict(
                    orientation="v",  # Vertical legend
                    xanchor="right",  # Anchor on right based on the x value
                    x=1.1,
                    yanchor="top"  # Anchor it to the top of the legend box
                ))
        
        if not isinstance(self.autoTrader, rsiStrat) and not isinstance(self.autoTrader, momentumStrat) and not isinstance(self.autoTrader, momentumOnlineLearnStrat) :

            fig.update_layout(
                legend_orientation="h",
                xaxis2=dict(
                    rangeslider=dict(
                        visible=True,
                        bgcolor="darkgray",  # Set the background color of the slider
                        thickness=0.03  # Set the thickness of the range slider
                    ),
                    showgrid=True,
                ),

                legend=dict(
                    orientation="v",  # Vertical legend
                    xanchor="right",  # Anchor on right based on the x value
                    x=1.1,
                    yanchor="top"  # Anchor it to the top of the legend box
                ))

        fig.update_layout(
            width=1500,  
            height=800, 
        )

        

        if show:
            fig.show()

        if save:
            if self.path == None:
                fig.write_html(f"{self.dashboardName}.html")
            else:
                fig.write_html(f"{self.path}/{self.dashboardName}.html")
        
        if streamlit:
            return fig
        
        return
    
    @classmethod
    def streamlitDashboard(self, fig):
        st.plotly_chart(fig, use_container_width=True)
        return 


    def dataBase(self):

        if self.path == None:
            db = SqlAlchemyDataBaseManager(f"{self.dashboardName}.db")
        else:
            db = SqlAlchemyDataBaseManager(f"{self.path}/{self.dbName}.db")


        historicalInventory = self.data[["time", "inventory"]]
        historicalPnL = self.data[["time", "PnL"]]
        historicalUnrealizedPnL = self.data[["time", "unrealPnl"]]
        df_mid = self.data[["time", "mids"]]

        db.update("historicalPrices",df_mid)
        db.update("historicalTrades",self.historicalTrades)
        db.update("historicalInventory",historicalInventory)
        db.update("historicalPnL",historicalPnL)
        db.update("historicalUnrealizedPnL",historicalUnrealizedPnL)

        if isinstance(self.autoTrader, rsiStrat) or isinstance(self.autoTrader, momentumStrat):
            historicalRSI = pd.DataFrame({"time":self.time,"RSI":self.autoTrader.historical_RSI})
            db.update("historicalRSI", historicalRSI)
        
        elif isinstance(self.autoTrader, movingAverageStrat) or isinstance(self.autoTrader, momentumStrat):
            historicalMA = pd.DataFrame({"time":self.time,
                                         "long_ma":self.autoTrader.historical_long_ma,
                                         "short_ma":self.autoTrader.historical_short_ma})
            
            db.update("historicalMA", historicalMA)
