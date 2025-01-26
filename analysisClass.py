import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from orderBookClass import OBData
from datetime import datetime
from debug import logger


class analysisClass:
    def __init__(self, autoTrader):
        self.autoTrader = autoTrader

    def dashboard(self):

        fig = make_subplots(specs=[[{"secondary_y": True}], [{}]], 
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        
        fig.update_layout(title_text="Execution Dashboard")
        

        time = OBData.OBData_[:,OBData.OBIndex["eventTime"]]
        time  = pd.to_datetime(time, unit='ms')
        bids = OBData.OBData_[:,OBData.OBIndex["bids"]]
        asks = OBData.OBData_[:,OBData.OBIndex["asks"]]
        mid = (bids+asks)/2

        historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["idx", "sendTime", "price", "quantity", "endTime", "status"])
        historicalTrades["sendTime"] = pd.to_datetime(historicalTrades["sendTime"], unit='ms')
        historicalTrades["endTime"] = pd.to_datetime(historicalTrades["endTime"], unit='ms')

        historicalInventory = pd.DataFrame({"time":time,"inventory":self.autoTrader.historical_inventory})
        historicalPnL = pd.DataFrame({"time":time,"Pnl":self.autoTrader.historical_pnl})
        historicalUnrealizedPnL = pd.DataFrame({"time":time,"unrealPnl":self.autoTrader.historical_unrealPnL})

        df_mid = pd.DataFrame({"time":time, "mids":mid})
        
        fig.add_trace(go.Scatter(
                            x=df_mid.time,
                            y=df_mid.mids,
                            name="mid price"
                                )
                        ,row=1, col=1, secondary_y=False)
    
        fig.add_trace(go.Scatter(
                        x=historicalPnL.time,
                        y=historicalPnL.Pnl,
                        name="pnl"
                            )
                    ,row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
                        x=historicalUnrealizedPnL.time,
                        y=historicalUnrealizedPnL.unrealPnl,
                        name="UnrealPnl"
                            )
                    ,row=1, col=1, secondary_y=True)
        
        fig.add_trace(
            go.Scatter(
                x=historicalTrades[historicalTrades.quantity > 0].sendTime,  # Timestamps for buy signals
                y=historicalTrades[historicalTrades.quantity > 0].price,  # Prices at which buy signals occurred
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Sent"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=historicalTrades[(historicalTrades.quantity > 0) & (historicalTrades.status==1)].endTime,  # Timestamps for buy signals
                y=historicalTrades[(historicalTrades.quantity > 0) & (historicalTrades.status==1)].price,  # Prices at which buy signals occurred
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Filled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=historicalTrades[(historicalTrades.quantity > 0) & (historicalTrades.status==-1)].endTime,  # Timestamps for buy signals
                y=historicalTrades[(historicalTrades.quantity > 0) & (historicalTrades.status==-1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='blue', size=10, symbol='triangle-up'),
                name="Buy Cancelled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=historicalTrades[(historicalTrades.quantity < 0)].sendTime,  # Timestamps for buy signals
                y=historicalTrades[(historicalTrades.quantity < 0)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name="Sell Sent"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=historicalTrades[(historicalTrades.quantity < 0) & (historicalTrades.status==1)].endTime,  # Timestamps for buy signals
                y=historicalTrades[(historicalTrades.quantity < 0) & (historicalTrades.status==1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name="Sell Filled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=historicalTrades[(historicalTrades.quantity < 0) & (historicalTrades.status==-1)].endTime,  # Timestamps for buy signals
                y=historicalTrades[(historicalTrades.quantity < 0) & (historicalTrades.status==-1)].price,  # Prices at which sell signals occurred
                mode='markers',
                marker=dict(color='blue', size=10, symbol='triangle-down'),
                name="Sell Cancelled"
            ),
            row=1,
            col=1
        )

        fig.add_trace(go.Scatter(
                            x=historicalInventory.time,
                            y=historicalInventory.inventory,
                            name="inventory"
                                )
                        ,row=2, col=1)

        fig.write_html("test.html")
