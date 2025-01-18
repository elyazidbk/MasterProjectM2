import pandas as pd
import plotly.express as px


class analysisClass:
    def __init__(self, autoTrader):
        self.autoTrader = autoTrader

    def dashboard(self):
        historicalTrades = pd.DataFrame(self.autoTrader.historical_trade, columns=["idx", "sendTime", "price", "quantity", "fillTime", "status"])
        print(historicalTrades)

