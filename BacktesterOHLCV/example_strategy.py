from ohlcv_trading_strat import OHLCVTradingStrategy
import pandas as pd

class ExampleMovingAverageCrossStrategy(OHLCVTradingStrategy):
    def __init__(self, short_window=10, long_window=30):
        super().__init__('MA_Cross')
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = [0] * len(data)
        short_ma = data['Close'].rolling(self.short_window).mean()
        long_ma = data['Close'].rolling(self.long_window).mean()
        for i in range(len(data)):
            if i < self.long_window:
                signals[i] = 0
            elif short_ma.iloc[i] > long_ma.iloc[i]:
                signals[i] = 1
            elif short_ma.iloc[i] < long_ma.iloc[i]:
                signals[i] = -1
            else:
                signals[i] = 0
        return signals
