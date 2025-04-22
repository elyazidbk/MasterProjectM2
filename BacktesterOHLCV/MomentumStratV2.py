from ohlcv_trading_strat import OHLCVTradingStrategy
from MAstrat import MAStrat
from RSIStrat import RSIStrat

class MomentumStratV2(OHLCVTradingStrategy):
    def __init__(self, name, short_window, long_window, rsi_window=14, sellThreshold=70, buyThreshold=30, alpha=2):
        super().__init__(name)
        self.ma_strat = MAStrat(name+"_MA", short_window, long_window)
        self.rsi_strat = RSIStrat(name+"_RSI", rsi_window, sellThreshold, buyThreshold, alpha)

    def generate_signals(self, data):
        ma_signals, ma_indicators = self.ma_strat.generate_signals(data)
        rsi_signals, rsi_indicators = self.rsi_strat.generate_signals(data)
        signals = [0] * len(ma_signals)
        for i in range(len(signals)):
            if ma_signals[i] == 1 and rsi_signals[i] == 1:
                signals[i] = 1
            elif ma_signals[i] == -1 and rsi_signals[i] == -1:
                signals[i] = -1
        indicators = list(zip(
            ma_indicators if ma_indicators else [None]*len(signals),
            rsi_indicators if rsi_indicators else [None]*len(signals)
        ))
        return signals, indicators
