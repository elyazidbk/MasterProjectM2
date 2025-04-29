from engine.strategyClass import OHLCVTradingStrategy
from strats.MAstrat import MAStrat
from strats.RSIStrat import RSIStrat

class MomentumStratV2(OHLCVTradingStrategy):
    def __init__(self, name, short_window, long_window, rsi_window=14, sellThreshold=70, buyThreshold=30, alpha=2):
        super().__init__(name)
        self.ma_strat = MAStrat(name + "_MA", short_window, long_window)
        self.rsi_strat = RSIStrat(name + "_RSI", rsi_window, sellThreshold, buyThreshold, alpha)
        self.buyThreshold = buyThreshold
        self.sellThreshold = sellThreshold

    def generate_signals(self, data):
        ma_signals, ma_indicators = self.ma_strat.generate_signals(data)
        rsi_signals, rsi_values = self.rsi_strat.generate_signals(data)

        signals = [0] * len(data)
        indicators = []

        for i in range(len(data)):
            ma = ma_indicators[i] if ma_indicators and ma_indicators[i] else (None, None)
            rsi = rsi_values[i]
            short_ma, long_ma = ma

            if short_ma is not None and long_ma is not None and rsi is not None:
                if short_ma > long_ma and rsi < self.buyThreshold:
                    signals[i] = 1
                elif short_ma < long_ma and rsi > self.sellThreshold:
                    signals[i] = -1

            indicators.append((short_ma, long_ma, rsi))

        return signals, indicators
