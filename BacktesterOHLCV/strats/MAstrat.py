from engine.strategyClass import OHLCVTradingStrategy
from collections import deque

class MAStrat(OHLCVTradingStrategy):
    def __init__(self, name, short_window, long_window):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.prices = deque(maxlen=max(short_window, long_window))
        self.short_sum = 0
        self.long_sum = 0

    def calculate_moving_averages(self):
        if len(self.prices) < self.short_window:
            return None, None
        if len(self.prices) == self.short_window:
            self.short_sum = sum(self.prices)
        else:
            self.short_sum += self.prices[-1] - self.prices[-self.short_window - 1]
        short_ma = self.short_sum / self.short_window
        if len(self.prices) < self.long_window:
            return short_ma, None
        if len(self.prices) == self.long_window:
            self.long_sum = sum(self.prices)
        else:
            self.long_sum += self.prices[-1] - self.prices[-self.long_window - 1]
        long_ma = self.long_sum / self.long_window
        return short_ma, long_ma

    def generate_signals(self, data):
        signals = [0] * len(data)
        indicators = [None] * len(data)
        prev_short_ma, prev_long_ma = None, None
        for i, price in enumerate(data['Adjusted Close']):
            self.prices.append(price)
            short_ma, long_ma = self.calculate_moving_averages()
            indicators[i] = (short_ma, long_ma)
            if short_ma is None or long_ma is None:
                prev_short_ma, prev_long_ma = short_ma, long_ma
                continue
            # Only signal at the crossover moment
            if prev_short_ma is not None and prev_long_ma is not None:
                if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                    signals[i] = 1  # Crossed above
                elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                    signals[i] = -1  # Crossed below
            prev_short_ma, prev_long_ma = short_ma, long_ma
        return signals, indicators
