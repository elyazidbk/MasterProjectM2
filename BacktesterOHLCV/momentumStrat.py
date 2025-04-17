from ohlcv_trading_strat import OHLCVTradingStrategy
from collections import deque

class MomentumStratOHLCV(OHLCVTradingStrategy):
    def __init__(self, name, short_window, long_window, RSI_window=14, sellThreshold=70, buyThreshold=30, alpha=2):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.RSI_window = RSI_window
        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha / (RSI_window+1)

        self.prices = deque(maxlen=max(RSI_window, long_window))
        self.short_sum = 0
        self.long_sum = 0

        self.avg_gain = 0
        self.avg_loss = 0

    def compute_RSI(self):
        if len(self.prices) < 2:
            return None

        delta = self.prices[-1] - self.prices[-2]
        gain = max(delta, 0)
        loss = max(-delta, 0)

        self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
        self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        if self.avg_loss == 0:
            return 100

        rs = self.avg_gain / self.avg_loss
        return 100 - (100 / (1 + rs))

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
        indicators = [None] * len(data)  # To store indicator values for debugging

        for i, price in enumerate(data['Close']):
            self.prices.append(price)

            rsi = self.compute_RSI()
            short_ma, long_ma = self.calculate_moving_averages()

            indicators[i] = rsi  # Store the RSI value for this step

            if rsi is None or short_ma is None or long_ma is None:
                continue

            if rsi <= self.buyThreshold and short_ma > long_ma:
                signals[i] = 1  # Buy signal
            elif rsi >= self.sellThreshold and short_ma < long_ma:
                signals[i] = -1  # Sell signal

        return signals, indicators  # Return both signals and indicator values
