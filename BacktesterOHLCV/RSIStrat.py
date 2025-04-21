from ohlcv_trading_strat import OHLCVTradingStrategy
from collections import deque

class RSIStrat(OHLCVTradingStrategy):
    def __init__(self, name, RSI_window=14, sellThreshold=70, buyThreshold=30, alpha=2):
        super().__init__(name)
        self.RSI_window = RSI_window
        self.sellThreshold = sellThreshold
        self.buyThreshold = buyThreshold
        self.alpha = alpha / (RSI_window+1)
        self.prices = deque(maxlen=RSI_window)
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

    def generate_signals(self, data):
        signals = [0] * len(data)
        indicators = [None] * len(data)
        for i, price in enumerate(data['Close']):
            self.prices.append(price)
            rsi = self.compute_RSI()
            indicators[i] = rsi
            if rsi is None:
                continue
            if rsi <= self.buyThreshold:
                signals[i] = 1  # Buy
            elif rsi >= self.sellThreshold:
                signals[i] = -1  # Sell
        return signals, indicators
