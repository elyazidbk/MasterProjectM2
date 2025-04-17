from abc import ABC, abstractmethod

class OHLCVTradingStrategy(ABC):
    def __init__(self, name):
        self.name = name
        self.trades = []
        self.position = 0
        self.cash = 0
        self.equity_curve = []

    @abstractmethod
    def generate_signals(self, data):
        pass

    def reset(self):
        self.trades = []
        self.position = 0
        self.cash = 0
        self.equity_curve = []
