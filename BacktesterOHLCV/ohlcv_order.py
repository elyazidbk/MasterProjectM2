class OHLCVOrder:
    def __init__(self, date, price, quantity, side, fee, indicator=None):
        self.date = date
        self.price = price
        self.quantity = quantity
        self.side = side  # 'buy' or 'sell'
        self.fee = fee
        self.indicator = indicator  # Store the indicator value for this order

    def __repr__(self):
        return f"{self.date}: {self.side} {self.quantity} @ {self.price} (fee={self.fee}, indicator={self.indicator})"
