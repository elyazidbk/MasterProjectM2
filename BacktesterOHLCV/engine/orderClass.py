class OHLCVOrder:
    def __init__(self, date, price, quantity, side, fee, indicator=None, trade_type="signal"):
        self.date = date
        self.price = price
        self.quantity = quantity
        self.side = side  # 'buy' or 'sell'
        self.fee = fee
        self.indicator = indicator  # Store the indicator value for this order
        self.trade_type = trade_type  # 'signal' or 'rebalance'

    def __repr__(self):
        return f"{self.date}: {self.side} {self.quantity} @ {self.price} (fee={self.fee}, indicator={self.indicator}, trade_type={self.trade_type})"
