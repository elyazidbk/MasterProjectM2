from ohlcv_data import OHLCVDataLoader
from example_strategy import ExampleMovingAverageCrossStrategy
from ohlcv_backtester import OHLCVBacktester
from ohlcv_analysis import OHLCVAnalysis
import os

# Path to a sample SP500 CSV file (e.g., Apple)
data_path = os.path.expanduser("~/Downloads/stock_market_data/sp500/csv/AAPL.csv")

# 1. Load and clean data
data_loader = OHLCVDataLoader(data_path)
data = data_loader.load()
data = data_loader.clean()

# 2. Instantiate strategy
strategy = ExampleMovingAverageCrossStrategy(short_window=10, long_window=30)

# 3. Run backtest
backtester = OHLCVBacktester(data, strategy, fee_perc=0.0005)
trades, equity_curve = backtester.run()

# 4. Analyze and visualize
analysis = OHLCVAnalysis(equity_curve, trades, data)
metrics = analysis.compute_metrics()
print(metrics)
analysis.plot()
