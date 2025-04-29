import pandas as pd
import numpy as np
from strats.RSIStrat import RSIStrat
from strats.MAstrat import MAStrat

def compute_features_per_asset(df, hybrid_strategy):
    df = df.copy()

    # Basic engineered features
    df['return'] = df['Adjusted Close'].pct_change()
    df['Previous Close'] = df['Adjusted Close'].shift(1)
    df['log_return'] = np.log(df['Adjusted Close'] / df['Adjusted Close'].shift(1))
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std())

    # Hybrid strategy signals and indicators
    signals, indicators = hybrid_strategy.generate_signals(df)

    # Extract short_ma, long_ma, rsi
    short_ma_vals = []
    long_ma_vals = []
    rsi_vals = []

    for ind in indicators:
        if ind and isinstance(ind, tuple) and len(ind) == 3:
            short_ma, long_ma, rsi = ind
        else:
            short_ma, long_ma, rsi = None, None, None

        short_ma_vals.append(short_ma)
        long_ma_vals.append(long_ma)
        rsi_vals.append(rsi)

    # Add to DataFrame
    df['short_ma'] = short_ma_vals
    df['long_ma'] = long_ma_vals
    df['ma_diff'] = df['short_ma'] - df['long_ma']
    df['rsi'] = rsi_vals
    df['signal'] = signals

    return df



def generate_flat_ml_dataframe(raw_df, asset_list, hybrid_strategy):
    all_rows = []

    for asset in asset_list:
        cols = [c for c in raw_df.columns if c.startswith(asset+'_')]
        df = raw_df[cols].copy()
        df.columns = [c.replace(asset+'_', '') for c in cols]
        df['ticker'] = asset
        df['date'] = raw_df.index

        # Use passed strategy object
        df = compute_features_per_asset(df, hybrid_strategy)

        # Ensure 'Previous Close' exists for selection
        if 'Previous Close' not in df.columns:
            df['Previous Close'] = df['Adjusted Close'].shift(1)

        selected = df[['date', 'ticker', 'Adjusted Close', 'Previous Close', 'return', 'log_return', 'volume_zscore',
                       'short_ma', 'long_ma', 'ma_diff', 'rsi', 'signal']]
        all_rows.append(selected)

    flat_df = pd.concat(all_rows).reset_index(drop=True)
    return flat_df

