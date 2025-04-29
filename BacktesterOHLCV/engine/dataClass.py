import os
import pandas as pd
from engine.debug import logger

class OHLCVDataLoader:
    def __init__(self, filepath: str, sp500_path: str = None):
        self.filepath = filepath
        self.data = None
        self.sp500_path = sp500_path
        self.sp500_tickers = None
        if sp500_path is not None:
            self._load_sp500_tickers()

    def _load_sp500_tickers(self):
        try:
            df = pd.read_csv(self.sp500_path)
            # Try to find a column with tickers
            for col in df.columns:
                if 'ticker' in col.lower() or 'symbol' in col.lower():
                    self.sp500_tickers = set(df[col].astype(str).str.upper().str.strip())
                    return
            # Fallback: use first column
            self.sp500_tickers = set(df[df.columns[0]].astype(str).str.upper().str.strip())
        except Exception as e:
            logger.warning(f"Could not load SP500 tickers from {self.sp500_path}: {e}")
            self.sp500_tickers = None

    def load(self):
        self.data = pd.read_csv(self.filepath)
        logger.info(f"Loaded data from {self.filepath} with shape {self.data.shape}")
        return self.data

    def clean(self, years=None):
        # Convert the first column (assumed to be Date) to datetime
        self.data[self.data.columns[0]] = pd.to_datetime(self.data[self.data.columns[0]], format='%d-%m-%Y')

        # Filter data for the last 'years' if specified
        if years is not None:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
            self.data = self.data[self.data.columns[0]] >= start_date

        # Handle missing values and sort
        self.data = self.data.fillna(method='ffill')
        self.data = self.data.sort_values(by=self.data.columns[0])
        self.data = self.data.reset_index(drop=True)

        logger.info(f"Cleaned data. New shape: {self.data.shape}")
        return self.data

    @staticmethod
    def get_sp500_tickers_from_historical(sp500_path, reference_date=None):
        df = pd.read_csv(sp500_path)
        df['date'] = pd.to_datetime(df['date'])
        if reference_date is None:
            reference_date = df['date'].max()
        else:
            reference_date = pd.to_datetime(reference_date)
        df = df[df['date'] <= reference_date]
        if df.empty:
            raise ValueError('No SP500 composition found for the given date.')
        tickers_str = df.sort_values('date').iloc[-1]['tickers']
        tickers = set(t.strip().upper() for t in tickers_str.split(','))
        return tickers

    @staticmethod
    def build_sp500_membership_df(sp500_path, all_dates, all_tickers):
        sp500_hist = pd.read_csv(sp500_path)
        sp500_hist['date'] = pd.to_datetime(sp500_hist['date'])
        # Build a DataFrame: index=all_dates, columns=all_tickers, values=True/False
        membership = pd.DataFrame(False, index=all_dates, columns=all_tickers)
        for idx, row in sp500_hist.iterrows():
            tickers = set(t.strip().upper() for t in row['tickers'].split(','))
            # All dates from this row's date up to the next row's date (or end)
            start_date = row['date']
            if idx + 1 < len(sp500_hist):
                end_date = sp500_hist.iloc[idx + 1]['date']
            else:
                end_date = all_dates[-1] if len(all_dates) > 0 else start_date
            mask = (all_dates >= start_date) & (all_dates < end_date)
            membership.loc[mask, list(tickers & set(all_tickers))] = True
        return membership

    @staticmethod
    def load_all_data(folder_path, clean=False, years=None, sp500_path=None, sp500_reference_date=None):
        """
        Load and process all CSV files in the specified folder.
        Only include stocks in the SP500 list if sp500_path is provided.
        Ensures date alignment, reindexing, and filling missing values.
        Optionally applies the clean method with a years filter.
        Returns a single DataFrame with all data and a SP500 membership DataFrame.
        """
        import os
        all_data = []
        sp500_tickers = None
        all_dates = None
        if sp500_path is not None:
            try:
                sp500_hist = pd.read_csv(sp500_path)
                sp500_hist['date'] = pd.to_datetime(sp500_hist['date'])
                # Use the latest date's tickers for filtering
                latest_tickers_str = sp500_hist.sort_values('date').iloc[-1]['tickers']
                sp500_tickers = set(t.strip().upper() for t in latest_tickers_str.split(','))
            except Exception as e:
                logger.warning(f"Could not load SP500 tickers from {sp500_path}: {e}")
                sp500_hist = None
                sp500_tickers = None

        # Iterate through all CSV files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                stock_symbol = os.path.splitext(file_name)[0].upper()
                if sp500_tickers is not None and stock_symbol not in sp500_tickers:
                    print(f"Skipping {file_name}: Stock not in SP500.")
                    continue  # Skip stocks not in SP500
                file_path = os.path.join(folder_path, file_name)
                try:
                    data = pd.read_csv(file_path)
                    # Always parse the first column as DD-MM-YYYY
                    date_col = data.columns[0]
                    data[date_col] = pd.to_datetime(data[date_col], format='%d-%m-%Y', errors='coerce')
                    data = data.dropna(subset=[data.columns[0]])  # Drop rows with invalid dates

                    # Apply the clean method if requested
                    if clean:
                        start_date = pd.Timestamp.now() - pd.DateOffset(years=years) if years else None
                        if start_date:
                            data = data[data[data.columns[0]] >= start_date]

                    # Skip files with no valid rows
                    if data.empty:
                        continue

                    # Check for _Adjusted Close column and skip if any value is negative, or if more than 20% are zero or missing
                    adj_close_cols = [col for col in data.columns if col.strip() == 'Adjusted Close' or col.strip() == '_Adjusted Close']
                    if not adj_close_cols:
                        logger.warning(f"Skipping {file_name}: No 'Adjusted Close' column found.")
                        continue
                    adj_close_col = adj_close_cols[0]
                    adj_close_series = pd.to_numeric(data[adj_close_col], errors='coerce')
                    if (adj_close_series < 0).any():
                        logger.warning(f"Skipping {file_name}: At least one 'Adjusted Close' value is negative.")
                        continue
                    zero_or_nan = ((adj_close_series.fillna(0) == 0) | adj_close_series.isna()).sum()
                    frac_zero_or_nan = zero_or_nan / len(adj_close_series)
                    if frac_zero_or_nan > 0.2:
                        logger.warning(f"Skipping {file_name}: More than 20% of 'Adjusted Close' values are zero or missing.")
                        continue

                    # Volume checks: skip if any negative, or if more than 10% are zero or missing
                    volume_cols = [col for col in data.columns if col.strip() == 'Volume' or col.strip() == '_Volume']
                    if not volume_cols:
                        logger.warning(f"Skipping {file_name}: No 'Volume' column found.")
                        continue
                    volume_col = volume_cols[0]
                    volume_series = pd.to_numeric(data[volume_col], errors='coerce')
                    if (volume_series < 0).any():
                        logger.warning(f"Skipping {file_name}: At least one 'Volume' value is negative.")
                        continue
                    zero_or_nan_vol = ((volume_series.fillna(0) == 0) | volume_series.isna()).sum()
                    frac_zero_or_nan_vol = zero_or_nan_vol / len(volume_series)
                    if frac_zero_or_nan_vol > 0.1:
                        logger.warning(f"Skipping {file_name}: More than 10% of 'Volume' values are zero or missing.")
                        continue

                    # Set the Date column as the index
                    data = data.set_index(data.columns[0])

                    # Rename columns to include the stock symbol (from the file name)
                    data = data.add_prefix(f"{stock_symbol}_")

                    all_data.append(data)
                except Exception as e:
                    logger.error(f"Failed to process {file_name}: {e}")

        # Combine all dataframes on the Date index
        if all_data:
            combined_data = pd.concat(all_data, axis=1, join='outer')  # Use outer join to include all dates
            combined_data = combined_data.sort_index()
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')  # Fill missing values
            all_dates = combined_data.index
            logger.info(f"Loaded and combined data from {len(all_data)} files. Final shape: {combined_data.shape}")
        else:
            combined_data = pd.DataFrame()
            logger.warning("No valid data loaded from the folder.")
            all_dates = pd.DatetimeIndex([])

        # Build membership DataFrame
        sp500_membership = None
        if sp500_path is not None and all_dates is not None and len(all_dates) > 0:
            sp500_membership = OHLCVDataLoader.build_sp500_membership_df(sp500_path, all_dates, list(combined_data.columns.str.split('_').str[0].unique()))

        return combined_data, sp500_membership
