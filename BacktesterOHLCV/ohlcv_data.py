import pandas as pd
from debug import logger

class OHLCVDataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

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
            self.data = self.data[self.data[self.data.columns[0]] >= start_date]

        # Handle missing values and sort
        self.data = self.data.fillna(method='ffill')
        self.data = self.data.sort_values(by=self.data.columns[0])
        self.data = self.data.reset_index(drop=True)

        logger.info(f"Cleaned data. New shape: {self.data.shape}")
        return self.data
