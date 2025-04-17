import os
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

    @staticmethod
    def load_all_data(folder_path, clean=False, years=None):
        """
        Load and process all CSV files in the specified folder.
        Ensures date alignment, reindexing, and filling missing values.
        Optionally applies the clean method with a years filter.
        Returns a single DataFrame with all data.
        """
        import os
        all_data = []

        # Iterate through all CSV files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
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

                    # Set the Date column as the index
                    data = data.set_index(data.columns[0])

                    # Rename columns to include the stock symbol (from the file name)
                    stock_symbol = os.path.splitext(file_name)[0]
                    data = data.add_prefix(f"{stock_symbol}_")

                    all_data.append(data)
                except Exception as e:
                    logger.error(f"Failed to process {file_name}: {e}")

        # Combine all dataframes on the Date index
        if all_data:
            combined_data = pd.concat(all_data, axis=1, join='outer')  # Use outer join to include all dates
            combined_data = combined_data.sort_index()
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')  # Fill missing values
            logger.info(f"Loaded and combined data from {len(all_data)} files. Final shape: {combined_data.shape}")
        else:
            combined_data = pd.DataFrame()
            logger.warning("No valid data loaded from the folder.")

        return combined_data
