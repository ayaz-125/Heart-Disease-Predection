# This is for the data preprocessing; different preprocessing techniques will be used in this file.

import numpy as np
import pandas as pd
import os
from src.logger import logging

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames specific columns in the DataFrame."""
    column_mapping = {
        'BP': 'Blood Pressure',
        'FBS over 120': 'Sugre Blood Value',
        'Max HR': 'Max heart rate',
        'ST depression': 'Segment Depression'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    return df

def main():
    try:
        # Fetch the data from data/raw
        logging.info("Fetching the data from the raw folder which is inside the data folder")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info("Data loaded properly")

        # Rename columns in the datasets
        train_renamed_data = rename_columns(train_data)
        test_renamed_data = rename_columns(test_data)

        # Store the data inside data/interim
        data_path = os.path.join("./data", 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_renamed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_renamed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logging.info("Processed data saved to %s", data_path)

    except Exception as e:
        logging.error("An error occurred during the data preprocessing: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
