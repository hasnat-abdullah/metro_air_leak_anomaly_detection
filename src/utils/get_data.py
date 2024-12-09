import pandas as pd

from config import TIME_COLUMN, CSV_FILE_PATH, VALUE_COLUMN, COLUMN_LIST_TO_IMPORT
from src.data.loader import CSVDataLoader


def get_data(one_data_in_x_minutes=""):
    data_loader = CSVDataLoader(file_path=CSV_FILE_PATH, usecols=COLUMN_LIST_TO_IMPORT)
    data = data_loader.load_data()

    print(data.columns)
    # SAMPLE DATA
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
    data.set_index(TIME_COLUMN, inplace=True)
    if one_data_in_x_minutes:
        # Step 2: Resample to one row per minute (mean for numerical values)
        data = data.resample(one_data_in_x_minutes).mean().dropna()
    print(f'total raw row - {len(data)}')
    data.reset_index(inplace=True)
    data.rename(columns={"index": TIME_COLUMN}, inplace=True)
    return data

def align_time_in_csv(df: pd.DataFrame, time_column='time', granularity='1800s'):
    """
    Align timestamps in a single CSV file to a specified granularity.

    Parameters:
        df (DataFrame): main data as DataFrame.
        time_column (str): The name of the time column.
        granularity (str): Resampling granularity (e.g., '1s' for seconds).

    Returns:
        pd.DataFrame: DataFrame with aligned timestamps.
    """

    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Round timestamps to the nearest granularity
    df[time_column] = df[time_column].dt.round(granularity)

    # Optionally group by the aligned time to handle duplicates
    df = df.groupby(time_column, as_index=False).mean()  # Taking mean as an example for aggregation
    print(f'rows after grouping by {time_column}: {len(df)}')
    # Assuming `df` is your DataFrame
    df.to_csv('output_filename.csv', index=False)
    return df

if __name__ == "__main__":
    get_data(one_data_in_x_minutes="")