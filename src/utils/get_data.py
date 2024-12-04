import pandas as pd

from config import TIME_COLUMN, CSV_FILE_PATH, VALUE_COLUMN
from src.data.loader import CSVDataLoader


def get_data(one_data_in_x_minutes=""):
    data_loader = CSVDataLoader(file_path=CSV_FILE_PATH,
                                usecols=[TIME_COLUMN, VALUE_COLUMN])
    data = data_loader.load_data()

    # SAMPLE DATA
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)
    start_time = "2023-12-01 00:00:00"
    end_time = "2024-02-29 04:55:02"
    data = data.loc[start_time:end_time]
    # Step 2: Resample to one row per minute (mean for numerical values)
    data = data.resample(one_data_in_x_minutes).mean().dropna()
    print(len(data))
    data.reset_index(inplace=True)
    data.rename(columns={"index": 'time'}, inplace=True)
    return data