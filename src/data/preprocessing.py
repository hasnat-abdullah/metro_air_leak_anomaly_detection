import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, data: pd.DataFrame, time_column: str, value_column: str):
        self.time_column = time_column
        self.value_column = value_column
        self.df = data
        self.freq = None

    def infer_frequency(self) -> None:
        time_deltas = self.df.index.to_series().diff().dropna()
        self.freq = time_deltas.mode()[0]
        print(f"Inferred frequency: {self.freq}")

    def handle_duplicates(self) -> None:
        """Handles duplicate timestamps by aggregating values."""
        if self.df.index.duplicated().any():
            print("Handling duplicate timestamps...")
            self.df = self.df.groupby(self.df.index).mean()

    def reindex_time_series(self) -> None:
        if self.freq is None:
            raise ValueError("Frequency must be inferred or set before reindexing.")
        self.handle_duplicates()
        full_index = pd.date_range(self.df.index.min(), self.df.index.max(), freq=self.freq)
        self.df = self.df.reindex(full_index)
        self.df = self.df.dropna()

    def drop_missing_values(self) -> None:
        self.df = self.df.dropna()

    def add_time_features(self) -> None:
        self.df['hour'] = self.df.index.hour
        self.df['weekday'] = self.df.index.weekday

    def add_rolling_features(self, window: int = 24) -> None:
        self.df[f'rolling_mean_{window}'] = self.df[self.value_column].rolling(window=window).mean()
        self.df[f'rolling_std_{window}'] = self.df[self.value_column].rolling(window=window).std()

    def add_diff_features(self) -> None:
        self.df['diff'] = self.df[self.value_column].diff()

    def scale_features(self) -> pd.DataFrame:
        """
        Scales the features using StandardScaler and ensures alignment of index and scaled data.
        """
        scaler = StandardScaler()
        non_na_data = self.df.dropna()
        scaled_data = scaler.fit_transform(non_na_data)
        self.df = pd.DataFrame(scaled_data, index=non_na_data.index, columns=non_na_data.columns)
        return self.df

    def convert_time_column_vale_to_datetime_type(self):
        self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])

    def set_index_by_time_column(self):
        self.df.set_index(self.time_column, inplace=True)

    def preprocess(self) -> pd.DataFrame:
        print("convert to datetime type...")
        self.convert_time_column_vale_to_datetime_type()
        print("set index by time column...")
        self.set_index_by_time_column()
        print("Inferring frequency...")
        self.infer_frequency()
        print("Reindexing time series...")
        self.reindex_time_series()
        print("Dropping missing values...")
        self.drop_missing_values()
        print("Adding time features...")
        self.add_time_features()
        print("Adding rolling features...")
        self.add_rolling_features()
        print("Adding diff features...")
        self.add_diff_features()
        print("Scaling features...")
        self.scale_features()
        print("Preprocessing complete.")
        print(self.df)
        print(self.df.columns)
        return self.df

    def save_preprocessed_data(self, output_path: str) -> None:
        self.df.to_csv(output_path)
        print(f"Preprocessed data saved to {output_path}")