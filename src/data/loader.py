from abc import ABC, abstractmethod
import pandas as pd
from sqlalchemy import create_engine


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self):
        """Method to be implemented by specific data loaders."""
        pass


class CSVDataLoader(DataLoader):
    def __init__(self, file_path, usecols=None):
        self.file_path = file_path
        self.usecols = usecols

    def load_data(self):
        return pd.read_csv(self.file_path, usecols=self.usecols)


class PostgreSQLDataLoader(DataLoader):
    def __init__(self, db_url: str, query: str):
        self.engine = create_engine(db_url)
        self.query = query

    def load_data(self):
        return pd.read_sql_query(self.query, con=self.engine)


class InfluxDBDataLoader(DataLoader):
    def __init__(self, client, query):
        self.client = client
        self.query = query

    def load_data(self):
        result = self.client.query(self.query)
        return pd.DataFrame(list(result.get_points()))