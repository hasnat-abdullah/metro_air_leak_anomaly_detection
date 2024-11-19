import pandas as pd
from sqlalchemy import create_engine

class DataLoader:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def load_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)