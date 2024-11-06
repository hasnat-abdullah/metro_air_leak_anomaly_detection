import os

# Content for each file
file_contents = {
    "src/data_preprocessing.py": """import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    \"\"\"Load data from a CSV file.\"\"\"
    return pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

def preprocess_data(df):
    \"\"\"Clean and scale the data.\"\"\"
    df = df.fillna(method='ffill').fillna(method='bfill')
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler
""",
    "src/feature_engineering.py": """def create_rolling_features(df, window=5):
    \"\"\"Generate rolling mean and std dev features for selected columns.\"\"\"
    rolling_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Motor_current', 'Oil_temperature']
    for col in rolling_cols:
        df[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
        df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
    df = df.dropna()
    return df
""",
    "src/models/isolation_forest.py": """from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.decision_function(X)
""",
    "src/models/autoencoder.py": """import tensorflow as tf
from tensorflow.keras import layers, models

class AutoencoderModel:
    def __init__(self, input_dim):
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X_train, epochs=50, batch_size=32):
        self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    def predict(self, X):
        reconstructions = self.model.predict(X)
        return tf.reduce_mean(tf.square(X - reconstructions), axis=1)
""",
    "src/models/lstm.py": """import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def compute_error(self, X, y):
        preds = self.predict(X)
        return np.mean(np.square(y - preds), axis=1)
""",
    "src/evaluation.py": """from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=-1)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    return {"precision": precision, "recall": recall, "f1_score": f1}
""",
    "src/main.py": """import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_rolling_features
from src.models.isolation_forest import IsolationForestModel
from src.evaluation import evaluate_model

def main():
    data = load_data('data/raw/compressor_data.csv')
    data, scaler = preprocess_data(data)
    data = create_rolling_features(data)
    X = data.drop(columns=['timestamp'])

    model = IsolationForestModel(contamination=0.01)
    model.fit(X)
    predictions = model.predict(X)

    y_true = data['labels'] if 'labels' in data else None
    if y_true is not None:
        results = evaluate_model(y_true, predictions)
        print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
""",
}


def fill_files(file_contents):
    """Fills existing files with specified contents if they are empty."""
    for filepath, content in file_contents.items():
        if os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Filled content in {filepath}")
        else:
            print(f"{filepath} does not exist; please create it first.")


if __name__ == "__main__":
    fill_files(file_contents)
    print("\nContent filled where necessary.")
