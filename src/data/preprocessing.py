import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime

class Preprocessor:
    def __init__(self, target_column="_status", timestamp_column="_timestamp"):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.scaler = None
        self.pca = None

    def clean_data(self, df):
        # Handle missing values
        df = df.dropna()  # Drop rows with missing values (or use interpolation if needed)

        # Remove duplicates
        df = df.drop_duplicates()

        return df

    def engineer_features(self, df):
        # Extract time-based features
        df["hour"] = pd.to_datetime(df[self.timestamp_column]).dt.hour
        df["day_of_week"] = pd.to_datetime(df[self.timestamp_column]).dt.dayofweek
        df["month"] = pd.to_datetime(df[self.timestamp_column]).dt.month

        # Drop original timestamp column
        df = df.drop(columns=[self.timestamp_column])

        return df

    def scale_features(self, X):
        if self.scaler is None:
            self.scaler = RobustScaler()  # More robust to outliers
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def reduce_dimensionality(self, X, n_components=10):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = self.pca.transform(X)

        return X_reduced

    def preprocess(self, df):
        # Data Cleaning
        df = self.clean_data(df)

        # Separate features and target
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Feature Engineering
        X = self.engineer_features(X)

        # Feature Scaling
        X_scaled = self.scale_features(X)

        # Dimensionality Reduction
        X_reduced = self.reduce_dimensionality(X_scaled)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test