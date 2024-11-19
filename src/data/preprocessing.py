import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from datetime import datetime

class Preprocessor:
    def __init__(self, target_column="_status", timestamp_column="_timestamp", n_features=10):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.scaler = None
        self.pca = None
        self.selector = None
        self.n_features = n_features  # Number of features to select

    def clean_data(self, df):
        df = df.dropna()  # Drop rows with missing values (or use interpolation if needed)
        df = df.drop_duplicates()

        return df

    def engineer_features(self, df):
        df = df.drop(columns=[self.timestamp_column])

        # # Extract time-based features
        # df["hour"] = pd.to_datetime(df[self.timestamp_column]).dt.hour
        # df["day_of_week"] = pd.to_datetime(df[self.timestamp_column]).dt.dayofweek
        # df["month"] = pd.to_datetime(df[self.timestamp_column]).dt.month

        return df

    def scale_features(self, X):
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Shift all values to be non-negative (if required by chi2)
        if (X_scaled < 0).any():
            X_scaled = X_scaled - np.min(X_scaled, axis=0)  # Shift features to make all values non-negative

        return X_scaled

    def reduce_dimensionality(self, X, n_components=10):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = self.pca.transform(X)

        return X_reduced

    def select_features(self, X, y):
        if self.selector is None:
            self.selector = SelectKBest(chi2, k=self.n_features)
            X_selected = self.selector.fit_transform(X, y)
        else:
            X_selected = self.selector.transform(X)

        return X_selected

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

        # Feature Selection
        X_selected = self.select_features(X_scaled, y)

        # Dimensionality Reduction
        X_reduced = self.reduce_dimensionality(X_selected)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test