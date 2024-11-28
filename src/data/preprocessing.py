import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self, target_column="_status", timestamp_column="_timestamp", n_features=10, use_smote=True):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.scaler = None
        self.pca = None
        self.selector = None
        self.n_features = n_features
        self.use_smote = use_smote

    def clean_data(self, df):
        df = df.dropna()
        df = df.drop_duplicates()

        return df

    def engineer_features(self, df):
        df = df.drop(columns=[self.timestamp_column])

        # Example: Adding statistical features (moving averages, min, max, etc.)
        for col in df.columns:
            if df[col].dtype != 'object':  # Skip categorical columns
                df[f'{col}_mean'] = df[col].rolling(window=10).mean()
                df[f'{col}_std'] = df[col].rolling(window=10).std()
                df[f'{col}_min'] = df[col].rolling(window=10).min()
                df[f'{col}_max'] = df[col].rolling(window=10).max()

        df = df.dropna()  # Drop rows created by rolling operations that contain NaN
        return df

    def scale_features(self, X):
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert back to DataFrame to preserve column names
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Shift all values to be non-negative (if required by chi2)
        if (X_scaled < 0).any().any():
            X_scaled += abs(X_scaled.min())  # Shift features to make all values non-negative

        return X_scaled

    def reduce_dimensionality(self, X, n_components=10):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = self.pca.transform(X)

        return X_reduced

    def select_features(self, X, y):
        # Ensure X and y are consistent
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert to NumPy array if needed
        if isinstance(y, pd.Series):
            y = y.values

        # Apply SelectKBest
        if self.selector is None:
            self.selector = SelectKBest(mutual_info_classif, k=self.n_features)
            X_selected = self.selector.fit_transform(X, y)
        else:
            X_selected = self.selector.transform(X)

        return X_selected

    def balance_data(self, X_train, y_train):
        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            return X_train_resampled, y_train_resampled
        return X_train, y_train

    def preprocess(self, df):
        # Data Cleaning
        df = self.clean_data(df)

        # Separate features and target
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Feature Engineering
        X = self.engineer_features(X)

        # Align `y` with the rows in `X` (handle rows dropped due to feature engineering)
        y = y.loc[X.index]

        # Feature Scaling
        X_scaled = self.scale_features(X)

        # Feature Selection
        X_selected = self.select_features(X_scaled, y)

        # Dimensionality Reduction
        # X_reduced = self.reduce_dimensionality(X_selected)

        # Split Data into Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance using SMOTE
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)

        return X_train_balanced, X_test, y_train_balanced, y_test


if __name__ == "__main__":
    # Load your dataset (replace this with your actual data loading code)
    df = pd.read_csv('your_dataset.csv')

    # Initialize Preprocessor
    preprocessor = Preprocessor(target_column='_status', timestamp_column='_timestamp', n_features=10, use_smote=True)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
