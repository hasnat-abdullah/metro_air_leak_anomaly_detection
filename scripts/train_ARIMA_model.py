import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
from memory_profiler import profile

from src.config import DATABASE_URL, QUERY
from src.data.loader import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

class UnsupervisedAnomalyDetectionPipeline:
    def __init__(self, n_features=10):
        self.n_features = n_features

    @staticmethod
    def clean_data(df):
        """Remove duplicates and missing values."""
        df = df.dropna()
        df = df.drop_duplicates()
        return df

    @staticmethod
    def check_stationarity(series):
        """Check stationarity using Augmented Dickey-Fuller test."""
        result = adfuller(series)
        if result[1] > 0.05:
            return False
        return True

    def preprocess_arima(self, series):
        """Preprocess data for ARIMA, ensure stationarity."""
        if not self.check_stationarity(series):
            series = series.diff().dropna()
        return series

    def fit_arima(self, series):
        """Fit ARIMA model and calculate residuals."""
        series = self.preprocess_arima(series)
        try:
            model = ARIMA(series, order=(2, 1, 2))
            model_fit = model.fit()
            residuals = series - model_fit.fittedvalues
        except Exception as e:
            print(f"ARIMA failed for the series: {e}")
            residuals = pd.Series(np.zeros(len(series)), index=series.index)
        return residuals

    def preprocess(self, df):
        """Preprocess data by cleaning and dropping unnecessary columns."""
        df = self.clean_data(df)

        if "_timestamp" in df.columns:
            df = df.drop(columns=["_timestamp"])

        return df

    def evaluate(self, y_true, y_pred):
        """Evaluate the model using precision, recall, and F1 score."""
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    @profile
    def fit_predict(self, df):
        """Main pipeline for ARIMA-based unsupervised anomaly detection."""
        X = self.preprocess(df)
        anomaly_scores = []

        # Process each column one at a time to reduce memory usage
        for col in X.columns:
            print(f"Processing column: {col}")
            residuals = self.fit_arima(X[col])

            # Thresholding based on residuals
            column_anomaly_scores = residuals.abs()
            anomaly_scores.append(column_anomaly_scores)

            # Optionally save residuals immediately to avoid large memory usage
            column_anomaly_scores.to_csv(f"{col}_anomaly_scores.csv", index=False)

        # Concatenate all column anomaly scores
        anomaly_scores = pd.concat(anomaly_scores, axis=1).fillna(0)

        # Sum anomaly scores across all columns and set a threshold for anomalies
        anomaly_scores_sum = anomaly_scores.sum(axis=1)
        threshold = np.percentile(anomaly_scores_sum, 95)  # Top 5% as anomalies
        y_pred = (anomaly_scores_sum > threshold).astype(int)

        return y_pred, anomaly_scores_sum

if __name__ == "__main__":
    loader = DataLoader(DATABASE_URL)
    df = loader.load_data(QUERY)

    y_true = df["_status"] if "_status" in df.columns else None

    if "_status" in df.columns:
        df = df.drop(columns=["_status"])

    pipeline = UnsupervisedAnomalyDetectionPipeline()
    y_pred, anomaly_scores = pipeline.fit_predict(df)

    if y_true is not None:
        pipeline.evaluate(y_true, y_pred)

    df["Anomaly_Score"] = anomaly_scores
    df["Anomaly_Label"] = y_pred
    df.to_csv("anomaly_detection_results.csv", index=False)
    print("Results saved to 'anomaly_detection_results.csv'")