"""
S-H-ESD (Seasonal Hybrid Extreme Studentized Deviate):
- A statistical anomaly detection method tailored for time series data with seasonality.
- Extends the classical ESD test by incorporating seasonal decomposition into the analysis.

Key Concepts:
- Seasonality: Identifies periodic patterns in time series data.
- Extreme Studentized Deviate (ESD) Test: Detects outliers iteratively using statistical measures.
- Hybrid Model: Combines robust statistical techniques for effective anomaly detection.

Advantages:
- Handles seasonal components and trends in time series data.
- Simple implementation for univariate time series.
- Effective for noisy datasets with regular patterns.

Applications:
- Monitoring seasonal metrics like sales or website traffic.
- Detecting anomalies in environmental and sensor data.
"""
import pandas as pd
from pyculiarity import detect_ts
from src.models.base_model import UnsupervisedModel


class SHESDModel(UnsupervisedModel):
    def __init__(self, alpha=0.05, max_anoms=0.05):
        self.alpha = alpha
        self.max_anoms = max_anoms

    def train(self, X):
        # S-H-ESD is stateless; no training is required.
        pass

    def predict(self, X):
        results = []
        for col in X.columns:
            # Prepare the input DataFrame for detect_ts
            df = pd.DataFrame({
                'timestamp': X.index,
                'value': X[col]
            })
            # Convert 'timestamp' to string before passing it to detect_ts
            df['timestamp'] = df['timestamp'].astype(str)

            anomalies = detect_ts(df, alpha=self.alpha, max_anoms=self.max_anoms)
            results.append(anomalies["anoms"].index)
        return results