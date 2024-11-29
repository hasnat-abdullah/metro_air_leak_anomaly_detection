"""
Prophet with Residual Analysis:
- A forecasting model designed for time series with trends, seasonality, and holidays.
- Uses residual analysis to detect anomalies by comparing observed and predicted values.

Key Concepts:
- Decomposable Model: Separates trend, seasonality, and holidays for forecasting.
- Residuals: Differences between actual and predicted values used for anomaly detection.
- Bayesian Framework: Handles missing data and outliers during training.

Advantages:
- Easy to use with interpretable components.
- Robust to missing data and outliers.
- Effective for time series with complex patterns.

Applications:
- Forecasting demand in retail and e-commerce.
- Anomaly detection in financial and operational metrics.
- Monitoring seasonal patterns in environmental data.
"""

from prophet import Prophet
from src.models.base_model import UnsupervisedModel


class ProphetModel(UnsupervisedModel):
    def __init__(self):
        self.model = Prophet()

    def train(self, X):
        X.columns = ["ds", "y"]
        self.model.fit(X)

    def predict(self, X):
        forecast = self.model.predict(X)
        residuals = X["y"] - forecast["yhat"]
        return residuals.abs()