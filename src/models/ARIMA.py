"""
ARIMA (AutoRegressive Integrated Moving Average):
- A statistical model for time series forecasting and anomaly detection.
- Combines autoregressive (AR), differencing (I), and moving average (MA) components to model temporal patterns.

Key Concepts:
- Autoregressive (AR): Models the relationship between current and past values.
- Integrated (I): Applies differencing to make the time series stationary.
- Moving Average (MA): Models the relationship between current observations and residuals.

Advantages:
- Effective for univariate time series with trends or seasonality.
- Captures both short-term dependencies and long-term patterns.
- Well-suited for forecasting and anomaly detection.

Applications:
- Demand forecasting in retail.
- Anomaly detection in sensor data.
- Financial market analysis and prediction.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.models.base_model import UnsupervisedModel


class ARIMAModel(UnsupervisedModel):
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.models = {}

    def train(self, X):
        for col in X.columns:
            series = X[col].dropna()
            try:
                model = ARIMA(series, order=self.order)
                self.models[col] = model.fit()
            except Exception as e:
                print(f"Failed to fit ARIMA for column {col}: {e}")

    def predict(self, X):
        residuals = pd.DataFrame()
        for col, model in self.models.items():
            series = X[col].dropna()
            fitted_values = model.fittedvalues.reindex_like(series).fillna(0)
            residuals[col] = series - fitted_values
        return residuals.abs().sum(axis=1)