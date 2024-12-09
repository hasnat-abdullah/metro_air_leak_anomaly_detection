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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def train(self, train_data: pd.DataFrame):
        """Train the Prophet model."""
        train_data.columns = ["ds", "y"]

        self.model.fit(train_data)

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions."""
        test_data.columns = ["ds"]
        forecast = self.model.predict(test_data)
        return forecast

    def plot_forecast(self, forecast: pd.DataFrame, test: pd.DataFrame):
        """Plot the forecast results vs actual values."""
        plt.figure(figsize=(10, 6))
        plt.plot(test['ds'], test['y'], label='Actual', color='blue')
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='orange')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Prophet Forecast vs Actual')
        plt.legend()
        plt.show()

    def plot_residuals(self, forecast: pd.DataFrame, test: pd.DataFrame):
        """Plot residuals (actual - predicted)."""
        test = test.merge(forecast[['ds', 'yhat']], on='ds')
        residuals = test['y'] - test['yhat_x']
        plt.figure(figsize=(10, 6))
        plt.plot(test['ds'], residuals, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.title('Residuals of Prophet Forecast')
        plt.show()

    def plot_anomalies(self, forecast: pd.DataFrame, test: pd.DataFrame, threshold: float):
        """Visualize anomalies based on residuals."""
        test = test.merge(forecast[['ds', 'yhat']], on='ds')
        residuals = test['y'] - test['yhat_x']
        anomalies = residuals.abs() > threshold
        plt.figure(figsize=(10, 6))
        plt.plot(test['ds'], test['y'], label='Actual', color='blue')
        plt.plot(test['ds'][anomalies], test['y'][anomalies], 'ro', label='Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Anomaly Detection with Threshold={threshold}')
        plt.legend()
        plt.show()

    def visualize(self, forecast: pd.DataFrame, test: pd.DataFrame, threshold: float):
        """Visualize forecast, residuals, and anomalies."""
        self.plot_forecast(forecast, test)
        self.plot_residuals(forecast, test)
        self.plot_anomalies(forecast, test, threshold)

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame)-> dict:
        """Train and evaluate the model with residuals and anomalies."""
        self.train(train)
        forecast = self.predict(test[['ds']])

        # Merge forecast with test set
        test = test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
        test['error'] = test['y'] - test['yhat']
        test['abs_error'] = test['error'].abs()

        # Calculate metrics
        mae = mean_absolute_error(test['y'], test['yhat'])
        rmse = np.sqrt(mean_squared_error(test['y'], test['yhat']))
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        return {
            "mae": mae,
            "rmse": rmse,
            "residuals": test['error'],
            "forecast": test,
            "merged_test": test,
        }

    def run_pipeline(self, df: pd.DataFrame, train_size: float = 0.8, threshold: float = 0.2):
        """
        Run the complete pipeline: train, predict, evaluate, and visualize.
        :param df: The dataframe containing the time series data.
        :param train_size: The fraction of data to use for training (default 80%).
        :param threshold: The threshold for anomaly detection based on residuals.
        """
        train = df.iloc[:int(len(df) * train_size)]
        test = df.iloc[int(len(df) * train_size):]

        results = self.evaluate(train, test)
        self.visualize(results["forecast"], test=results["merged_test"], threshold=threshold)

        return results


if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data(one_data_in_x_minutes="10T")
    data = data.rename(columns={"time": "ds", "Oxygen": "y"})
    prophet_model = ProphetModel()
    # Run the pipeline
    evaluation_results = prophet_model.run_pipeline(data)

    # Print results
    print("\nProphet Model Forecast Evaluation:")
    print(evaluation_results)