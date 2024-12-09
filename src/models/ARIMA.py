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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from config import TIME_COLUMN, VALUE_COLUMN


class ARIMAModel:
    def __init__(self,time_column, value_column, order=(1, 1, 1)):
        """
        Initialize ARIMA model with specified (p, d, q) parameters.
        :param order: (p, d, q) tuple representing the ARIMA model order
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.time_column = time_column
        self.value_column = value_column

    def train(self, df: pd.DataFrame):
        """Train the ARIMA model on the provided data."""
        print("Training ARIMA model...")
        # Ensure 'time' is datetime and set as index
        print(df)
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df.set_index(self.time_column, inplace=True)

        # Train ARIMA model on the 'Oxygen' column
        self.model = ARIMA(df['Oxygen'], order=self.order)
        self.fitted_model = self.model.fit()

        print(f"ARIMA model trained with order {self.order}")

    def predict(self, steps=10):
        """Make predictions using the trained ARIMA model."""
        print(f"Making predictions for {steps} steps...")
        self.predictions = self.fitted_model.forecast(steps=steps)
        return self.predictions

    def evaluate(self, df: pd.DataFrame):
        """Evaluate the ARIMA model using Mean Squared Error."""
        # Predict using the model on the same dataset
        predictions = self.fitted_model.predict(start=0, end=len(df)-1)

        # Compute Mean Squared Error
        mse = mean_squared_error(df['Oxygen'], predictions)
        print(f"Mean Squared Error: {mse}")
        return mse

    def visualize(self, df: pd.DataFrame):
        """Visualize the actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Oxygen'], label='Actual', color='blue')
        plt.plot(df.index, self.fitted_model.predict(start=0, end=len(df)-1), label='Predicted', color='red')
        plt.xlabel(self.time_column)
        plt.ylabel('Oxygen')
        plt.title('ARIMA Model: Actual vs Predicted Oxygen Levels')
        plt.legend()
        plt.show()

    def run_pipeline(self, df: pd.DataFrame, time_column: str, value_column: str, steps=10):
        """Run the ARIMA pipeline."""
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        # Train
        self.train(df)

        # Predict
        predictions = self.predict(steps)

        # Evaluate the model
        mse = self.evaluate(df)

        # Visualize the results
        self.visualize(df)
        print(f"Predictions: {predictions}")
        print(f"Mean Squared Error: {mse}")

        return df, predictions, mse


if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data(one_data_in_x_minutes="20T")
    arima_model = ARIMAModel(TIME_COLUMN, VALUE_COLUMN,order=(1, 1, 1))
    result_df, predictions, mse = arima_model.run_pipeline(data,time_column=TIME_COLUMN, value_column=VALUE_COLUMN, steps=3)

    # Output results

    print(result_df)