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


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model with specified (p, d, q) parameters.
        :param order: (p, d, q) tuple representing the ARIMA model order
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.predictions = None

    def train(self, df: pd.DataFrame):
        """Train the ARIMA model on the provided data."""
        print("Training ARIMA model...")
        # Ensure 'time' is datetime and set as index
        print(df)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

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
        plt.xlabel('Time')
        plt.ylabel('Oxygen')
        plt.title('ARIMA Model: Actual vs Predicted Oxygen Levels')
        plt.legend()
        plt.show()

    def run_pipeline(self, df: pd.DataFrame, steps=10):
        """Run the ARIMA pipeline."""
        # Preprocess the data: ensure 'time' column is datetime
        df['time'] = pd.to_datetime(df['time'])
        # Train the ARIMA model
        self.train(df)

        # Predict the next steps
        predictions = self.predict(steps)

        # Evaluate the model
        mse = self.evaluate(df)

        # Visualize the results
        self.visualize(df)

        return predictions, mse, df


if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data(one_data_in_x_minutes="20T")
    arima_model = ARIMAModel(order=(1, 1, 1))
    predictions, mse, result_df = arima_model.run_pipeline(data, steps=3)

    # Output results
    print(f"Predictions: {predictions}")
    print(f"Mean Squared Error: {mse}")
    print(result_df)