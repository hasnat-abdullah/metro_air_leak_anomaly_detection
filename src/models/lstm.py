"""
Long Short-Term Memory (LSTM)

LSTM is a type of recurrent neural network (RNN) architecture designed to capture long-term dependencies in sequential data. Unlike traditional RNNs, LSTMs are capable of learning and remembering over long sequences without suffering from the vanishing gradient problem, making them particularly effective for time series forecasting, natural language processing, and anomaly detection tasks.

Key Concepts:
- **Gates**: LSTM units consist of three gates (input, forget, and output) that control the flow of information within the unit:
  - **Forget Gate**: Decides what information from the previous state should be discarded.
  - **Input Gate**: Controls what new information is added to the memory cell.
  - **Output Gate**: Determines what the current state of the memory cell will output.
- **Cell State**: The memory cell state stores the long-term memory of the network, which is updated through the forget and input gates.
- **Hidden State**: The hidden state is the output of the LSTM unit, which is used as input for the next unit in the sequence.

Working:
1. **Initialization**: The initial hidden state and cell state are usually initialized to zeros or learned during training.
2. **Memory Update**: For each timestep in the sequence, the LSTM updates its memory by applying the three gates. The forget gate decides how much of the previous state should be discarded, the input gate controls how much new information to store, and the output gate determines the next hidden state.
3. **Sequence Processing**: The LSTM processes sequences of data one timestep at a time, retaining important information in the memory to make predictions or learn patterns in the data.

Advantages:
- **Capturing Long-Term Dependencies**: LSTMs are capable of learning long-range dependencies in sequential data, making them highly suitable for tasks like time series prediction, speech recognition, and anomaly detection in sequential data.
- **Avoiding Vanishing Gradient Problem**: LSTMs mitigate the vanishing gradient problem by maintaining a memory cell that stores information over time, allowing them to retain information for much longer durations than traditional RNNs.

Disadvantages:
- **Computationally Expensive**: LSTMs require more parameters and computations compared to standard RNNs, making them computationally expensive, especially for long sequences.
- **Requires Large Datasets**: LSTMs require large amounts of data to train effectively, particularly for tasks like time series forecasting or natural language processing.

Applications:
- **Time Series Forecasting**: LSTMs are widely used for predicting future values in time series data, such as stock prices, sensor readings, and weather patterns.
- **Anomaly Detection in Sequential Data**: LSTMs can detect anomalies in sequential data by learning the normal behavior over time and flagging points that deviate from this learned pattern.
- **Natural Language Processing (NLP)**: LSTMs are used in NLP tasks such as language translation, speech recognition, and sentiment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

from config import TIME_COLUMN, VALUE_COLUMN


class LSTMModel:
    def __init__(self,time_column, value_column, seq_length=10, dropout=0.2, epochs=50, batch_size=32):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.dropout = dropout
        self.time_column = time_column
        self.value_column = value_column

    def preprocess_data(self, df):
        """Preprocess the data by scaling and creating sequences."""
        # Scale Oxygen values
        df[f'{self.value_column}_scaled'] = self.scaler.fit_transform(df[[self.value_column]])

        # Create sequences for LSTM
        sequences = []
        for i in range(len(df) - self.seq_length):
            seq = df[f'{self.value_column}_scaled'].iloc[i:i + self.seq_length].values
            sequences.append(seq)

        sequences = np.array(sequences)
        return sequences

    def build_model(self):
        """Build the LSTM model."""
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.seq_length, 1), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train(self, sequences):
        """Train the LSTM model."""
        print("Training LSTM model...")
        self.model.fit(
            sequences[:, :-1, np.newaxis],  # Features
            sequences[:, -1],  # Target
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=2
        )
        print("Model training completed.")

    def predict(self, sequences):
        """Predict anomalies using the trained LSTM model."""
        predictions = self.model.predict(sequences[:, :-1, np.newaxis])
        mse = np.mean((predictions - sequences[:, -1])**2, axis=1)

        # Define anomalies based on a threshold (e.g., 3 standard deviations above the mean MSE)
        threshold = np.mean(mse) + 3 * np.std(mse)
        anomalies = mse > threshold
        return anomalies, mse, threshold

    def visualize(self, df, anomalies, anomaly_scores, threshold):
        """Visualize anomalies on the time vs. Oxygen plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(df[self.time_column], df[self.value_column], label=self.value_column, color='blue')
        plt.scatter(
            df[self.time_column][anomalies],
            df[self.value_column][anomalies],
            label='Anomalies',
            color='red'
        )
        plt.axhline(y=threshold, color='green', linestyle='--', label='Anomaly Threshold')
        plt.xlabel(self.time_column)
        plt.ylabel(self.value_column)
        plt.title('Anomaly Detection using LSTM')
        plt.legend()
        plt.show()

    def run_pipeline(self, df, time_column, value_column):
        """Run the LSTM anomaly detection pipeline."""
        sequences = self.preprocess_data(df)

        # Build and train the model
        self.build_model()
        self.train(sequences)

        # Predict anomalies
        anomalies, anomaly_scores, threshold = self.predict(sequences)

        # Add anomaly scores to the original dataframe
        df = df.iloc[self.seq_length:].reset_index(drop=True)
        df['anomaly_score'] = anomaly_scores
        df['anomaly'] = anomalies

        # Visualize anomalies
        self.visualize(df, anomalies, anomaly_scores, threshold)
        print(f"Detected anomalies: {sum(anomalies)}")
        return df, anomalies, anomaly_scores


# Usage
if __name__ == "__main__":
    # Assuming 'data' is your dataframe
    from src.utils.get_data import get_data
    data = get_data("10T")  # Replace with your actual data source

    lstm_model = LSTMModel(time_column=TIME_COLUMN, value_column=VALUE_COLUMN, seq_length=10, epochs=50, batch_size=32)
    result_df, anomalies, anomaly_scores = lstm_model.run_pipeline(data, TIME_COLUMN, VALUE_COLUMN)

    # Output results
    print(result_df.head())