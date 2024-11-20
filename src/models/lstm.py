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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from .base_model import UnsupervisedModel

class LSTMModel(UnsupervisedModel):
    def __init__(self, input_dim, time_steps=10, latent_dim=64):
        self.time_steps = time_steps
        self.model = Sequential([
            LSTM(latent_dim, activation="relu", input_shape=(time_steps, input_dim), return_sequences=False),
            Dense(input_dim)
        ])
        self.model.compile(optimizer="adam", loss="mse")

    def _reshape_data(self, X):
        """Reshape data for LSTM input."""
        n_samples = X.shape[0] - self.time_steps + 1
        return np.array([X[i:i + self.time_steps] for i in range(n_samples)])

    def train(self, X_train):
        X_train_reshaped = self._reshape_data(X_train)
        self.model.fit(X_train_reshaped, X_train[self.time_steps - 1:], epochs=10, batch_size=128, verbose=0)

    def predict(self, X_test):
        X_test_reshaped = self._reshape_data(X_test)
        reconstructions = self.model.predict(X_test_reshaped)
        return np.mean((X_test[self.time_steps - 1:] - reconstructions) ** 2, axis=1)  # Reconstruction error