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