"""
Autoencoder for Anomaly Detection

An autoencoder is a type of artificial neural network used to learn efficient representations of data, typically for the purpose of dimensionality reduction or feature learning. It is composed of two parts: the encoder and the decoder.

- Encoder: Compresses the input data into a lower-dimensional latent space (bottleneck).
- Decoder: Reconstructs the original input from the compressed representation.

In the context of anomaly detection, the autoencoder is trained to reconstruct the normal data well. During testing, if an input sample is significantly different from the normal data, the reconstruction error (difference between the original input and the reconstructed input) will be high, indicating that the sample is anomalous.

Key Features:
- Unsupervised learning: It does not require labeled data.
- Anomaly detection: Can detect novel or unseen patterns that differ from the training data.
- Robust to noise: Capable of learning complex, non-linear representations.

Applications:
- Fraud detection
- Image and video anomaly detection
- Industrial equipment monitoring
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from src.models.base_model import UnsupervisedModel

class AutoEncoder(UnsupervisedModel):
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def train(self, X_train):
        self.model.fit(X_train, X_train, epochs=50, batch_size=128, validation_split=0.1, verbose=0)

    def predict(self, X_test):
        reconstructions = self.model.predict(X_test)
        return np.mean((X_test - reconstructions) ** 2, axis=1)  # Reconstruction error
