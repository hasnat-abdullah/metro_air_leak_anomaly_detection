"""
DeepAD (Deep Anomaly Detection):
- A deep learning-based anomaly detection framework using dense or recurrent neural networks.
- Trains on normal data and learns to reconstruct input sequences with minimal error.
- High reconstruction errors during inference indicate potential anomalies.

Key Concepts:
- Reconstruction Errors: Used to measure deviations from normal patterns.
- Deep Neural Networks: Capture complex and high-dimensional patterns in data.
- Unsupervised Learning: No labeled anomalies are needed during training.

Advantages:
- Handles non-linear and high-dimensional data effectively.
- Scalable for large datasets.
- Adaptable to various input types.

Applications:
- Industrial monitoring and predictive maintenance.
- Anomaly detection in network traffic and cybersecurity.
- Healthcare and bioinformatics anomaly detection.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.models.base_model import UnsupervisedModel


class DeepAD(UnsupervisedModel):
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='sigmoid'),
        ])

    def train(self, X):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, X, epochs=10, batch_size=32, verbose=0)

    def predict(self, X):
        reconstruction = self.model.predict(X)
        return ((X - reconstruction) ** 2).mean(axis=1)