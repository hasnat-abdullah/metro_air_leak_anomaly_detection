"""
Deep Autoencoder

A Deep Autoencoder is a type of artificial neural network used for unsupervised learning and dimensionality reduction. It consists of an encoder and a decoder, both of which are neural networks. The encoder compresses the input into a latent-space representation (a lower-dimensional encoding), and the decoder reconstructs the input from this representation. The network is trained to minimize the difference between the input and the reconstructed output.

Key Concepts:
- Encoder: The part of the autoencoder that learns to map the input data into a lower-dimensional latent space (encoding).
- Decoder: The part of the autoencoder that reconstructs the input data from the encoded representation.
- Latent Space: A compressed, lower-dimensional representation of the input data.
- Reconstruction Loss: The error between the original input and the output after reconstruction. The model is trained to minimize this loss.

In the context of anomaly detection:
- The autoencoder is trained to learn the normal patterns of the data. When it is presented with new data, it attempts to reconstruct it.
- Anomalies are detected by measuring the reconstruction error: If the reconstruction error is high, the data point is considered an anomaly since the autoencoder struggles to reconstruct outliers that differ from the normal patterns.

Key Features:
- Unsupervised learning: Does not require labeled data.
- Anomaly detection: High reconstruction error indicates an anomalous data point.
- Nonlinear feature extraction: Can learn complex, nonlinear representations of the data.
- Flexible: Can be adapted for various types of data, including time-series and images.

Parameters:
- Encoder architecture: The structure of the neural network used to encode the input into a latent space (e.g., number of layers and neurons per layer).
- Decoder architecture: The structure of the neural network used to decode the latent space representation back to the input space.
- Latent dimension: The size of the latent space, which determines the level of compression applied to the input data.
- Activation functions: Functions like ReLU, sigmoid, or tanh used in the neural network layers.

Applications:
- Anomaly detection in sensor data, network traffic, and financial transactions.
- Dimensionality reduction and feature learning.
- Image denoising and data compression.
"""

from tensorflow.keras import layers, models

from src.models.base_model import UnsupervisedModel


class DeepAutoEncoder(UnsupervisedModel):
    def __init__(self, input_dim, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.latent_dim, activation='relu')(input_layer)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(encoded)

        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def train(self, X):
        self.model.fit(X, X, epochs=50, batch_size=256, validation_split=0.2)

    def predict(self, X):
        return self.model.predict(X)