import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.base_model import UnsupervisedModel


class LSTMAutoencoder(UnsupervisedModel):
    def __init__(self, input_dim, timesteps, latent_dim):
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.encoder, self.decoder = self._build_model()

    def _build_model(self):
        # Encoder
        inputs = layers.Input(shape=(self.timesteps, self.input_dim))
        encoded = layers.LSTM(self.latent_dim, activation='relu')(inputs)

        # Decoder
        decoded_input = layers.RepeatVector(self.timesteps)(encoded)
        decoded = layers.LSTM(self.input_dim, return_sequences=True, activation='relu')(decoded_input)

        encoder = models.Model(inputs, encoded, name="encoder")
        autoencoder = models.Model(inputs, decoded, name="autoencoder")
        return encoder, autoencoder

    def train(self, X):
        self.decoder.compile(optimizer='adam', loss='mse')
        self.decoder.fit(X, X, epochs=10, batch_size=32, verbose=0)

    def predict(self, X):
        reconstructed = self.decoder.predict(X)
        reconstruction_error = tf.reduce_mean(tf.square(X - reconstructed), axis=1).numpy()
        return reconstruction_error