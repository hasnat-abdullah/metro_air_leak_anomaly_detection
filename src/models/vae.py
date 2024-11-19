"""Variational Autoencoder """

import tensorflow as tf
from tensorflow.keras import layers, models
from .base_model import UnsupervisedModel


class VAEModel(UnsupervisedModel):
    def __init__(self, input_dim, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        z_mean = layers.Dense(self.latent_dim)(input_layer)
        z_log_var = layers.Dense(self.latent_dim)(input_layer)

        z = layers.Lambda(self.sampling)([z_mean, z_log_var])

        encoder = models.Model(input_layer, [z_mean, z_log_var, z])

        decoder_h = layers.Dense(64, activation='relu')
        decoder_mean = layers.Dense(self.input_dim, activation='sigmoid')

        z_input = layers.Input(shape=(self.latent_dim,))
        h_decoded = decoder_h(z_input)
        x_decoded_mean = decoder_mean(h_decoded)

        decoder = models.Model(z_input, x_decoded_mean)

        vae = models.Model(input_layer, decoder(encoder(input_layer)[2]))
        vae.compile(optimizer='adam', loss='mse')

        return vae

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def train(self, X):
        self.model.fit(X, X, epochs=50, batch_size=256, validation_split=0.2)

    def predict(self, X):
        return self.model.predict(X)