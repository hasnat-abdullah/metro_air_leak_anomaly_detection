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