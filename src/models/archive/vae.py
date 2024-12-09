"""
Variational Autoencoder (VAE)

Variational Autoencoder (VAE) is a generative model that combines the principles of autoencoders with variational inference. The goal of a VAE is to learn a probabilistic mapping between observed data and a lower-dimensional latent space, from which new data samples can be generated. It differs from standard autoencoders by treating the latent variables as distributions rather than fixed values, making it a powerful tool for generative tasks.

Key Concepts:
- **Autoencoder Architecture**: Like a traditional autoencoder, VAE consists of an encoder and a decoder. The encoder maps the input data to a distribution in the latent space, and the decoder reconstructs the data from the latent variables sampled from that distribution.
- **Latent Space and Variational Inference**: VAE assumes that the data is generated from some hidden (latent) variables, and it uses variational inference to approximate the true posterior distribution of the latent variables given the data. The encoder outputs a mean and variance for each latent variable, and the decoder learns to reconstruct the data from samples drawn from this distribution.
- **Reparameterization Trick**: VAE uses a technique called the "reparameterization trick" to allow for backpropagation during training. Instead of sampling directly from the latent variable distribution, it samples from a standard Gaussian distribution and shifts and scales the result using the mean and variance predicted by the encoder.

Working:
1. **Encoder**: The encoder takes an input and maps it to a distribution (mean and variance) in the latent space. This distribution represents the uncertainty in the encoding of the data.
2. **Reparameterization Trick**: The latent variable is sampled using the mean and variance output by the encoder, enabling the gradient descent optimization of the generative model.
3. **Decoder**: The decoder reconstructs the input data from the latent variable by learning to map the latent space back to the original data space.
4. **Loss Function**: VAE minimizes a loss function composed of two terms: the reconstruction loss (measuring how well the data is reconstructed) and the Kullback-Leibler (KL) divergence (measuring how closely the learned latent distribution matches the prior distribution).

Advantages:
- **Generative Model**: VAE can generate new, unseen data by sampling from the latent space, making it a powerful tool for tasks such as image generation, denoising, and inpainting.
- **Probabilistic Latent Variables**: VAE allows for a probabilistic interpretation of the latent space, enabling more flexible and robust learning of complex data distributions.
- **Scalability**: VAE can be scaled to large datasets and high-dimensional data, such as images or time series.

Disadvantages:
- **Blurry Outputs**: VAE-generated images can sometimes appear blurry due to the use of the mean value in the latent space, which may not capture fine details.
- **Training Instability**: Training VAEs can sometimes be unstable, requiring careful tuning of hyperparameters such as the learning rate and the balance between the reconstruction loss and KL divergence.

Applications:
- **Image Generation**: VAE is widely used in generative tasks, such as generating new images from learned distributions.
- **Anomaly Detection**: VAE can be used for anomaly detection by learning the distribution of normal data and flagging data points that deviate significantly from the learned distribution.
- **Data Imputation**: VAE can be used to fill in missing data by generating plausible values from the latent space distribution.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.base_model import UnsupervisedModel


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