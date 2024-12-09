"""
TimeGAN:
- A Generative Adversarial Network (GAN) designed for synthetic time series generation.
- Consists of generator and discriminator networks with an embedded encoder-decoder structure.
- Captures temporal dependencies and complex patterns in sequential data.
- Can also be adapted for anomaly detection by evaluating the likelihood of generated samples.

Key Concepts:
- GAN Framework: Uses adversarial learning with a generator and discriminator.
- Temporal Dependencies: Models sequential patterns in data.
- Embedding Network: Learns latent representations of time series data for better generation.

Advantages:
- Generates realistic time series data.
- Captures both temporal and feature-level dependencies.
- Useful for anomaly detection and data augmentation.

Applications:
- Synthetic data generation for imbalanced datasets.
- Anomaly detection in sequential data.
- Financial time series modeling and forecasting.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

class TimeGANModel:
    def __init__(self, latent_dim=64, n_epochs=100, batch_size=32, seq_length=10):
        """Initialize the TimeGAN model."""
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.sequence_autoencoder = self.build_sequence_autoencoder()

        # Compile the discriminator
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Compile the combined model (generator + discriminator)
        self.combined = self.build_combined_model()
        self.combined.compile(optimizer='adam', loss='binary_crossentropy')  # Ensure combined model is compiled

    def build_generator(self):
        """Build the generator network."""
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.seq_length, self.latent_dim)))  # Input layer
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(128, return_sequences=False))
        model.add(layers.Dense(self.seq_length, activation='linear'))  # Output shape for time series
        return model

    def build_discriminator(self):
        """Build the discriminator network."""
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.seq_length, 1)))  # Input layer
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(128, return_sequences=False))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification: Real or Fake
        return model

    def build_combined_model(self):
        """Build the combined model (generator + discriminator)."""
        self.discriminator.trainable = False  # We only train the generator in the combined model
        model = tf.keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def build_sequence_autoencoder(self):
        """Build the sequence autoencoder for training purposes."""
        model = tf.keras.Sequential()
        model.add(layers.LSTM(128, return_sequences=True, input_shape=(self.seq_length, 1)))
        model.add(layers.LSTM(128, return_sequences=False))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.RepeatVector(self.seq_length))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.Dense(1, activation='linear'))
        return model

    def train(self, X_train):
        """Train the TimeGAN model."""
        # Preprocess input data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Trim data to make sure it's divisible by seq_length
        n_samples = len(X_train_scaled) // self.seq_length * self.seq_length
        X_train_scaled = X_train_scaled[:n_samples]

        # Reshape after trimming
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0] // self.seq_length, self.seq_length, 1))

        # Training loop
        for epoch in range(self.n_epochs):
            for i in range(0, len(X_train_scaled), self.batch_size):
                batch_data = X_train_scaled[i:i + self.batch_size]

                # Generate fake data
                noise = np.random.normal(0, 1, (batch_data.shape[0], self.seq_length, self.latent_dim))
                generated_data = self.generator.predict(noise)

                # Train the discriminator (real vs fake)
                real_labels = np.ones((batch_data.shape[0], 1))
                fake_labels = np.zeros((batch_data.shape[0], 1))
                d_loss_real = self.discriminator.train_on_batch(batch_data, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

                # Train the generator via the discriminator's feedback
                noise = np.random.normal(0, 1, (batch_data.shape[0], self.seq_length, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, real_labels)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.n_epochs} - Discriminator Loss: {d_loss_real + d_loss_fake}, Generator Loss: {g_loss}")

    def generate(self, n_samples):
        """Generate synthetic time series data."""
        noise = np.random.normal(0, 1, (n_samples, self.seq_length, self.latent_dim))
        generated_data = self.generator.predict(noise)
        return generated_data

    def evaluate(self, X_test):
        """Evaluate the TimeGAN model."""
        X_test_scaled = MinMaxScaler().fit_transform(X_test)
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], self.seq_length, 1))

        # Calculate loss and accuracy (using reconstruction error as a simple measure)
        reconstructions = self.sequence_autoencoder.predict(X_test_scaled)
        reconstruction_error = np.mean(np.abs(X_test_scaled - reconstructions), axis=1)
        return np.mean(reconstruction_error)

    def visualize_generated_data(self, generated_data):
        """Visualize generated time series data."""
        plt.figure(figsize=(10, 6))
        plt.plot(generated_data[0], label="Generated Sample")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title("Generated Time Series Data")
        plt.legend()
        plt.show()

    def run_pipeline(self, df: pd.DataFrame):
        """Run the full TimeGAN pipeline."""
        # Preprocessing: Scale and reshape the data for training
        X = df[['Oxygen']].values  # Use only the 'Oxygen' feature
        self.train(X)

        # Generate synthetic data
        generated_data = self.generate(len(df))

        # Evaluate the model
        reconstruction_error = self.evaluate(X)
        print(f"Reconstruction Error: {reconstruction_error}")

        # Visualize generated data
        self.visualize_generated_data(generated_data)

        return generated_data, reconstruction_error


if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data("10T")  # Assume this is your time series data
    timegan_model = TimeGANModel(latent_dim=64, n_epochs=100, batch_size=32, seq_length=10)
    generated_data, reconstruction_error = timegan_model.run_pipeline(data)

    # Output results
    print(f"Generated Data Sample: {generated_data[0]}")
    print(f"Reconstruction Error: {reconstruction_error}")