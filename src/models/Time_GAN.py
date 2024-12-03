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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.base_model import UnsupervisedModel


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class TimeGAN(UnsupervisedModel):
    def __init__(self, time_steps=24, hidden_dim=128, epochs=1000):
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.epochs = epochs

        # Generator and Discriminator Initialization
        self.generator = Generator(input_dim=self.time_steps, hidden_dim=self.hidden_dim, output_dim=self.time_steps)
        self.discriminator = Discriminator(input_dim=self.time_steps, hidden_dim=self.hidden_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss function
        self.criterion = nn.BCELoss()

    def train(self, X_train):
        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)

        for epoch in range(self.epochs):
            # Train Discriminator
            self.discriminator.zero_grad()
            real_data = X_train
            fake_data = self.generator(torch.randn(X_train.shape[0], self.time_steps).to(self.device))

            # Ensure correct shape for real and fake data
            real_data = real_data.view(-1, self.time_steps)  # Reshape if necessary
            fake_data = fake_data.view(-1, self.time_steps)  # Reshape if necessary

            # Real data labels (1s for real data)
            real_labels = torch.ones(X_train.shape[0], 1).to(self.device)
            fake_labels = torch.zeros(X_train.shape[0], 1).to(self.device)

            real_output = self.discriminator(real_data)
            real_loss = self.criterion(real_output, real_labels)

            fake_output = self.discriminator(fake_data.detach())
            fake_loss = self.criterion(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_d.step()

            # Train Generator
            self.generator.zero_grad()
            fake_output = self.discriminator(fake_data)
            g_loss = self.criterion(fake_output, real_labels)  # Generator tries to fool the discriminator

            g_loss.backward()
            self.optimizer_g.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{self.epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    def predict(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        generated_data = self.generator(torch.randn(X_test.shape[0], self.time_steps).to(self.device))
        return generated_data.detach().cpu().numpy()