from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from .base_model import UnsupervisedModel

class AutoEncoder(UnsupervisedModel):
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])
        # Use 'tensorflow.keras.optimizers.Adam' for compatibility
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def train(self, X_train):
        self.model.fit(X_train, X_train, epochs=50, batch_size=128, validation_split=0.1, verbose=0)

    def predict(self, X_test):
        reconstructions = self.model.predict(X_test)
        return np.mean((X_test - reconstructions) ** 2, axis=1)  # Reconstruction error
