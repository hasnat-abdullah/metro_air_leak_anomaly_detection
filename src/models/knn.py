from sklearn.neighbors import NearestNeighbors
import numpy as np
from .base_model import UnsupervisedModel

class KNNModel(UnsupervisedModel):
    def __init__(self, n_neighbors=5):
        self.model = NearestNeighbors(n_neighbors=n_neighbors)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        # Get distances of nearest neighbors for each point in X_test
        distances, _ = self.model.kneighbors(X_test)
        # Return mean distance to neighbors as the anomaly score
        return np.mean(distances, axis=1)