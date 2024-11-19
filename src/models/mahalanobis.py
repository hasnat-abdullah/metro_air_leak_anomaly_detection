import numpy as np
from scipy.spatial.distance import mahalanobis
from .base_model import UnsupervisedModel

class MahalanobisModel(UnsupervisedModel):
    def __init__(self):
        self.inv_cov_matrix = None
        self.mean_vector = None

    def train(self, X_train):
        self.mean_vector = np.mean(X_train, axis=0)
        cov_matrix = np.cov(X_train, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)

    def predict(self, X_test):
        distances = [mahalanobis(x, self.mean_vector, self.inv_cov_matrix) for x in X_test]
        return np.array(distances)  # Larger distances indicate anomalies