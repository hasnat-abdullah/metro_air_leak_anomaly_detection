"""Robust Principal Component Analysis"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from .base_model import UnsupervisedModel


class RPCA(UnsupervisedModel):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = RobustScaler()

    def train(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)