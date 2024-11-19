"""Gaussian Mixture Model"""

from sklearn.mixture import GaussianMixture

from src.models.base_model import UnsupervisedModel


class GMMModel(UnsupervisedModel):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=self.n_components)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)