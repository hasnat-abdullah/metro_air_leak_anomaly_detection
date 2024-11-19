from sklearn.cluster import DBSCAN
from .base_model import UnsupervisedModel

class DBSCANModel(UnsupervisedModel):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def train(self, X):
        # DBSCAN is an unsupervised learning model that doesn't require training labels.
        self.model.fit(X)

    def predict(self, X):
        # DBSCAN does not provide a typical classification output, it gives clusters or noise (-1)
        return self.model.fit_predict(X)