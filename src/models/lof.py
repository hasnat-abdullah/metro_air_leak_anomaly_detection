from sklearn.neighbors import LocalOutlierFactor
from .base_model import UnsupervisedModel

class LOFModel(UnsupervisedModel):
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors)

    def train(self, X):
        # LOF is an unsupervised model, no need to pass labels
        self.model.fit(X)

    def predict(self, X):
        # LOF returns -1 for outliers and 1 for inliers
        return self.model.fit_predict(X)