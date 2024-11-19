from sklearn.cluster import KMeans

from src.models.base_model import UnsupervisedModel


class KMeansModel(UnsupervisedModel):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)