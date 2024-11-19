from pyod.models.loda import LODA
from .base_model import UnsupervisedModel

class LODAModel(UnsupervisedModel):
    def __init__(self, n_bins=10):
        self.model = LODA(n_bins=n_bins)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.decision_function(X_test)  # Anomaly scores