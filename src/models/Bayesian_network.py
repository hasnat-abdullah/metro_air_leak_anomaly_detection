from pomegranate import BayesianNetwork
from src.models.base_model import UnsupervisedModel


class BayesianNetworkModel(UnsupervisedModel):
    def __init__(self):
        self.model = None

    def train(self, X):
        self.model = BayesianNetwork.from_samples(X)

    def predict(self, X):
        return self.model.log_probability(X)