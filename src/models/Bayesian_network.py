"""
Bayesian Network:
- A probabilistic graphical model representing variables and their conditional dependencies using a directed acyclic graph (DAG).
- Captures the joint probability distribution of variables to reason under uncertainty.

Key Concepts:
- Conditional Independence: Encodes relationships between variables using probabilities.
- Directed Acyclic Graph (DAG): Represents dependencies among variables.
- Inference: Uses Bayes' theorem to calculate probabilities of unknown variables.

Advantages:
- Handles uncertainty and incomplete data effectively.
- Scalable for high-dimensional datasets.
- Interpretable structure for domain experts.

Applications:
- Fault diagnosis in industrial systems.
- Risk assessment in finance and insurance.
- Medical decision support systems.
"""

from pomegranate.bayesian_network import BayesianNetwork
from src.models.base_model import UnsupervisedModel


class BayesianNetworkModel(UnsupervisedModel):
    def __init__(self):
        self.model = None

    def train(self, X):
        self.model = BayesianNetwork.from_samples(X)

    def predict(self, X):
        return self.model.log_probability(X)