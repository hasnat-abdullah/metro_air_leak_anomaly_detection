"""
Gaussian Process Regression (GPR):
- A non-parametric, Bayesian approach to regression tasks.
- Defines a distribution over functions, providing predictions with confidence intervals.

Key Concepts:
- Gaussian Process: A collection of random variables, any subset of which is jointly Gaussian.
- Kernel Function: Defines the covariance structure of the data, capturing smoothness or periodicity.
- Bayesian Framework: Incorporates prior knowledge and provides uncertainty estimates.

Advantages:
- Provides uncertainty bounds for predictions.
- Flexible and adaptable to various datasets.
- Handles small datasets well due to its non-parametric nature.

Applications:
- Anomaly detection in engineering systems.
- Time series forecasting with confidence intervals.
- Optimization of expensive-to-evaluate functions.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from src.models.base_model import SupervisedModel


class GaussianProcessModel(SupervisedModel):
    def __init__(self):
        self.model = GaussianProcessRegressor(kernel=RBF())

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test, return_std=True)