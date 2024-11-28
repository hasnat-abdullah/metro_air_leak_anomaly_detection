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