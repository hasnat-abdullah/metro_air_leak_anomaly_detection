from matrixprofile import compute
from src.models.base_model import UnsupervisedModel


class MatrixProfileModel(UnsupervisedModel):
    def __init__(self, window_size):
        self.window_size = window_size

    def train(self, X):
        pass

    def predict(self, X):
        profiles = compute(X, self.window_size)
        return profiles["mp"]