from dtaidistance import dtw
from src.models.base_model import UnsupervisedModel


class DTWModel(UnsupervisedModel):
    def __init__(self, reference_series):
        self.reference_series = reference_series

    def train(self, X):
        # DTW doesn't require explicit training
        pass

    def predict(self, X):
        distances = [dtw.distance(series, self.reference_series) for series in X.T]
        return distances