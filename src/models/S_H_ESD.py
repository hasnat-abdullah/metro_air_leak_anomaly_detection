from pyculiarity import detect_ts
from src.models.base_model import UnsupervisedModel


class SHESDModel(UnsupervisedModel):
    def __init__(self, alpha=0.05, max_anoms=0.05):
        self.alpha = alpha
        self.max_anoms = max_anoms

    def train(self, X):
        # S-H-ESD is stateless; no training is required.
        pass

    def predict(self, X):
        results = []
        for col in X.columns:
            anomalies = detect_ts(X[col], alpha=self.alpha, max_anoms=self.max_anoms)
            results.append(anomalies["anoms"].index)
        return results