from sklearn.ensemble import IsolationForest
from .base_model import UnsupervisedModel

class RandomCutForest(UnsupervisedModel):
    def __init__(self, n_estimators=100, max_samples='auto'):
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=42)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return -self.model.decision_function(X_test)  # Anomaly scores