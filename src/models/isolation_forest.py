from sklearn.ensemble import IsolationForest
from .base_model import UnsupervisedModel

class IsolationForestModel(UnsupervisedModel):
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return -self.model.decision_function(X_test)  # Scores