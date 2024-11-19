from sklearn.ensemble import RandomForestClassifier
from .base_model import SupervisedModel

class RandomForestModel(SupervisedModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  # Probability scores