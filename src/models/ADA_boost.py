from sklearn.ensemble import AdaBoostClassifier
from src.models.base_model import SupervisedModel


class AdaBoostModel(SupervisedModel):
    def __init__(self, n_estimators=50):
        self.model = AdaBoostClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
