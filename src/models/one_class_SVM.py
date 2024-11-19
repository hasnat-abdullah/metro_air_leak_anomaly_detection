from sklearn.svm import OneClassSVM
from .base_model import UnsupervisedModel

class OneClassSVMModel(UnsupervisedModel):
    def __init__(self, kernel="rbf", nu=0.05):
        self.model = OneClassSVM(kernel=kernel, nu=nu)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return -self.model.decision_function(X_test)  # Scores