from abc import ABC, abstractmethod

class UnsupervisedModel(ABC):
    """Base class for unsupervised models."""
    @abstractmethod
    def train(self, X_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class SupervisedModel(ABC):
    """Base class for supervised models."""
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass