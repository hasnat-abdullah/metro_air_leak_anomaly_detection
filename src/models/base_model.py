from abc import ABC, abstractmethod
from typing import Optional

from src.config import ModelType


class BaseModel(ABC):
    """Base class for all models."""
    ModelType: ModelType

    @abstractmethod
    def train(self, X_train, y_train=None):
        """Train the model. For supervised models, provide y_train."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions on the test data."""
        pass


class UnsupervisedModel(BaseModel):
    """Base class for unsupervised models."""
    MODEL_TYPE= ModelType.UNSUPERVISED

    @abstractmethod
    def train(self, X_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class SupervisedModel(BaseModel):
    """Base class for supervised models."""
    MODEL_TYPE= ModelType.SUPERVISED

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass