from fbprophet import Prophet
from src.models.base_model import UnsupervisedModel


class ProphetModel(UnsupervisedModel):
    def __init__(self):
        self.model = Prophet()

    def train(self, X):
        X.columns = ["ds", "y"]
        self.model.fit(X)

    def predict(self, X):
        forecast = self.model.predict(X)
        residuals = X["y"] - forecast["yhat"]
        return residuals.abs()