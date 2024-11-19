import xgboost as xgb
from .base_model import SupervisedModel

class XGBoostModel(SupervisedModel):
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        self.model = xgb.XGBClassifier(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  # Probability scores