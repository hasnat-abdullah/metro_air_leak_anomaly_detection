"""
AdaBoost (Adaptive Boosting):
- A boosting algorithm that combines multiple weak classifiers to create a strong classifier.
- Emphasizes samples that are misclassified by previous classifiers by assigning higher weights.
- Iteratively improves model performance by focusing on errors.

Key Concepts:
- Boosting: Combines weak learners sequentially, where each learner corrects the errors of its predecessor.
- Weight Adjustment: Assigns higher weights to misclassified samples to improve their prediction.
- Ensemble Learning: Combines predictions from multiple models for better accuracy.

Advantages:
- Handles both classification and regression tasks effectively.
- Robust to overfitting when weak learners are constrained.
- Simple and interpretable algorithm.

Applications:
- Fraud detection in financial systems.
- Spam detection in email filtering.
- Customer churn prediction in marketing.
"""

from sklearn.ensemble import AdaBoostClassifier
from src.models.base_model import SupervisedModel


class AdaBoostModel(SupervisedModel):
    def __init__(self, n_estimators=50):
        self.model = AdaBoostClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
