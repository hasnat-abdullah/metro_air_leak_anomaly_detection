"""
Random Forest (RF)

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It is widely used for both classification and regression tasks. Random Forest works by training a collection of decision trees on bootstrapped samples of the data and then aggregating their predictions (via majority voting for classification or averaging for regression) to produce a more robust and accurate model.

Key Concepts:
- **Ensemble Learning**: Random Forest is an ensemble technique, meaning it combines the results of multiple models (decision trees) to improve overall performance.
- **Bootstrapping**: The algorithm creates multiple decision trees by training each tree on a random subset of the data (with replacement). This ensures diversity among the trees.
- **Feature Randomness**: In addition to bootstrapping, Random Forest introduces randomness in the feature selection process. For each split in a tree, only a random subset of features is considered, which helps reduce correlation between trees.
- **Majority Voting**: For classification tasks, Random Forest aggregates the predictions of all trees by majority voting, while for regression tasks, it takes the average of the predictions.

Working:
1. **Tree Construction**: Random Forest builds multiple decision trees by randomly selecting subsets of data points and features, making each tree different and reducing overfitting.
2. **Prediction**: For classification, each tree in the forest predicts a class, and the class with the most votes is selected as the final prediction. For regression, the average prediction of all trees is used.
3. **Out-of-Bag Error Estimation**: Random Forest can use out-of-bag (OOB) data (data not used in training a particular tree) to estimate the generalization error of the model.

Advantages:
- **High Accuracy**: Random Forest generally provides high accuracy and can handle complex datasets without much tuning.
- **Reduced Overfitting**: By averaging multiple decision trees, Random Forest reduces overfitting and improves generalization.
- **Handles Missing Data**: Random Forest can handle missing data by using surrogate splits and can work with both categorical and continuous data.

Disadvantages:
- **Complexity**: Although Random Forest is powerful, it can be computationally expensive, especially with a large number of trees and features.
- **Model Interpretability**: While individual decision trees are interpretable, the ensemble nature of Random Forest makes it less interpretable compared to a single decision tree.

Applications:
- **Classification**: Random Forest is widely used for classification tasks, such as fraud detection, image classification, and medical diagnosis.
- **Regression**: It is also used for regression tasks, such as predicting house prices, stock market trends, and demand forecasting.
- **Feature Importance**: Random Forest can be used to assess the importance of different features in making predictions, which is useful in feature selection and understanding the dataset.
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import SupervisedModel

class RandomForestModel(SupervisedModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  # Probability scores