import xgboost as xgb
from .base_model import SupervisedModel

"""
XGBoost (Extreme Gradient Boosting)

XGBoost is a highly efficient and scalable implementation of gradient boosting, a machine learning algorithm for supervised learning tasks such as classification and regression. It builds an ensemble of decision trees by training them sequentially, with each tree correcting the errors made by the previous one. XGBoost is known for its performance and is widely used in machine learning competitions and real-world applications.

Key Concepts:
- **Gradient Boosting**: XGBoost is based on the gradient boosting framework, where a series of weak learners (typically decision trees) are trained sequentially. Each tree is trained to minimize the residual errors of the previous trees in the sequence, and the predictions are combined to form the final output.
- **Boosting**: In boosting, the model gives more weight to the errors made by the previous model, which helps in reducing bias and improving model accuracy. Each tree in the sequence corrects the errors of the combined ensemble of trees.
- **Regularization**: XGBoost introduces a regularization term in the objective function, which helps prevent overfitting and improves model generalization. This is one of the key reasons for its superior performance compared to other gradient boosting implementations.
- **Tree Pruning**: XGBoost uses a novel tree pruning method that prevents overfitting by stopping the growth of trees early (using a depth-first approach and pruning branches with insufficient gain).
- **Parallelization**: XGBoost is highly optimized and supports parallel processing, making it much faster compared to other gradient boosting methods.

Working:
1. **Sequential Tree Building**: XGBoost builds an ensemble of decision trees sequentially, where each new tree is fitted to correct the errors (residuals) of the previous trees.
2. **Loss Function**: The algorithm uses a differentiable loss function that measures how well the model's predictions match the actual target values. The optimization process aims to minimize this loss.
3. **Gradient Descent**: The training process involves using gradient descent to optimize the loss function. Each new tree is added to minimize the gradient of the loss with respect to the current ensemble of trees.
4. **Regularization**: XGBoost applies L1 (lasso) and L2 (ridge) regularization to the weights of the trees, which helps reduce overfitting and improves the generalization of the model.
5. **Prediction**: The final prediction is made by aggregating the predictions from all the individual trees, either through weighted averaging for regression or majority voting for classification.

Advantages:
- **High Performance**: XGBoost is known for its speed and efficiency, making it suitable for large-scale datasets.
- **Regularization**: The regularization terms help prevent overfitting, improving the generalization of the model.
- **Flexibility**: XGBoost supports various types of loss functions and can be applied to a wide range of problems, including classification, regression, and ranking tasks.
- **Feature Importance**: XGBoost provides built-in methods for calculating feature importance, helping in feature selection and model interpretation.

Disadvantages:
- **Complexity**: XGBoost can be complex to tune due to the large number of hyperparameters, including tree depth, learning rate, and regularization parameters.
- **Memory Usage**: XGBoost can be memory-intensive, especially when working with large datasets, due to its tree-building and data storage requirements.

Applications:
- **Classification**: XGBoost is widely used for binary and multi-class classification tasks, such as fraud detection, spam detection, and medical diagnosis.
- **Regression**: XGBoost is also used for regression tasks, such as predicting house prices, stock market forecasting, and sales prediction.
- **Ranking and Recommendation**: XGBoost has been applied in ranking problems, such as in search engine ranking and recommendation systems.
- **Feature Selection**: XGBoost can be used to evaluate feature importance and perform feature selection in high-dimensional datasets.
"""

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