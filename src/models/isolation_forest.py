"""
Isolation Forest

Isolation Forest (iForest) is an unsupervised machine learning algorithm primarily used for anomaly detection. It isolates anomalies instead of profiling normal data points, making it highly efficient for large datasets with high-dimensional features.

Key Concepts:
- Anomaly Isolation: The core idea behind Isolation Forest is that anomalies are few and different, meaning they are easier to isolate from the rest of the data. The algorithm randomly selects a feature and splits the data at a random value, recursively creating isolation trees.
- Isolation Trees: The algorithm builds an ensemble of decision trees (called isolation trees). Each tree isolates data points by randomly selecting a feature and a random split value, and the process continues recursively until data points are isolated.
- Path Length: The number of splits required to isolate a data point is used as a measure of its "anomaly score." Points that are isolated with fewer splits are considered anomalies because they are different from the majority of the data.
- Anomaly Scoring: A score is computed based on the average path length of a data point in the forest of trees. Data points with shorter path lengths (requiring fewer splits) are deemed anomalies, while those with longer path lengths are considered normal.

In the context of anomaly detection:
- Isolation Forest is particularly well-suited for detecting anomalies in high-dimensional datasets.
- It works by isolating outliers quickly, making it highly efficient in terms of computational cost, especially for large datasets.
- The algorithm does not require labeled data and operates purely on the characteristics of the data, making it an effective unsupervised anomaly detection technique.

Key Features:
- Unsupervised learning: The algorithm detects anomalies without needing labeled data, making it ideal for real-world applications where labeled data may be sparse or unavailable.
- Efficient and scalable: The algorithm is highly efficient, especially for large datasets, as it only requires a small number of trees to achieve good performance.
- Easy to interpret: The anomaly score provides a straightforward interpretation, with higher scores indicating higher likelihood of being an anomaly.
- Non-parametric: No assumptions are made about the underlying data distribution, making it flexible across a wide variety of applications.

Parameters:
- Number of trees: The number of isolation trees (estimators) in the forest. More trees usually lead to better performance but increased computational cost.
- Subsampling size: The number of data points to sample from the training set for each tree. A smaller sample size can increase the diversity of the trees and help identify anomalies better.
- Contamination: The proportion of outliers in the dataset, used to adjust the decision threshold for anomaly detection.
- Maximum sample size: The maximum number of data points that can be used to build each tree.

Applications:
- Anomaly detection: Detecting outliers or anomalous data points in datasets, such as fraudulent transactions, rare disease detection, and defect detection in manufacturing.
- Data cleaning: Identifying and removing noisy or irrelevant data points from the dataset before further analysis or model building.
- Intrusion detection: Identifying abnormal patterns of behavior in network traffic or system logs.
- Outlier detection in high-dimensional spaces: Effective in scenarios where traditional methods like distance-based methods struggle with high-dimensional data.

Limitations:
- Assumes that anomalies are "few and different," so it may not perform well when anomalies are similar to normal data points or when the dataset contains imbalanced data.
- It can be sensitive to the number of trees and the subsampling size; improper tuning of these hyperparameters can affect model performance.
- The algorithm may not perform well in cases where the data has complex dependencies between features, as it isolates based on random splits.

Benefits:
- Highly efficient for large datasets.
- No need for labeled data (unsupervised).
- Works well even with high-dimensional data.

Isolation Forest has become a popular choice for anomaly detection due to its efficiency and robustness in handling high-dimensional data with a relatively low computational cost.
"""

from sklearn.ensemble import IsolationForest
from .base_model import UnsupervisedModel

class IsolationForestModel(UnsupervisedModel):
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return -self.model.decision_function(X_test)  # Scores