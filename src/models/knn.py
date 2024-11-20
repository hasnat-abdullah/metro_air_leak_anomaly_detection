"""
K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a simple, instance-based, supervised machine learning algorithm used for classification and regression tasks. The algorithm works by finding the 'K' closest training examples to a new data point, and then predicting the label (or value) based on the majority class (for classification) or average (for regression) of these neighbors.

Key Concepts:
- Instance-Based Learning: KNN is an instance-based learning algorithm, meaning that it does not learn an explicit model. Instead, it stores the training dataset and makes predictions based on the similarity to the training points during testing.
- Distance Metric: The algorithm relies on a distance metric to determine the "closeness" between data points. The most commonly used metric is **Euclidean distance**, but other metrics such as **Manhattan**, **Cosine similarity**, or **Minkowski distance** can also be used.
- K (the number of neighbors): The key parameter for KNN is 'K', which defines how many nearest neighbors to consider when making a prediction. A small value of K makes the algorithm sensitive to noise, while a large value may cause it to underfit by smoothing over important distinctions between classes.

Classification vs. Regression:
- **Classification**: In classification tasks, KNN assigns a new data point to the majority class of its 'K' nearest neighbors. For example, in a binary classification problem, if the majority of the nearest neighbors belong to class 0, the new data point will be classified as 0.
- **Regression**: In regression tasks, KNN predicts the target value based on the average of the target values of its nearest neighbors. For example, in a regression task predicting house prices, the predicted price will be the average price of the K nearest houses.

Algorithm:
1. Compute the distance between the test point and all training points in the dataset.
2. Sort the training data by distance, selecting the 'K' nearest neighbors.
3. For classification, take a majority vote of the 'K' neighbors' labels and assign the most frequent label to the test point.
4. For regression, calculate the mean (or weighted mean) of the target values of the K neighbors and assign that as the predicted value.

Hyperparameters:
- **K (Number of Neighbors)**: The number of nearest neighbors to consider when making predictions. It is a critical hyperparameter that should be chosen carefully using techniques like cross-validation to prevent underfitting or overfitting.
- **Distance Metric**: The metric used to measure the "closeness" of data points. Common choices include Euclidean, Manhattan, and Minkowski distances.
- **Weights**: KNN can assign equal weight to all neighbors or give more weight to closer neighbors. The "distance-based" weighting assigns higher weights to nearer neighbors.
- **Algorithm**: The algorithm used to compute the nearest neighbors, options include 'auto', 'ball_tree', 'kd_tree', or 'brute'.

Applications:
- **Classification**: Used in pattern recognition tasks such as spam detection, image classification, and medical diagnosis.
- **Regression**: Used for predicting continuous values such as stock prices, housing prices, and sales forecasting.
- **Anomaly Detection**: KNN can also be used to detect anomalies by identifying points that are distant from all others in the training set.

Advantages:
- Simple and easy to understand and implement.
- No training phase is required, making it computationally inexpensive in terms of time complexity during the training phase.
- Flexible: Can be used for both classification and regression tasks.
- Works well with smaller datasets and can handle multi-class classification problems.

Disadvantages:
- **Computationally Expensive**: The prediction phase can be slow, especially with large datasets, as the algorithm needs to compute distances to all training points for each prediction.
- **Memory Intensive**: KNN requires storing the entire training dataset in memory, which can be a problem with very large datasets.
- **Sensitive to Irrelevant Features**: KNN performance can degrade if the data has many irrelevant features or is high-dimensional (curse of dimensionality).
- **Sensitive to Scaling**: KNN is sensitive to the scale of the data, as distances are calculated based on feature values. Feature scaling (e.g., using normalization or standardization) is often necessary.

Optimization and Variants:
- **Weighted KNN**: A variant where closer neighbors are given more weight when making predictions.
- **KD-Trees and Ball-Trees**: Data structures that speed up the process of finding nearest neighbors, especially in high-dimensional spaces.
- **KNN with Dimensionality Reduction**: KNN can be enhanced with techniques like PCA or t-SNE for dimensionality reduction, which can improve performance in high-dimensional data.

In practice, KNN is widely used in a variety of applications, especially when the relationships between data points are complex and not easily captured by simpler linear models.
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
from .base_model import UnsupervisedModel

class KNNModel(UnsupervisedModel):
    def __init__(self, n_neighbors=5):
        self.model = NearestNeighbors(n_neighbors=n_neighbors)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        # Get distances of nearest neighbors for each point in X_test
        distances, _ = self.model.kneighbors(X_test)
        # Return mean distance to neighbors as the anomaly score
        return np.mean(distances, axis=1)