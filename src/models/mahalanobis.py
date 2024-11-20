"""
Mahalanobis Distance

Mahalanobis Distance is a measure of the distance between a point and a distribution. Unlike the Euclidean distance, which measures the straight-line distance between two points, the Mahalanobis distance accounts for correlations between variables and the overall distribution of the data. This makes it particularly useful for identifying outliers in multivariate datasets, as it is more sensitive to the covariance structure of the data.

Key Concepts:
- **Covariance Matrix**: Mahalanobis distance takes into account the covariance of the dataset, making it scale-invariant and more effective for datasets where variables are correlated.
- **Distance Metric**: The Mahalanobis distance between a point `x` and a mean `μ` of a multivariate distribution with covariance matrix `Σ` is calculated as:

  D(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))

  Where `x` is the vector of the data point, `μ` is the mean vector, and `Σ⁻¹` is the inverse of the covariance matrix.

- **Anomaly Detection**: Mahalanobis distance can be used to detect outliers by computing the distance between data points and the center of a distribution. Points with a large Mahalanobis distance are considered anomalies.

Advantages:
- **Multivariate**: Unlike Euclidean distance, Mahalanobis distance can be applied to datasets with multiple variables, making it more suitable for anomaly detection in multivariate settings.
- **Scale-Invariance**: The distance metric accounts for correlations between features, meaning it is not biased by different scales of the data (e.g., some features being much larger in magnitude than others).
- **Effective in Normalized Data**: Mahalanobis distance works well when the data is centered (mean is zero) and covariance is accounted for.

Disadvantages:
- **Requires Covariance Matrix**: Calculating the Mahalanobis distance requires computing the covariance matrix, which can be computationally expensive for high-dimensional data.
- **Sensitive to the Covariance Matrix**: The performance of Mahalanobis distance is highly sensitive to the accuracy of the covariance matrix. For small datasets or datasets with highly correlated features, the covariance matrix might be ill-conditioned.

Applications:
- **Outlier Detection**: Mahalanobis distance is commonly used for detecting multivariate outliers or anomalies, especially in data with correlated variables.
- **Multivariate Statistical Process Control**: It is used in process control and quality management, where the Mahalanobis distance is applied to monitor the quality of processes based on multivariate data.
- **Face Recognition**: Mahalanobis distance is also applied in image recognition, where it is used to measure the distance between a test image and a reference image in the feature space.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from .base_model import UnsupervisedModel

class MahalanobisModel(UnsupervisedModel):
    def __init__(self):
        self.inv_cov_matrix = None
        self.mean_vector = None

    def train(self, X_train):
        self.mean_vector = np.mean(X_train, axis=0)
        cov_matrix = np.cov(X_train, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)

    def predict(self, X_test):
        distances = [mahalanobis(x, self.mean_vector, self.inv_cov_matrix) for x in X_test]
        return np.array(distances)  # Larger distances indicate anomalies