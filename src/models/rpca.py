"""
Robust Principal Component Analysis (RPCA)

RPCA is an extension of Principal Component Analysis (PCA) designed to deal with noisy and corrupted data. RPCA decomposes a matrix into two components: a low-rank matrix representing the underlying structure of the data, and a sparse matrix capturing the outliers or anomalies. This decomposition is particularly useful for identifying anomalous data in scenarios where the data is affected by noise or outliers.

Key Concepts:
- **Low-Rank Matrix**: The low-rank component captures the underlying structure of the data, which is assumed to be smooth or continuous.
- **Sparse Matrix**: The sparse component captures the outliers or anomalies that deviate significantly from the normal data.
- **Optimization Problem**: RPCA formulates the decomposition as an optimization problem, where the goal is to find a low-rank matrix and a sparse matrix that together approximate the original data matrix.

Working:
1. **Matrix Decomposition**: RPCA seeks to decompose a given data matrix `X` into a sum of a low-rank matrix `L` and a sparse matrix `S`, i.e., X = L + S, where `L` is the low-rank part, and `S` contains the anomalies.
2. **Optimization**: The decomposition is done through an optimization algorithm (such as alternating minimization) that minimizes the rank of `L` and the sparsity of `S` simultaneously.
3. **Anomaly Detection**: The sparse matrix `S` identifies the outliers or anomalies, which are the data points that are significantly different from the low-rank structure of the data.

Advantages:
- **Handles Outliers**: RPCA is robust to noise and outliers, making it effective for anomaly detection in noisy datasets.
- **Dimensionality Reduction**: RPCA can be used for dimensionality reduction, similar to PCA, while handling noisy or corrupted data.
- **Effective in Complex Datasets**: RPCA is effective in scenarios where the data is not purely low-rank, but contains a significant amount of noise or outliers.

Disadvantages:
- **Computationally Expensive**: RPCA can be computationally expensive, especially for large datasets, due to the complexity of the optimization problem.
- **Sensitive to Parameters**: RPCA requires careful tuning of regularization parameters to balance the rank of `L` and sparsity of `S`.

Applications:
- **Anomaly Detection**: RPCA is used for anomaly detection in applications like fraud detection, sensor monitoring, and intrusion detection.
- **Image Denoising**: RPCA is applied in image processing to remove noise while preserving the underlying structure of the image.
- **Data Recovery**: RPCA can be used to recover missing or corrupted data by separating out the outliers and reconstructing the low-rank structure of the data.
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from .base_model import UnsupervisedModel


class RPCA(UnsupervisedModel):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = RobustScaler()

    def train(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)