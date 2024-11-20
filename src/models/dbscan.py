"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is an unsupervised clustering algorithm that groups together closely packed points, marking points that lie alone in low-density regions as outliers. Unlike K-means, DBSCAN does not require specifying the number of clusters in advance, and it can identify arbitrarily shaped clusters.

Key Concepts:
- Core Points: Points that have at least a specified number of points (MinPts) within a given distance (epsilon).
- Border Points: Points that are within the epsilon distance from a core point but do not have enough neighbors to be core points themselves.
- Noise Points: Points that are neither core points nor border points.

In the context of anomaly detection, DBSCAN can be used to identify anomalous or outlier points as those that do not belong to any cluster. These points are considered noise points and can be flagged as anomalies.

Key Features:
- Unsupervised learning: Does not require labeled data.
- Outlier detection: Points that do not belong to any cluster are flagged as anomalies.
- Arbitrary-shaped clusters: DBSCAN can find clusters of arbitrary shape and is not restricted to circular or spherical clusters.
- Robust to noise: DBSCAN can identify and ignore noise points that do not fit into any clusters.

Parameters:
- Epsilon (ε): The maximum distance between two points to be considered as neighbors.
- MinPts: The minimum number of points required to form a dense region (a cluster).

Applications:
- Spatial data analysis
- Image segmentation
- Anomaly detection in various domains such as fraud detection, network intrusion, and sensor data monitoring
"""

from sklearn.cluster import DBSCAN
from .base_model import UnsupervisedModel

class DBSCANModel(UnsupervisedModel):
    def __init__(self, eps=0.3, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        # DBSCAN does not provide a typical classification output, it gives clusters or noise (-1)
        return self.model.fit_predict(X)