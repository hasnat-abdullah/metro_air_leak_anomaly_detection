"""
Local Outlier Factor (LOF)

The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection technique that identifies outliers in data by comparing the local density of a data point with the local densities of its neighbors. It is particularly effective in detecting local outliers that may not be detected by global anomaly detection methods, making it useful for identifying anomalies in datasets with varying densities or clusters.

Key Concepts:
- **Local Density**: The LOF algorithm computes the local density of a data point based on its distance to its neighbors. The density is defined by how close or far the data points are from each other in the feature space.
- **Reachability Distance**: LOF calculates the reachability distance between points, which takes into account the density of the neighbors. This is used to compute the local density of a point.
- **Anomaly Score**: A point's LOF score measures how much its local density deviates from that of its neighbors. If a point has a significantly lower density compared to its neighbors, it is considered an outlier.

Algorithm:
1. **K-Nearest Neighbors (K-NN)**: The algorithm starts by determining the K-nearest neighbors for each data point. The parameter `k` specifies how many neighbors to consider.
2. **Reachability Distance Calculation**: For each data point, the algorithm calculates the reachability distance to each of its K-nearest neighbors. The reachability distance is defined as the maximum of the distance between the points and the k-distance of the neighbor.
3. **Local Reachability Density (LRD)**: The LRD of each data point is computed based on the reachability distances. Points that are close to many other points (i.e., have a high local density) have a higher LRD, while points with fewer neighbors or those located in sparse regions of the data have a lower LRD.
4. **LOF Calculation**: The LOF score is calculated as the ratio of the average local reachability density of a point’s neighbors to the local reachability density of the point itself. If a point has an LOF score significantly greater than 1, it is considered an outlier.

Hyperparameters:
- **Number of Neighbors (k)**: The number of nearest neighbors used to calculate the local density. A larger value of `k` may make the algorithm less sensitive to local outliers, while a smaller `k` may increase sensitivity to noise.
- **Anomaly Threshold**: The threshold used to classify data points as anomalies. Points with LOF scores greater than a certain value (e.g., 1.5 or 2) are considered anomalies.

Advantages:
- **Local Anomaly Detection**: LOF excels in detecting local outliers that may be missed by global anomaly detection algorithms. It is useful when the dataset has regions with varying densities or non-uniform distribution of points.
- **Unsupervised**: LOF is an unsupervised learning algorithm, meaning it does not require labeled data for training, making it suitable for situations where only the data is available and no labels are provided.
- **Flexibility**: LOF can be applied to a wide variety of data types and can be combined with other techniques such as clustering or dimensionality reduction to improve performance in complex datasets.

Disadvantages:
- **Sensitivity to `k`**: The performance of LOF depends on the choice of the number of neighbors (`k`). A poor choice of `k` can affect the sensitivity of the algorithm and result in false positives or missed anomalies.
- **Computational Complexity**: LOF requires computing distances between all pairs of data points, which can become computationally expensive for large datasets, especially when the dimensionality is high.
- **Difficulty with High-Dimensional Data**: Like many distance-based methods, LOF may suffer from the "curse of dimensionality" in high-dimensional datasets, where the notion of distance becomes less meaningful.

Applications:
- **Anomaly Detection in Datasets with Varying Densities**: LOF is particularly useful when the data has clusters or regions of varying density, as it can detect anomalies in less dense regions that might be overlooked by global anomaly detection methods.
- **Fraud Detection**: LOF is used in fraud detection systems to identify transactions that deviate from normal patterns of behavior, especially when the normal behavior is highly variable across different users or transaction types.
- **Network Intrusion Detection**: LOF can be applied to detect unusual activity in network traffic, such as intrusion attempts or abnormal behavior patterns that differ from typical traffic.
- **Sensor Data Monitoring**: LOF can help in detecting anomalous behavior in sensor data, such as irregular readings from industrial sensors, which may indicate equipment failures or malfunctions.
- **Image and Video Anomaly Detection**: LOF can be used in computer vision applications to detect anomalies in image or video data, such as identifying abnormal objects or actions.

Optimization and Variants:
- **LOF with Local Clustering**: A variant of LOF involves using clustering techniques (e.g., DBSCAN or K-means) to identify regions of high density in the data, which can improve the sensitivity of the algorithm to anomalies.
- **Weighted LOF**: In some cases, LOF can be extended to use weighted distances or weighted reachability distances to give more importance to certain features or points during the anomaly detection process.
- **FastLOF**: FastLOF is a more efficient version of the LOF algorithm designed to speed up the computation of the LOF scores by using approximate methods or optimizations to reduce the time complexity.

In practice, LOF is useful for anomaly detection in scenarios where the dataset exhibits varying densities or where the anomalies are localized in specific regions of the data. Its ability to detect outliers based on local density makes it a powerful tool for identifying anomalous behavior in complex and dynamic environments.
"""

from sklearn.neighbors import LocalOutlierFactor
from src.models.base_model import UnsupervisedModel

class LOFModel(UnsupervisedModel):
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors)

    def train(self, X):
        # LOF is an unsupervised model, no need to pass labels
        self.model.fit(X)

    def predict(self, X):
        # LOF returns -1 for outliers and 1 for inliers
        return self.model.fit_predict(X)