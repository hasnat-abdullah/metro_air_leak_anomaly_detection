"""
K-Means Clustering

K-Means is a popular unsupervised machine learning algorithm used for clustering tasks. The algorithm divides data into K clusters, where each cluster is represented by the mean (centroid) of its members. K-Means aims to minimize the variance within clusters by iteratively adjusting cluster assignments and centroids.

Key Concepts:
- Clustering: The core idea of K-Means is to partition the data into K distinct, non-overlapping clusters based on the similarity of data points. Each cluster is represented by a centroid, which is the mean of the points within that cluster.
- Centroid: The centroid is the center of a cluster, calculated as the mean of all the data points within the cluster. During the algorithm's execution, the centroids are updated to minimize the distance between the points and their corresponding centroids.
- Iterative Process: The K-Means algorithm follows an iterative process consisting of two main steps:
  1. **Assignment Step**: Each data point is assigned to the cluster whose centroid is closest (based on distance metrics such as Euclidean distance).
  2. **Update Step**: The centroid of each cluster is recomputed as the mean of the points assigned to it. This process repeats until convergence, i.e., when the centroids no longer change significantly.
- Convergence: The algorithm converges when the assignments of data points to clusters no longer change, or the centroids stop moving. This typically occurs after a predefined number of iterations or when the change in centroids is below a certain threshold.

Parameters:
- Number of clusters (K): The number of clusters (K) is a critical parameter that must be set before training. The choice of K can significantly impact the performance of the algorithm. A common approach to selecting K is using the Elbow Method or the Silhouette Score.
- Initialization method: The method used to initialize the centroids. The most common method is **K-Means++**, which helps choose better initial centroids to improve convergence speed and reduce the likelihood of poor results.
- Distance metric: The distance metric used to calculate the similarity between data points and centroids. The default is typically **Euclidean distance**, but other metrics can be used depending on the application.
- Max iterations: The maximum number of iterations to run the algorithm before stopping, typically used as a stopping condition.
- Tolerance (tolerance for convergence): A threshold for determining when the algorithm has converged (i.e., when centroid movements are minimal).

Applications:
- Market segmentation: Grouping customers into segments with similar behaviors or preferences to tailor marketing strategies.
- Image compression: Reducing the number of colors in an image by clustering similar colors and representing them with a single centroid.
- Document clustering: Grouping documents into topics or themes based on their content, useful in text mining and information retrieval.
- Anomaly detection: Identifying outliers in data by determining which data points do not belong to any of the clusters.

Limitations:
- **Choosing K**: Selecting the appropriate number of clusters (K) can be challenging, as it requires prior knowledge or estimation. This can lead to suboptimal results if K is incorrectly chosen.
- **Sensitivity to initialization**: The algorithm can converge to local minima depending on the initial placement of centroids. This is why K-Means++ is often preferred over random initialization.
- **Sensitivity to outliers**: K-Means is sensitive to outliers, as they can heavily influence the centroids. Robust versions, like K-Medoids, can handle outliers better.
- **Assumes spherical clusters**: K-Means assumes that clusters are spherical and equally sized, which may not hold in cases where clusters have different shapes or densities.

Benefits:
- Simple and easy to implement.
- Efficient for clustering large datasets.
- Scalable with respect to the number of data points and features.

Variants:
- **K-Means++**: A smarter initialization method to reduce the chances of poor local minima.
- **Mini-Batch K-Means**: A variant that uses small random samples of the data to update centroids, improving scalability and performance for large datasets.
- **K-Medoids**: Similar to K-Means, but it uses actual data points as centroids rather than the mean, making it more robust to outliers.

K-Means is widely used for clustering problems due to its simplicity and efficiency, though its performance can degrade if the clusters have complex shapes or if the data contains significant outliers.
"""

from sklearn.cluster import KMeans

from src.models.base_model import UnsupervisedModel


class KMeansModel(UnsupervisedModel):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)