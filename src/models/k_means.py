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

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt


class KMeansAnomalyDetection:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = MinMaxScaler()
        self.kmeans = None
        self.threshold = None

    def preprocess_data(self, df):
        """Scale the data for K-Means."""
        df['Oxygen_scaled'] = self.scaler.fit_transform(df[['Oxygen']])
        return df[['Oxygen_scaled']].values

    def train(self, X):
        """Train the K-Means model and calculate the distance threshold for anomalies."""
        print("Training K-Means model...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        print("Model training completed.")

        # Calculate distances of all points to their closest cluster center
        distances = euclidean_distances(X, self.kmeans.cluster_centers_).min(axis=1)
        # Define threshold as the mean + 3 standard deviations
        self.threshold = distances.mean() + 3 * distances.std()
        print(f"Anomaly detection threshold: {self.threshold}")

    def predict_anomalies(self, X):
        """Predict anomalies based on the distance threshold."""
        distances = euclidean_distances(X, self.kmeans.cluster_centers_).min(axis=1)
        anomalies = distances > self.threshold
        return anomalies, distances

    def visualize_anomalies(self, df, anomalies, anomaly_scores):
        """Visualize anomalies on the time vs. Oxygen plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['Oxygen'], label='Oxygen', color='blue')
        plt.scatter(
            df['time'][anomalies],
            df['Oxygen'][anomalies],
            label='Anomalies',
            color='red'
        )
        plt.xlabel('Time')
        plt.ylabel('Oxygen')
        plt.title('Anomaly Detection using K-Means')
        plt.legend()
        plt.show()

    def run_pipeline(self, df):
        """Run the K-Means anomaly detection pipeline."""
        # Preprocess data
        X = self.preprocess_data(df)

        # Train the K-Means model
        self.train(X)

        # Predict anomalies
        anomalies, anomaly_scores = self.predict_anomalies(X)

        # Add anomaly information to the original dataframe
        df['anomaly_score'] = anomaly_scores
        df['anomaly'] = anomalies

        # Visualize anomalies
        self.visualize_anomalies(df, anomalies, anomaly_scores)

        return df, anomalies, anomaly_scores


# Usage
if __name__ == "__main__":
    # Assuming 'data' is your dataframe
    from src.utils.get_data import get_data
    data = get_data("1T")  # Replace with your actual data source

    kmeans_model = KMeansAnomalyDetection(n_clusters=3)
    result_df, anomalies, anomaly_scores = kmeans_model.run_pipeline(data)

    # Output results
    print(f"Detected anomalies: {sum(anomalies)}")
    print(result_df.head())