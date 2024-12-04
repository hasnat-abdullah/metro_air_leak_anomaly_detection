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
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


class DBSCANModel:
    def __init__(self, eps=0.5, min_samples=5):
        # Initialize DBSCAN model with specified eps and min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = None

    def train(self, X):
        """Train the DBSCAN model."""
        print("Training DBSCAN...")
        self.model.fit(X)
        self.labels = self.model.labels_
        print(f"DBSCAN model trained with {len(set(self.labels))} clusters (including noise)")

    def predict(self, X):
        """Predict the labels for the input data."""
        print("Predicting with DBSCAN...")
        return self.model.fit_predict(X)

    def evaluate(self, X):
        """Evaluate the DBSCAN model using silhouette score."""
        if len(set(self.labels)) > 1:
            silhouette = silhouette_score(X, self.labels)
            print(f"Silhouette Score: {silhouette}")
            return silhouette
        else:
            print("Silhouette Score cannot be computed, only one cluster detected.")
            return None

    def visualize_clusters(self, df):
        """Visualize the clustering result on the time vs Oxygen plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(df['time'], df['Oxygen'], c=self.labels, cmap='viridis', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Oxygen')
        plt.title('DBSCAN Clustering on Oxygen Levels')
        plt.colorbar(label='Cluster Label')
        plt.show()

    def run_pipeline(self, df: pd.DataFrame):
        """Run the full DBSCAN pipeline with visualization."""
        # Ensure time is in datetime format
        df['time'] = pd.to_datetime(df['time'])

        # Preprocessing: Extract Oxygen values and use 'time' for index (but not directly in DBSCAN)
        X = df[['Oxygen']]  # DBSCAN works only on numeric data, so we use only 'Oxygen'

        # Train the DBSCAN model
        self.train(X)

        # Predict using the model
        predictions = self.predict(X)
        df['cluster'] = predictions  # Store cluster results in the dataframe

        # Evaluate the model
        silhouette = self.evaluate(X)

        # Visualize the clustering result
        self.visualize_clusters(df)

        return predictions, silhouette, df


if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data("60T")
    dbscan_model = DBSCANModel(eps=0.2, min_samples=2)
    predictions, silhouette, result_df = dbscan_model.run_pipeline(data)

    # Output results
    print(f"Predicted labels: {predictions}")
    print(f"Silhouette score: {silhouette}")
    print(result_df)