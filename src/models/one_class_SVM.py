"""
One-Class Support Vector Machine (One-Class SVM)

One-Class SVM is an unsupervised learning algorithm used for anomaly detection. It learns a decision boundary that separates the majority of the data from the origin or the outliers, making it effective for scenarios where only "normal" data is available and the goal is to identify any instances that deviate significantly from this normal behavior.

Key Concepts:
- **Support Vectors**: One-Class SVM uses support vectors to define the boundary of normal data. It tries to fit a hyperplane that best encapsulates the majority of the data points, where points outside this region are considered anomalies.
- **Kernel Trick**: The algorithm can use different kernels (e.g., linear, radial basis function (RBF)) to map data into higher-dimensional spaces, allowing for non-linear decision boundaries. This is useful when the data is not linearly separable.
- **Decision Function**: One-Class SVM learns a function that assigns a score to each data point, indicating how likely it is to belong to the same distribution as the majority of the data. Points with a low score are considered anomalies.

Working:
1. **Training**: One-Class SVM is trained using only the normal data. The algorithm attempts to find a decision boundary that encapsulates most of the data points, effectively learning the normal behavior of the dataset.
2. **Anomaly Detection**: After training, the algorithm evaluates new data points. If a point lies outside the learned boundary, it is classified as an anomaly.
3. **Parameter Tuning**: The main parameters for tuning are the kernel type (linear, RBF), the regularization parameter (C), and the decision function threshold. These parameters control the model's sensitivity to anomalies.

Advantages:
- **Unsupervised**: One-Class SVM does not require labeled data for training, making it suitable for anomaly detection tasks where only normal data is available.
- **Effective in High-Dimensional Spaces**: One-Class SVM can handle high-dimensional data using kernels to create non-linear decision boundaries.
- **Versatile**: It works well with various types of data, including non-linear, and can handle data with complex patterns.

Disadvantages:
- **Sensitivity to Parameters**: One-Class SVM is sensitive to the choice of kernel and regularization parameters. Incorrect parameter settings can lead to poor performance or overfitting.
- **Computationally Expensive**: Training a One-Class SVM can be computationally expensive, especially for large datasets and complex kernel functions.
- **Assumes Normality of Data**: One-Class SVM assumes that the majority of the data follows a similar distribution, so it may struggle with datasets that have significant variance or skew.

Applications:
- **Anomaly Detection**: One-Class SVM is widely used for anomaly detection in scenarios where the data is largely unlabelled and the goal is to detect outliers, such as fraud detection, industrial fault detection, and network intrusion detection.
- **Image and Video Anomaly Detection**: It is used in computer vision tasks for detecting anomalies in images or video frames by learning the normal distribution of visual data.
- **Time Series Anomaly Detection**: One-Class SVM can also be applied in time series analysis to detect anomalous behaviors or deviations from normal trends over time.
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from config import TIME_COLUMN, VALUE_COLUMN


class OneClassSVMModel:
    def __init__(self, time_column, value_column, kernel='rbf', nu=0.05, gamma='scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = MinMaxScaler()
        self.time_column = time_column
        self.value_column = value_column
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def preprocess_data(self, df):
        """Scale the data for One-Class SVM."""
        df[f'{self.value_column}_scaled'] = self.scaler.fit_transform(df[[self.value_column]])
        return df[[f'{self.value_column}_scaled']].values

    def train(self, X):
        """Train the One-Class SVM model."""
        print("Training One-Class SVM model...")
        self.model.fit(X)
        print("Model training completed.")

    def predict(self, X):
        """Predict anomalies using the trained model."""
        predictions = self.model.predict(X)
        anomaly_scores = self.model.decision_function(X)
        # Anomalies are labeled as -1
        anomalies = predictions == -1
        return anomalies, anomaly_scores

    def visualize(self, df, anomalies):
        """Visualize anomalies on the time vs. Oxygen plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(df[self.time_column], df[self.value_column], label=self.value_column, color='blue')
        plt.scatter(
            df[self.time_column][anomalies],
            df[self.value_column][anomalies],
            label='Anomalies',
            color='red'
        )
        plt.xlabel(self.time_column)
        plt.ylabel(self.value_column)
        plt.title('Anomaly Detection using One-Class SVM')
        plt.legend()
        plt.show()

    def run_pipeline(self, df, time_column, value_column):
        """Run the One-Class SVM anomaly detection pipeline."""
        # Preprocess data
        X = self.preprocess_data(df)

        self.train(X)
        anomalies, anomaly_scores = self.predict(X)

        df['anomaly_score'] = anomaly_scores
        df['anomaly'] = anomalies

        self.visualize(df, anomalies)
        print(f"Detected anomalies: {sum(anomalies)}")
        return df, anomalies, anomaly_scores


# Usage
if __name__ == "__main__":
    from src.utils.get_data import get_data

    data = get_data("10min")
    svm_model = OneClassSVMModel(time_column=TIME_COLUMN, value_column=VALUE_COLUMN, nu=0.05, gamma='scale')
    result_df, anomalies, anomaly_scores = svm_model.run_pipeline(data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)

    print(result_df.head())