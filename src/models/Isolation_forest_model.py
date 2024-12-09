"""
Random Forest (RF)

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It is widely used for both classification and regression tasks. Random Forest works by training a collection of decision trees on bootstrapped samples of the data and then aggregating their predictions (via majority voting for classification or averaging for regression) to produce a more robust and accurate model.

Key Concepts:
- **Ensemble Learning**: Random Forest is an ensemble technique, meaning it combines the results of multiple models (decision trees) to improve overall performance.
- **Bootstrapping**: The algorithm creates multiple decision trees by training each tree on a random subset of the data (with replacement). This ensures diversity among the trees.
- **Feature Randomness**: In addition to bootstrapping, Random Forest introduces randomness in the feature selection process. For each split in a tree, only a random subset of features is considered, which helps reduce correlation between trees.
- **Majority Voting**: For classification tasks, Random Forest aggregates the predictions of all trees by majority voting, while for regression tasks, it takes the average of the predictions.

Working:
1. **Tree Construction**: Random Forest builds multiple decision trees by randomly selecting subsets of data points and features, making each tree different and reducing overfitting.
2. **Prediction**: For classification, each tree in the forest predicts a class, and the class with the most votes is selected as the final prediction. For regression, the average prediction of all trees is used.
3. **Out-of-Bag Error Estimation**: Random Forest can use out-of-bag (OOB) data (data not used in training a particular tree) to estimate the generalization error of the model.

Advantages:
- **High Accuracy**: Random Forest generally provides high accuracy and can handle complex datasets without much tuning.
- **Reduced Overfitting**: By averaging multiple decision trees, Random Forest reduces overfitting and improves generalization.
- **Handles Missing Data**: Random Forest can handle missing data by using surrogate splits and can work with both categorical and continuous data.

Disadvantages:
- **Complexity**: Although Random Forest is powerful, it can be computationally expensive, especially with a large number of trees and features.
- **Model Interpretability**: While individual decision trees are interpretable, the ensemble nature of Random Forest makes it less interpretable compared to a single decision tree.

Applications:
- **Classification**: Random Forest is widely used for classification tasks, such as fraud detection, image classification, and medical diagnosis.
- **Regression**: It is also used for regression tasks, such as predicting house prices, stock market trends, and demand forecasting.
- **Feature Importance**: Random Forest can be used to assess the importance of different features in making predictions, which is useful in feature selection and understanding the dataset.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination=0.05):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.scaler = StandardScaler()

    def preprocess_data(self, df):
        """Preprocess the data by extracting time features and scaling oxygen levels."""
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        X = df[['hour', 'day_of_week', 'Oxygen']]
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    def train(self, X):
        """Train the Isolation Forest model."""
        print("Training Isolation Forest model...")
        self.model.fit(X)
        print(f"Model trained with contamination ratio: {self.model.contamination}")

    def predict(self, X):
        """Predict anomalies using Isolation Forest model."""
        print("Predicting anomalies...")
        # Predict anomalies: -1 for anomalies, 1 for normal points
        anomaly_labels = self.model.predict(X)
        anomalies = anomaly_labels == -1
        anomaly_scores = self.model.decision_function(X)
        return anomalies, anomaly_scores

    def visualize(self, df, anomalies):
        """Visualize the anomalies on the time vs Oxygen plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(df['time'], df['Oxygen'], c=anomalies, cmap='coolwarm', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Oxygen')
        plt.title('Anomaly Detection using Isolation Forest')
        plt.colorbar(label='Anomaly (1) or Normal (0)')
        plt.show()

    def run_pipeline(self, df):
        """Run the full anomaly detection pipeline."""
        X_scaled = self.preprocess_data(df)
        self.train(X_scaled)
        anomalies, anomaly_scores = self.predict(X_scaled)
        df['anomaly'] = anomalies
        self.visualize(df, anomalies)

        return df, anomalies, anomaly_scores


# Usage
if __name__ == "__main__":
    from src.utils.get_data import get_data

    data = get_data("1T")
    rf_model = IsolationForestModel(n_estimators=100, contamination=0.05)
    result_df, anomalies, anomaly_scores = rf_model.run_pipeline(data)

    print(f"Detected anomalies: {sum(anomalies)}")
    print(result_df.head())