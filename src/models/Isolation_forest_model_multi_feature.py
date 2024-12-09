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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple

from config import TIME_COLUMN


class IsolationForestModel:
    def __init__(self, time_column: str, feature_columns: List[str], n_estimators: int = 100, contamination: float = 0.05):
        """
        Initialize the Isolation Forest Model.
        :param time_column: Name of the time column in the dataset.
        :param feature_columns: List of feature column names to include in the model.
        :param n_estimators: Number of estimators (trees) in the Isolation Forest.
        :param contamination: Proportion of outliers in the data.
        """
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.scaler = StandardScaler()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by extracting time features and scaling the feature columns."""
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df['hour'] = df[self.time_column].dt.hour
        df['day_of_week'] = df[self.time_column].dt.dayofweek

        # Select the features
        features = df[['hour', 'day_of_week'] + self.feature_columns]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')  # You can use 'mean', 'median', or a constant
        features_imputed = imputer.fit_transform(features)
        print(features_imputed)

        # Scale the features
        scaled_features = self.scaler.fit_transform(features_imputed)

        return scaled_features

    def train(self, X: pd.DataFrame) -> None:
        """Train the Isolation Forest model."""
        print("Training Isolation Forest model...")
        self.model.fit(X)
        print(f"Model trained with contamination ratio: {self.model.contamination}")

    def predict(self, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Predict anomalies using the Isolation Forest model."""
        print("Predicting anomalies...")
        anomaly_labels = self.model.predict(X)  # -1 for anomalies, 1 for normal
        anomalies = anomaly_labels == -1
        anomaly_scores = self.model.decision_function(X)
        return anomalies, anomaly_scores

    def visualize(self, df: pd.DataFrame, anomalies: pd.Series, value_column: str) -> None:
        """Visualize the anomalies on a time vs feature plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(df[self.time_column], df[value_column], c=anomalies, cmap='coolwarm', marker='o')
        plt.xlabel(self.time_column)
        plt.ylabel(value_column)
        plt.title(f'Anomaly Detection on {value_column} using Isolation Forest')
        plt.colorbar(label='Anomaly (1) or Normal (0)')
        plt.show()

    def run_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the full anomaly detection pipeline.
        :param df: Input DataFrame.
        :param value_column: The feature to visualize anomalies against time.
        :return: Updated DataFrame with anomaly labels and scores.
        """
        X_scaled = self.preprocess_data(df)
        self.train(X_scaled)
        anomalies, anomaly_scores = self.predict(X_scaled)

        # For each feature column, visualize anomalies separately
        for feature in self.feature_columns:
            df['anomaly'] = anomalies
            self.visualize(df, anomalies, value_column=feature)
            print(f"Detected anomalies in {feature}: {sum(anomalies)}")

        df['anomaly'] = anomalies
        return df, anomalies, anomaly_scores


# Usage
if __name__ == "__main__":
    from src.utils.get_data import get_data, align_time_in_csv

    input_data = get_data()
    input_data = align_time_in_csv(input_data, time_column=TIME_COLUMN)
    TIME_COLUMN = "time"
    FEATURE_COLUMNS = ['Bisulfide', 'CO2', 'Conductivity', 'H2S', 'Nitrate', 'Nitrite','Oxygen', 'PH', 'TOCeq', 'Temperature', 'Turbidity', 'UV254f', 'UV254t']

    rf_model = IsolationForestModel(
        time_column=TIME_COLUMN,
        feature_columns=FEATURE_COLUMNS,
        n_estimators=100,
        contamination=0.05
    )
    result_df, anomalies, anomaly_scores = rf_model.run_pipeline(input_data)

    print(result_df.head())