"""
Bayesian Network:
- A probabilistic graphical model representing variables and their conditional dependencies using a directed acyclic graph (DAG).
- Captures the joint probability distribution of variables to reason under uncertainty.

Key Concepts:
- Conditional Independence: Encodes relationships between variables using probabilities.
- Directed Acyclic Graph (DAG): Represents dependencies among variables.
- Inference: Uses Bayes' theorem to calculate probabilities of unknown variables.

Advantages:
- Handles uncertainty and incomplete data effectively.
- Scalable for high-dimensional datasets.
- Interpretable structure for domain experts.

Applications:
- Fault diagnosis in industrial systems.
- Risk assessment in finance and insurance.
- Medical decision support systems.
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class BayesianNetworkModel:
    def __init__(self):
        self.model = None
        self.inference = None

    def preprocess_data(self, data: pd.DataFrame):
        """
        Preprocess the data to add features for Bayesian Network.
        Args:
            data (pd.DataFrame): The dataset with 'time' and 'value' columns.

        Returns:
            pd.DataFrame: The preprocessed dataset with additional features.
        """
        data['hour'] = data['time'].dt.hour  # Extract hour
        data['day_period'] = pd.cut(data['hour'], bins=[-1, 6, 12, 18, 24],
                                    labels=['Night', 'Morning', 'Afternoon', 'Evening'])  # Time bucket
        data['trend'] = data['Oxygen'].diff().fillna(0).apply(lambda x: 'Increase' if x > 0 else 'Decrease')
        return data.dropna()

    def train(self, data: pd.DataFrame, structure: list):
        """
        Train the Bayesian Network with the given data and structure.
        Args:
            data (pd.DataFrame): Preprocessed dataset with features.
            structure (list): List of tuples representing Bayesian Network edges.
        """
        self.model = BayesianNetwork(structure)
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)

    def predict(self, test_data: pd.DataFrame, target: str):
        """
        Predict the target variable using the Bayesian Network.
        Args:
            test_data (pd.DataFrame): Test data with features.
            target (str): Target variable to predict.

        Returns:
            pd.Series: Predicted values for the target variable.
        """
        if not self.inference:
            raise ValueError("Model is not trained. Train the model before making predictions.")

        predictions = []
        for _, row in test_data.iterrows():
            observed = row.drop(target).dropna().to_dict()
            prediction = self.inference.map_query([target], evidence=observed)
            predictions.append(prediction[target])
        return pd.Series(predictions, index=test_data.index)

    def evaluate(self, test_data: pd.DataFrame, target: str):
        """
        Evaluate the model using MAE and RMSE.
        Args:
            test_data (pd.DataFrame): Test dataset.
            target (str): Target variable.

        Returns:
            dict: Evaluation metrics.
        """
        y_true = test_data[target]
        y_pred = self.predict(test_data, target)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        return {"MAE": mae, "RMSE": rmse}

    def plot_results(self, test_data: pd.DataFrame, target: str):
        """
        Plot true vs predicted values.
        Args:
            test_data (pd.DataFrame): Test dataset.
            target (str): Target variable.
        """
        y_true = test_data[target]
        y_pred = self.predict(test_data, target)

        plt.figure(figsize=(12, 6))
        plt.plot(test_data['time'], y_true, label='True Values', color='blue')
        plt.plot(test_data['time'], y_pred, label='Predicted Values', color='red', linestyle='--')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Oxygen')
        plt.title('True vs Predicted Values')
        plt.show()

    def run_pipeline(self, data: pd.DataFrame, target: str):
        """
        Run the complete Bayesian Network pipeline.
        Args:
            data (pd.DataFrame): Original dataset with 'time' and 'value' columns.
            target (str): Target variable.
        """
        data['time'] = pd.to_datetime(data['time'])  # Ensure time is datetime
        data = self.preprocess_data(data)

        # Split into train and test
        train_data = data.iloc[:int(0.8 * len(data))]
        test_data = data.iloc[int(0.8 * len(data)):]

        # Define structure (example)
        structure = [('day_period', 'trend'), ('trend', target), ('day_period', target)]

        # Train and evaluate
        self.train(train_data, structure)
        evaluation_results = self.evaluate(test_data, target)
        self.plot_results(test_data, target)
        return evaluation_results


# Example Usage
if __name__ == "__main__":
    from src.utils.get_data import get_data
    data = get_data("10T")

    bn_model = BayesianNetworkModel()
    results = bn_model.run_pipeline(data, target='Oxygen')
    print(results)