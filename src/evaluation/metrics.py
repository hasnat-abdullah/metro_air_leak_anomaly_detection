from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Abstract Base Class for model evaluation
class BaseEvaluator(ABC):
    def __init__(self):
        self.evaluations = []

    @abstractmethod
    def evaluate(self, model_name: str, y_true, y_pred_or_scores):
        pass

    def store_results(self, output_path="model_evaluations.csv"):
        # Convert evaluations to a DataFrame and save to CSV
        df = pd.DataFrame(self.evaluations)
        df.to_csv(output_path, index=False)
        print(f"Evaluations saved to {output_path}")


class SupervisedEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        # Define supervised metrics
        self.metrics = [roc_auc_score, f1_score, precision_score, recall_score, accuracy_score]

    def evaluate(self, model_name: str, y_true, y_pred):
        # Calculate metrics for supervised model
        metrics = {"Model": model_name}
        for metric in self.metrics:
            metrics[metric.__name__] = metric(y_true, y_pred)

        self.evaluations.append(metrics)
        return metrics


class UnsupervisedEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        # Define unsupervised metrics
        self.metrics = [roc_auc_score, f1_score, precision_score, recall_score]

    def evaluate(self, model_name: str, y_true, scores, threshold=0.5):
        # Binarize predictions based on threshold
        predictions = (scores >= threshold).astype(int)

        # Calculate metrics for unsupervised model
        metrics = {"Model": model_name}
        for metric in self.metrics:
            metrics[metric.__name__] = metric(y_true, predictions)

        self.evaluations.append(metrics)
        return metrics