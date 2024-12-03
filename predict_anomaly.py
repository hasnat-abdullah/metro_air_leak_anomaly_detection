"""
The script to predict anomaly score using a trained model.
"""

from config import CSV_FILE_PATH, TIME_COLUMN, VALUE_COLUMN
from src.data.loader import CSVDataLoader
from src.utils.model_loader import ModelLoader


class ModelPredictor:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.model = None

    def load_model(self):
        """Load the trained model from the saved path."""
        print('Loading model...')
        print(f'model_file_path: {self.model_file_path}')
        self.model = ModelLoader.load_model(self.model_file_path)

    def predict(self, X):
        """Predict anomalies using the loaded model."""
        scores = self.model.predict(X)
        # Convert the predictions: -1 -> 1 (anomaly), 1 -> 0 (normal)
        return [1 if score == -1 else 0 for score in scores]


def main():
    # -------- Import Test data from CSV --------
    data_loader = CSVDataLoader(file_path=CSV_FILE_PATH,
                                usecols=[TIME_COLUMN, VALUE_COLUMN])  # import only time_column and value_column
    test_data = data_loader.load_data()

    model_path = "training_results/001_2024-12-02_17-32/models/isolation_forest_model.pkl"

    model_predictor = ModelPredictor(model_file_path=model_path)
    model_predictor.load_model()

    # Make predictions (unsupervised)
    predictions = model_predictor.predict(test_data)
    test_data['status'] = predictions

    print(test_data)


if __name__ == "__main__":
    main()