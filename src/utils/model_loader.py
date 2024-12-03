import joblib

class ModelLoader:
    @staticmethod
    def load_model(model_file_path):
        """Load the trained model from the saved file."""
        model = joblib.load(model_file_path)
        print(f"Model '{model_file_path}' loaded successfully.")
        print(model)
        return model