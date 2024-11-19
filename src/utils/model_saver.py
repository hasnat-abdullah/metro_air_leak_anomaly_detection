from joblib import dump

class ModelSaver:
    @staticmethod
    def save_model(model, filename: str):
        dump(model, filename)