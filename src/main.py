from data.loader import DataLoader
from data.preprocessing import Preprocessor
from models.isolation_forest import IsolationForestModel
from models.autoencoder import AutoEncoder
from models.one_class_SVM import OneClassSVMModel
from models.random_forest import RandomForestModel
from models.loda import LODAModel
from models.lstm import LSTMModel
from models.mahalanobis import MahalanobisModel
from models.random_cut_forest import RandomCutForest
from models.xgboost import XGBoostModel
from evaluation.metrics import SupervisedEvaluator, UnsupervisedEvaluator
from utils.model_saver import ModelSaver
from config import DATABASE_URL, QUERY
from src.models.base_model import SupervisedModel


def main():
    # Load data
    loader = DataLoader(DATABASE_URL)
    df = loader.load_data(QUERY)

    # Preprocess data
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

    # Initialize models
    models = {
        # "Isolation Forest": IsolationForestModel(contamination=0.01),
        "AutoEncoder": AutoEncoder(input_dim=X_train.shape[1]),
        # "One-Class SVM": OneClassSVMModel(kernel="rbf", nu=0.05),
        "LODA": LODAModel(n_bins=10),
        # "LSTM": LSTMModel(input_dim=X_train.shape[1]),
        # "Mahalanobis Distance": MahalanobisModel(),
        # "Random Cut Forest": RandomCutForest(),
        # "XGBoost": XGBoostModel(),
    }

    # Initialize evaluators
    supervised_evaluator = SupervisedEvaluator()
    unsupervised_evaluator = UnsupervisedEvaluator()

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")

        # Train the model (supervised or unsupervised)
        if isinstance(model, SupervisedModel):
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = supervised_evaluator.evaluate(name, y_test, y_pred)
        else:
            model.train(X_train)
            scores = model.predict(X_test)
            results[name] = unsupervised_evaluator.evaluate(name, y_test, scores)

        # Save model
        ModelSaver.save_model(model, f"{name.lower().replace(' ', '_')}_model.pkl")

    # Display results
    print("\nEvaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    # Store results to CSV
    supervised_evaluator.store_results()
    unsupervised_evaluator.store_results()


if __name__ == "__main__":
    main()