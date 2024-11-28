from datetime import datetime
import traceback
import pandas as pd
from data.loader import PostgreSQLDataLoader
from data.preprocessing import Preprocessor
from models.isolation_forest import IsolationForestModel
from models.autoencoder import AutoEncoder
from models.one_class_SVM import OneClassSVMModel
from models.random_forest import RandomForestModel
from models.loda import LODAModel
from models.lstm import LSTMModel
from models.mahalanobis import MahalanobisModel
from models.random_cut_forest import RandomCutForest
from models.knn import KNNModel
from models.vae import VAEModel
from evaluation.metrics import SupervisedEvaluator, UnsupervisedEvaluator
from src.models.dbscan import DBSCANModel
from src.models.deep_autoencoder import DeepAutoEncoder
from src.models.gmm import GMMModel
from src.models.k_means import KMeansModel
from src.models.lof import LOFModel
from src.models.rpca import RPCA
from src.utils.others import create_result_output_folder
from utils.model_saver import ModelSaver
from config import DATABASE_URL, QUERY
from src.models.base_model import SupervisedModel, UnsupervisedModel

RESULT_OUTPUT_FOLDER = "results"

def main():
    loader = PostgreSQLDataLoader(db_url=DATABASE_URL, query=QUERY)
    df = loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

    models = {
        "Isolation Forest": IsolationForestModel(contamination=0.01),
        "AutoEncoder": AutoEncoder(input_dim=X_train.shape[1]),
        "One-Class SVM": OneClassSVMModel(kernel="rbf", nu=0.05),
        "LODA": LODAModel(n_bins=10),
        "LSTM": LSTMModel(input_dim=X_train.shape[1]),
        "Mahalanobis Distance": MahalanobisModel(),
        "KNN": KNNModel(),
        "Random Cut Forest": RandomCutForest(),
        "DBSCAN": DBSCANModel(eps=0.3, min_samples=10),
        "Local Outlier Factor": LOFModel(n_neighbors=20),
        "Deep AutoEncoder": DeepAutoEncoder(input_dim=X_train.shape[1]),
        "Variational Autoencoder": VAEModel(input_dim=X_train.shape[1]),
        "Gaussian Mixture Model": GMMModel(n_components=2),
        "Robust PCA": RPCA(),
        "K-Means": KMeansModel(n_clusters=2),
    }


    supervised_evaluator = SupervisedEvaluator()
    unsupervised_evaluator = UnsupervisedEvaluator()


    output_folder = create_result_output_folder()

    results = {}
    for name, model in models.items():
        try:
            print(f"Training {name}...")


            start_time = datetime.now()

            if isinstance(model, SupervisedModel):
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = supervised_evaluator.evaluate(name, y_test, y_pred)
            else:
                model.train(X_train)
                scores = model.predict(X_test)
                results[name] = unsupervised_evaluator.evaluate(name, y_test, scores)

            # Record training time
            end_time = datetime.now()
            training_time = end_time - start_time
            results[name]["training_time"] = training_time

            # Save model to the batch folder
            model_filename = f"{output_folder}/models/{name.lower().replace(' ', '_')}_model.pkl"
            ModelSaver.save_model(model, model_filename)

        except Exception as ex:
            print(ex)
            traceback.print_exc()

    print("\nEvaluation Results:")

    df = pd.DataFrame.from_dict(results, orient='index')
    pd.set_option('display.max_columns', None)
    print(df)

    # Store results to CSV in the batch folder
    supervised_evaluator.store_results(output_path=f'{output_folder}/supervised_results.csv')
    unsupervised_evaluator.store_results(output_path=f'{output_folder}/unsupervised_results.csv')

if __name__ == "__main__":
    main()