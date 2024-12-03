from datetime import datetime
import traceback
import pandas as pd
from src.data.loader import CSVDataLoader
from src.data.preprocessing import Preprocessor
from src.models import *
from src.evaluation.metrics import UnsupervisedEvaluator

from src.utils.others import create_result_output_folder
from src.utils.model_saver import ModelSaver
from config import CSV_FILE_PATH, TIME_COLUMN, VALUE_COLUMN, TRAINED_MODEL_SAVE_PATH

RESULT_OUTPUT_FOLDER = "results"

def train(data: pd.DataFrame, time_column, value_column):
    # -------- Preprocess data --------
    preprocessor = Preprocessor(data=data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)
    data = preprocessor.preprocess()

    # -------- Anomaly detection models to train --------
    models = {
        # "Isolation Forest": IsolationForestModel(contamination=0.01),
        # "AutoEncoder": AutoEncoder(input_dim=data.shape[1]),
        # "One-Class SVM": OneClassSVMModel(kernel="rbf", nu=0.05),
        # "LODA": LODAModel(n_bins=10),
        # "LSTM": LSTMModel(input_dim=data.shape[1]),
        # "Mahalanobis Distance": MahalanobisModel(),
        # "KNN": KNNModel(),
        # "Random Cut Forest": RandomCutForest(),
        # "DBSCAN": DBSCANModel(eps=0.3, min_samples=10),
        # "Local Outlier Factor": LOFModel(n_neighbors=20),
        # "Deep AutoEncoder": DeepAutoEncoder(input_dim=data.shape[1]),
        # "Variational Autoencoder": VAEModel(input_dim=data.shape[1]),
        # "Gaussian Mixture Model": GMMModel(n_components=2),
        # "Robust PCA": RPCA(),
        # "K-Means": KMeansModel(n_clusters=2),
        # "LSTM Autoencoder": LSTMAutoencoder(input_dim=10, timesteps=5, latent_dim=3),
        # "S-H-ESD": SHESDModel(alpha=0.05),
        "TimeGAN": TimeGAN(),
        # "DeepAD": DeepAD(input_dim=10),
        # "Bayesian Network": BayesianNetworkModel(),
        # "Gaussian Process": GaussianProcessModel(),
        "Prophet": ProphetModel(),
        # "DTW": DTWModel(reference_series=[1, 2, 3]),
        # "Matrix Profile": MatrixProfileModel(window_size=10),
        # "AdaBoost": AdaBoostModel(n_estimators=100),
        # "Arima": ARIMAModel(),
    }

    # -------- trained model evaluation --------
    # supervised_evaluator = SupervisedEvaluator()
    unsupervised_evaluator = UnsupervisedEvaluator()

    # -------- train models --------
    output_folder = create_result_output_folder(results_dir=TRAINED_MODEL_SAVE_PATH)
    results = {}
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            start_time = datetime.now()

            # if model.MODEL_TYPE==ModelType.SUPERVISED:
            #     model.train(X_train, y_train)
            #     y_pred = model.predict(X_test)
            #     results[name] = supervised_evaluator.evaluate(name, y_test, y_pred)
            # else:

            model.train(data)
            scores = model.predict(data)
            results[name] = unsupervised_evaluator.evaluate(name, data, scores)

            # Record training time
            end_time = datetime.now()
            training_time = end_time - start_time
            results[name]["training_time"] = training_time

            # -------- Save model to the batch folder --------
            model_filename = f"{output_folder}/models/{name.lower().replace(' ', '_')}_model.pkl"
            ModelSaver.save_model(model, model_filename)

        except Exception as ex:
            print(ex)
            traceback.print_exc()

    # -------- Print models evaluation --------
    print("\nEvaluation Results:")
    df = pd.DataFrame.from_dict(results, orient='index')
    pd.set_option('display.max_columns', None)
    print(df)

    # -------- Store results to CSV in the batch folder --------
    # supervised_evaluator.store_results(output_path=f'{output_folder}/supervised_results.csv')
    unsupervised_evaluator.store_results(output_path=f'{output_folder}/unsupervised_results.csv')

if __name__ == "__main__":

    # -------- Import data from Postgres --------
    # data_loader = PostgreSQLDataLoader(db_url=DATABASE_URL, query=QUERY)
    # df = data_loader.load_data()

    # -------- Import data from CSV --------
    data_loader = CSVDataLoader(file_path=CSV_FILE_PATH, usecols=[TIME_COLUMN, VALUE_COLUMN]) # import only time_column and value_column
    data = data_loader.load_data()

    train(data=data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)

