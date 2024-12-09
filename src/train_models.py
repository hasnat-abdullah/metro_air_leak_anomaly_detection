from datetime import datetime
import traceback
import pandas as pd
from src.data.loader import CSVDataLoader
from src.models import *

from src.utils.others import create_result_output_folder
from src.utils.model_saver import ModelSaver
from config import CSV_FILE_PATH, TIME_COLUMN, VALUE_COLUMN, TRAINED_MODEL_SAVE_PATH

RESULT_OUTPUT_FOLDER = "results"

def train(data: pd.DataFrame, time_column, value_column):
    # -------- Anomaly detection models to train --------
    models = {
        "Isolation Forest": IsolationForestModel(time_column, value_column,n_estimators=100, contamination=0.05),
        "One-Class SVM": OneClassSVMModel(time_column=TIME_COLUMN, value_column=VALUE_COLUMN ,nu=0.05, gamma='scale'),
        "LSTM": LSTMModel(time_column=time_column, value_column=value_column,seq_length=10, epochs=50, batch_size=32),
        "DBSCAN": DBSCANModel(eps=0.2, min_samples=2),
        "K-Means": KMeansModel(time_column=time_column, value_column=value_column,n_clusters=3),
        "Prophet": ProphetModel(),
        "ARIMA": ARIMAModel(time_column=time_column, value_column=value_column, order=(1, 1, 1)),
    }

    # -------- train models --------
    output_folder = create_result_output_folder(results_dir=TRAINED_MODEL_SAVE_PATH)
    model_execution_time = {}
    for name, model in models.items():
        try:
            df= data.copy()
            print(f"Training {name}...")
            start_time = datetime.now()

            result_df, anomalies, anomaly_scores = model.run_pipeline(df, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)

            # Record training time
            end_time = datetime.now()
            training_time = end_time - start_time
            model_execution_time[name] = str(training_time)

            # -------- Save model to the batch folder --------
            model_filename = f"{output_folder}/models/{name.lower().replace(' ', '_')}_model.pkl"
            ModelSaver.save_model(model, model_filename)

        except Exception as ex:
            print(ex)
            traceback.print_exc()
            return
    print("-------Execution Time-------")
    print(pd.DataFrame(list(model_execution_time.items()), columns=['Model Name', 'Execution Time']))

if __name__ == "__main__":
    from src.utils.get_data import get_data
    input_data = get_data("50min")
    train(data=input_data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)
