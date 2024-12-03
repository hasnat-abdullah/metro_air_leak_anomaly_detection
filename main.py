import sys
import traceback
from config import CSV_FILE_PATH, TIME_COLUMN, VALUE_COLUMN, PLOT_SAVE_PATH
from src.data.loader import CSVDataLoader
from src.utils.others import track_execution_time
from src.generate_plots import generate
from src.train_models import train


@track_execution_time
def main():
    try:
        # -------- Import data from CSV --------
        print("Loading data from CSV...")
        data_loader = CSVDataLoader(file_path=CSV_FILE_PATH, usecols=[TIME_COLUMN, VALUE_COLUMN], )
        data = data_loader.load_data()


        # Validate Data
        if data.empty:
            print("Error: No data loaded from the CSV file. Exiting pipeline.")
            sys.exit(1)

        print("Data loaded successfully!")
        print(f"Data Overview:\n{data[VALUE_COLUMN].describe()}")

        # -------- Generate Visualizations --------
        print("\nGenerating visualizations...")
        generate(data=data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN, save_dir=PLOT_SAVE_PATH)

        # -------- Train Models --------
        print("\nStarting model training...")
        train(data=data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)
        print("\nPipeline completed successfully!")

    except Exception as ex:
        print(f"An error occurred in the pipeline: {ex}")
        traceback.print_exc()


if __name__ == "__main__":
    main()