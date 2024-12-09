import sys
import traceback
from config import TIME_COLUMN, VALUE_COLUMN, PLOT_SAVE_PATH
from src.utils.others import track_execution_time
from src.generate_plots import generate
from src.train_models import train
import logging

logging.basicConfig(level=logging.ERROR)


@track_execution_time
def main():
    try:
        # -------- Import dat --------
        print("Loading data...")
        # check get_data to set your csv file
        # Also change the value of time_column and value_column accordingly.
        from src.utils.get_data import get_data
        input_data = get_data("50min") # example: take 1 row per 50 min


        # Validate Data
        if input_data.empty:
            print("Error: No data loaded from the CSV file. Exiting pipeline.")
            sys.exit(1)

        print("Data loaded successfully!")
        print(f"Data Overview:\n{input_data[VALUE_COLUMN].describe()}")

        # -------- Generate Visualizations --------
        print("\nGenerating visualizations...")
        generate(data=input_data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN, save_dir=PLOT_SAVE_PATH)

        # -------- Train Models --------
        print("\nStarting model training...")
        train(data=input_data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN)
        print("\nPipeline completed successfully!")

    except Exception as ex:
        print(f"An error occurred in the pipeline: {ex}")
        traceback.print_exc()


if __name__ == "__main__":
    main()