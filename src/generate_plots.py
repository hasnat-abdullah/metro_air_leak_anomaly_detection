import pandas as pd
from pandas import DataFrame

from config import CSV_FILE_PATH, VALUE_COLUMN, TIME_COLUMN, PLOT_SAVE_PATH
from src.data.loader import CSVDataLoader
from src.utils.others import track_execution_time
from src.visualization.plotter import Plotter


@track_execution_time
def generate(data: DataFrame, time_column, value_column, save_dir):
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
    data = data.sort_values(time_column)

    # Plots
    plotter = Plotter(data, time_column, value_column, save_dir)

    plotter.plot_values_over_time()
    plotter.plot_scatter_plot_for_outliers()
    plotter.plot_boxplot_for_anomalies()
    plotter.plot_distribution_of_values()
    plotter.plot_rolling_average(window=1000)
    plotter.plot_seasonal_decomposition(period=24)
    plotter.plot_heatmap_of_hourly_trends()
    plotter.plot_heatmap_of_hourly_aggregated_data()
    plotter.plot_anomaly_highlight(lower_threshold=10, upper_threshold=90)
    plotter.plot_violin_plot_by_hour()
    plotter.plot_scatter_plot_with_moving_averages(short_window=50, long_window=200)
    plotter.plot_lag_plot()


if __name__ == "__main__":
    # -------- Import data from CSV --------
    data_loader = CSVDataLoader(file_path=CSV_FILE_PATH)
    data = data_loader.load_data()

    data['time'] = pd.to_datetime(data['time'])
    start_time = pd.to_datetime('2023-12-01 00:00:00')
    end_time = '2024-02-29 04:55:02'
    data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]

    # -------- Show Data overview --------
    print(data[VALUE_COLUMN].describe())

    # -------- Generate Plots --------
    generate(data=data, time_column=TIME_COLUMN, value_column=VALUE_COLUMN, save_dir=PLOT_SAVE_PATH)
