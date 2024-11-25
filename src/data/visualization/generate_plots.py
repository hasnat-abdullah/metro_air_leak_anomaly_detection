import pandas as pd

from src.data.loader import DataLoader, CSVDataLoader, PostgreSQLDataLoader, InfluxDBDataLoader
from src.data.visualization.plotter import Plotter

def generate(data_loader: DataLoader,time_column, value_column,save_dir):
    data = data_loader.load_data()

    # preprocess data
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
    data = data.sort_values(time_column)
    print(data[value_column].describe())

    # Plots
    plotter = Plotter(data,time_column, value_column,save_dir)

    plotter.plot_values_over_time()
    # plotter.plot_scatter_plot_for_outliers()
    # plotter.plot_boxplot_for_anomalies()
    # plotter.plot_distribution_of_values()
    # plotter.plot_rolling_average(window=1000)
    # plotter.plot_seasonal_decomposition(period=24)
    # plotter.plot_heatmap_of_hourly_trends()
    # plotter.plot_heatmap_of_hourly_aggregated_data()
    # plotter.plot_anomaly_highlight(lower_threshold = 10, upper_threshold = 90)
    # plotter.plot_violin_plot_by_hour()
    # plotter.plot_scatter_plot_with_moving_averages(short_window=50, long_window = 200)
    # plotter.plot_lag_plot()

if __name__ == "__main__":
    save_dir = "./image"
    data_loader = CSVDataLoader(file_path="../../../raw_data/oxygen_sample.csv")
    generate(data_loader=data_loader, time_column="time", value_column="Oxygen", save_dir=save_dir )
