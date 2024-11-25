"""
     _
 __(.)<
 \_)_)
Duck! yes, duck-typing is implemented here. interesting haa!
This script is here to give you beautiful, insightful, and maybe even magical visualizations of your data.
Each plot is like a carefully crafted artwork—except with data instead of paint.
Grab your brush (or, um, your keyboard), and let's create some masterpieces together.
And yes, we’ll even save those masterpieces because, you know, who wants to lose their genius ideas?

Happy plotting, and remember—data never lies (unless it's an outlier).
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot

# Setting up a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Plotter:
    def __init__(self, data: pd.DataFrame, time_column: str = 'time', value_column: str = 'Oxygen',
                 save_dir: str = './images'):
        self.data = data
        self.time_column = time_column
        self.value_column = value_column
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _save_plot(self, plot_name: str):
        """Helper method to save the plot."""
        file_path = os.path.join(self.save_dir, plot_name)
        plt.savefig(file_path, bbox_inches='tight')
        logger.info(f"Plot saved to {file_path}")

    def _plot_and_save(self, plot_func, plot_name: str):
        """Generic method to handle plotting and saving."""
        plt.figure(figsize=(14, 6))
        plot_func()  # Call the plot function passed to it
        plt.legend(loc='right')
        plt.grid(True)
        plt.tight_layout()
        self._save_plot(plot_name)
        plt.show()

    def plot_values_over_time(self):
        """Plot values over time."""

        def plot_func():
            plt.plot(self.data[self.time_column], self.data[self.value_column], label=self.value_column, alpha=0.7,
                     linewidth=1)
            plt.title(f'{self.value_column} Levels Over Time')
            plt.xlabel('Time')
            plt.ylabel(f'{self.value_column} Level')

        self._plot_and_save(plot_func, f'{self.value_column}_over_time.png')

    def plot_scatter_plot_for_outliers(self, alpha: float = 0.5, size: int = 10, color: str = 'blue'):
        """Plot a scatter plot for outliers in the values."""

        def plot_func():
            plt.scatter(self.data[self.time_column], self.data[self.value_column], alpha=alpha, s=size, c=color)
            plt.title(f'Scatter Plot of {self.value_column} Levels')
            plt.xlabel('Time')
            plt.ylabel(f'{self.value_column} Level')

        self._plot_and_save(plot_func, f'{self.value_column}_scatter_plot_outliers.png')

    def plot_boxplot_for_anomalies(self):
        """Plot a boxplot to detect anomalies in the values."""

        def plot_func():
            sns.boxplot(x=self.data[self.value_column])
            plt.title(f'{self.value_column} Levels - Boxplot (Anomaly Detection)')

        self._plot_and_save(plot_func, f'{self.value_column}_boxplot.png')

    def plot_distribution_of_values(self):
        """Plot the distribution of values."""

        def plot_func():
            sns.histplot(self.data[self.value_column], kde=True, bins=50, color='blue')
            plt.title(f'Distribution of {self.value_column} Levels')
            plt.xlabel(f'{self.value_column} Level')
            plt.ylabel('Frequency')

        self._plot_and_save(plot_func, f'{self.value_column}_distribution.png')

    def plot_rolling_average(self, window: int = 1000):
        """Plot values with rolling average."""
        self.data[f'{self.value_column}_RollingMean'] = self.data[self.value_column].rolling(window=window).mean()

        def plot_func():
            plt.plot(self.data[self.time_column], self.data[self.value_column], label=self.value_column, alpha=0.3,
                     linewidth=1)
            plt.plot(self.data[self.time_column], self.data[f'{self.value_column}_RollingMean'],
                     label=f'Rolling Mean (Window={window})', color='red', linewidth=2)
            plt.title(f'{self.value_column} Levels with Rolling Mean')
            plt.xlabel('Time')
            plt.ylabel(f'{self.value_column} Level')

        self._plot_and_save(plot_func, f'{self.value_column}_rolling_mean.png')

    def plot_seasonal_decomposition(self, period: int = 24):
        """Plot seasonal decomposition of values."""
        data_resampled = self.data.set_index(self.time_column).resample('H').mean()
        decomposition = seasonal_decompose(data_resampled[self.value_column].dropna(), model='additive', period=period)

        def plot_func():
            plt.rcParams.update({'figure.figsize': (12, 8)})
            decomposition.plot()
            plt.suptitle(f'Seasonal Decomposition of {self.value_column} Levels', fontsize=16)

        self._plot_and_save(plot_func, f'{self.value_column}_seasonal_decomposition.png')

    def plot_heatmap_of_hourly_trends(self):
        """Plot a heatmap showing hourly trends of values."""
        self.data['hour'] = self.data[self.time_column].dt.hour
        self.data['day'] = self.data[self.time_column].dt.dayofweek
        pivot_table = self.data.pivot_table(index='hour', columns='day', values=self.value_column, aggfunc='mean')

        def plot_func():
            sns.heatmap(pivot_table, cmap='coolwarm', annot=False, cbar=True)
            plt.title(f'Average {self.value_column} Levels by Hour and Day of Week')
            plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
            plt.ylabel('Hour of Day')

        self._plot_and_save(plot_func, f'{self.value_column}_heatmap_hourly_trends.png')

    def plot_heatmap_of_hourly_aggregated_data(self):
        """Plot a heatmap of hourly aggregated values."""
        self.data['hour'] = self.data[self.time_column].dt.hour
        self.data['day'] = self.data[self.time_column].dt.day
        hourly_pivot = self.data.pivot_table(index='day', columns='hour', values=self.value_column, aggfunc='mean')

        def plot_func():
            sns.heatmap(hourly_pivot, cmap='coolwarm', annot=False, cbar=True)
            plt.title(f'Heatmap of Hourly Average {self.value_column} Levels')
            plt.xlabel('Hour of Day')
            plt.ylabel('Day of Month')

        self._plot_and_save(plot_func, f'{self.value_column}_hourly_aggregated_data_heatmap.png')

    def plot_anomaly_highlight(self, lower_threshold: float = 10, upper_threshold: float = 90):
        """Highlight anomalies in values."""
        anomalies = self.data[
            (self.data[self.value_column] < lower_threshold) | (self.data[self.value_column] > upper_threshold)]

        def plot_func():
            plt.plot(self.data[self.time_column], self.data[self.value_column], label=self.value_column, alpha=0.7,
                     linewidth=1)
            plt.scatter(anomalies[self.time_column], anomalies[self.value_column], color='red', label='Anomalies',
                        zorder=5)
            plt.title(f'{self.value_column} Levels with Anomalies Highlighted')
            plt.xlabel('Time')
            plt.ylabel(f'{self.value_column} Level')

        self._plot_and_save(plot_func, f'{self.value_column}_anomaly_highlight.png')

    def plot_violin_plot_by_hour(self):
        """Plot a violin plot showing values by hour of the day."""

        def plot_func():
            sns.violinplot(x=self.data['hour'], y=self.data[self.value_column])
            plt.title(f'Violin Plot of {self.value_column} Levels by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel(f'{self.value_column} Level')

        self._plot_and_save(plot_func, f'{self.value_column}_violin_plot_by_hour.png')

    def plot_scatter_plot_with_moving_averages(self, short_window: int = 50, long_window: int = 200):
        """Plot scatter plot with short-term and long-term moving averages."""
        self.data['short_mavg'] = self.data[self.value_column].rolling(window=short_window).mean()
        self.data['long_mavg'] = self.data[self.value_column].rolling(window=long_window).mean()

        def plot_func():
            plt.scatter(self.data.index, self.data[self.value_column], alpha=0.5, label=self.value_column)
            plt.plot(self.data.index, self.data['short_mavg'], label=f'Short-Term Avg ({short_window})', color='blue',
                     linewidth=2)
            plt.plot(self.data.index, self.data['long_mavg'], label=f'Long-Term Avg ({long_window})', color='orange',
                     linewidth=2)
            plt.title(f'Moving Averages with Scatter Plot of {self.value_column}')
            plt.xlabel('Time')
            plt.ylabel(f'{self.value_column} Level')
            plt.legend(loc='best')

        self._plot_and_save(plot_func, f'{self.value_column}_moving_averages.png')

    def plot_lag_plot(self):
        """Plot lag plot for values."""

        def plot_func():
            lag_plot(self.data[self.value_column])
            plt.title(f'Lag Plot of {self.value_column} Levels')
            plt.xlabel(f'{self.value_column}(t)')
            plt.ylabel(f'{self.value_column}(t+1)')

        self._plot_and_save(plot_func, f'{self.value_column}_lag_plot.png')
