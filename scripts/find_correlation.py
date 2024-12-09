from config import TIME_COLUMN
from src.utils.get_data import get_data, align_time_in_csv
from src.visualization.correlation_plot import calculate_and_visualize_correlation

def main():
    input_data = get_data("50min")
    df = align_time_in_csv(input_data, time_column=TIME_COLUMN, granularity='50s')
    correlation_matrix = calculate_and_visualize_correlation(df, time_column=TIME_COLUMN, method='pearson')

if __name__ == '__main__':
    main()