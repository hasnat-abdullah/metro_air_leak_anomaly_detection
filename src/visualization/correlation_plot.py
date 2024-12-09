import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import TIME_COLUMN


def calculate_and_visualize_correlation(df: pd.DataFrame, time_column: str = 'time', method: str = 'pearson'):
    """
    Calculate and visualize the correlation matrix for a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical features.
    - time_column (str): The column name for time, to be excluded from correlation calculations.
    - method (str): Correlation method ('pearson', 'spearman', or 'kendall').

    Returns:
    - pd.DataFrame: The correlation matrix.
    """
    if time_column in df.columns:
        feature_df = df.drop(columns=[time_column])
    else:
        feature_df = df

    correlation_matrix = feature_df.corr(method=method)

    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Feature Correlation Heatmap ({method.capitalize()} method)')
    plt.show()

    return correlation_matrix

if __name__ == "__main__":
    from src.utils.get_data import get_data
    input_data = get_data("50min")

    correlation_matrix = calculate_and_visualize_correlation(input_data, time_column=TIME_COLUMN, method='pearson')
