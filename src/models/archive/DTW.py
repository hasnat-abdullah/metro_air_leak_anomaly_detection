"""
Dynamic Time Warping (DTW):
- A time series analysis method that measures similarity between sequences by aligning them non-linearly.
- Allows for flexible alignment, even with varying speeds or phases in the data.

Key Concepts:
- Warping Path: Matches points in two sequences to minimize cumulative distance.
- Dynamic Programming: Efficiently computes the optimal alignment.
- Robust Similarity: Handles time series with shifts or distortions.

Advantages:
- Works well with time series of different lengths.
- Handles temporal misalignments effectively.
- Interpretable distance metric for time series comparison.

Applications:
- Anomaly detection in sensor signals.
- Pattern recognition in speech or handwriting.
- Gesture analysis in video data.
"""

from dtaidistance import dtw
from src.models.base_model import UnsupervisedModel


class DTWModel(UnsupervisedModel):
    def __init__(self, reference_series):
        self.reference_series = reference_series

    def train(self, X):
        # DTW doesn't require explicit training
        pass

    def predict(self, X):
        distances = [dtw.distance(series, self.reference_series) for series in X.T]
        return distances