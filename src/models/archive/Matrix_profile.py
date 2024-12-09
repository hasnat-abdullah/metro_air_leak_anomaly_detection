"""
Matrix Profile:
- A data mining technique that identifies patterns, motifs, and anomalies in time series data.
- Computes the similarity between subsequences using a sliding window.

Key Concepts:
- Self-Join and AB-Join: Measures similarity within or between time series.
- Motif Discovery: Identifies recurring patterns.
- Anomaly Detection: Spots subsequences with low similarity scores.

Advantages:
- Scalable for large datasets using efficient algorithms.
- Handles time series with diverse patterns.
- Provides interpretable results for motif and anomaly discovery.

Applications:
- Anomaly detection in sensor readings.
- Identifying patterns in genomic sequences.
- Monitoring financial time series data.
"""

import numpy as np
import stumpy
from src.models.base_model import UnsupervisedModel


class MatrixProfileModel(UnsupervisedModel):
    def __init__(self, window_size):
        self.window_size = window_size

    def train(self, X):
        """No training necessary for matrix profile, just data prep if needed."""
        pass

    def predict(self, X):
        """Compute matrix profile using stumpy."""
        # Ensure that X is a numpy array
        X = np.asarray(X)

        # Compute the matrix profile using STUMPY
        mp, _ = stumpy.stump(T_A=X, m=self.window_size)

        return mp