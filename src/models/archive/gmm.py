"""
Gaussian Mixture Model (GMM)

A Gaussian Mixture Model (GMM) is a probabilistic model that represents a mixture of several Gaussian distributions, each corresponding to a different cluster or component in the data. It assumes that the data points are generated from a mixture of several Gaussian distributions with unknown parameters, and the goal is to estimate those parameters.

Key Concepts:
- Mixture of Gaussians: GMM assumes that the data is generated from a combination of multiple Gaussian distributions, each with its own mean, variance, and mixing coefficient.
- Expectation-Maximization (EM): GMM uses the EM algorithm to iteratively estimate the parameters (means, variances, and mixing coefficients) of the Gaussian distributions. The process alternates between the E-step (estimating the probabilities of data points belonging to each Gaussian) and the M-step (updating the parameters based on these probabilities).
- Latent Variables: GMM treats the component membership of each data point as a hidden (latent) variable. The goal is to estimate the distribution of these latent variables and the parameters of the underlying Gaussian distributions.
- Clustering: Each data point is assigned to a Gaussian component based on the likelihood of its occurrence within that component.

In the context of anomaly detection:
- GMM can be used for anomaly detection by modeling the normal behavior of the data as a mixture of Gaussian distributions.
- New data points are evaluated based on their likelihood of belonging to the modeled distribution(s).
- Anomalies are detected when the likelihood of a data point under the fitted GMM is low, meaning it doesn't fit well within any of the normal Gaussian clusters.

Key Features:
- Unsupervised learning: No need for labeled data; GMM learns clusters based on the data's distribution.
- Soft clustering: Unlike traditional clustering methods (e.g., k-means), GMM assigns probabilities to data points for each cluster, allowing for a soft assignment to multiple clusters.
- Probabilistic model: Provides a probabilistic interpretation of cluster membership, useful for uncertainty quantification.
- Handles mixed distributions: Can model complex data distributions with multiple peaks (clusters) in the data.

Parameters:
- Number of components (clusters): The number of Gaussian distributions to fit to the data.
- Covariance type: Determines the shape of the Gaussians (e.g., spherical, diagonal, full).
- Convergence criteria: The conditions under which the EM algorithm stops iterating.
- Initialization: Methods for initializing the parameters, such as random initialization or k-means.

Applications:
- Clustering: Identifying groups or clusters within data, especially when clusters have different shapes and sizes.
- Anomaly detection: Identifying data points that have low likelihood of belonging to any of the modeled clusters.
- Density estimation: Modeling the underlying distribution of data for density estimation or data generation.
- Image segmentation and speech recognition.

Limitations:
- GMM assumes that the data can be modeled by a mixture of Gaussian distributions, which may not always be true for complex or non-Gaussian data.
- GMM can be sensitive to initialization and may require careful tuning of the number of components and other hyperparameters.
"""

from sklearn.mixture import GaussianMixture

from src.models.base_model import UnsupervisedModel


class GMMModel(UnsupervisedModel):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=self.n_components)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)