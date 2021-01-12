"""
Initializers are used to define
initial map weights for Self-Organizing Maps.
"""

import torch
from kmeans_pytorch import kmeans as _kmeans
from torch import Tensor

__all__ = [
    "som_initializers",
    "WeightsInitializer",
    "KMeansInitializer",
    "RandomInitializer",
]


class WeightsInitializer:
    """Weight initializer base class."""

    def __call__(self, x: Tensor, k: int, **kwargs) -> Tensor:
        raise NotImplementedError


class KMeansInitializer(WeightsInitializer):
    """
    Initializes weights using K-Means.

    Parameters
    ----------
    distance : str default='euclidean'
        The type of distance function to be used. Can be `euclidean` or `cosine`.
    """

    def __init__(self, distance: str = "euclidean"):
        self.distance = distance
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def __call__(self, x: Tensor, k: int, **kwargs) -> Tensor:
        """
        Calculates K-Means on `x` and returns the centroids.

        Parameters
        ----------
        x : Tensor
            The input data.
        k: int
            The number of weights to be returned.
        """
        # Run the KMeans algorithm over the input
        _, cluster_centers = _kmeans(X=x, num_clusters=k, distance=self.distance, device=self.device)
        # Reshape it to fit the SOM size
        return cluster_centers


class RandomInitializer(WeightsInitializer):
    """
    Initializes weights randomly.
    """

    def __call__(self, x: Tensor, k: int, **kwargs) -> Tensor:
        """
        Uniform random  weight initialization.

        Parameters
        ----------
        x: Tensor
            The input data.
        k: int
            The number of weights to be returned.
        """
        x_min = x.min(dim=0)[0]
        x_max = x.max(dim=0)[0]
        return (x_max - x_min) * torch.zeros(k, x.shape[-1]).uniform_(0, 1).to(x.device) - x_min


som_initializers = {
    "random"       : RandomInitializer(),
    "kmeans"       : KMeansInitializer(distance="euclidean"),
    "kmeans_cosine": KMeansInitializer(distance="cosine"),
}
