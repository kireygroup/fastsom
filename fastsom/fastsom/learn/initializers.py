"""
Initializers are used to define
initial map weights for Self-Organizing Maps.
"""

from typing import Tuple
from functools import reduce
from torch import Tensor
from kmeans_pytorch import kmeans as _kmeans
import torch

__all__ = [
    "som_initializers",
    "SomInitializer",
    "KMeansInitializer",
    "RandomInitializer",
]


class SomInitializer():
    """SOM weight initializer base class."""

    def __call__(self, x: Tensor, size: Tuple, **kwargs) -> Tensor:
        raise NotImplementedError


class KMeansInitializer(SomInitializer):
    """Initializes SOM weights using KMeans."""

    def __init__(self, distance: str = 'euclidean'):
        self.distance = distance
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, x: Tensor, size: Tuple, **kwargs):
        """
        Performs K-Means over an input dataset.\n
        Params:\n
        `x`         : the input Tensor\n
        `size`      : the SOM size\n
        """
        k = reduce(lambda acc, x: x * acc, size[:-1])
        # Run the KMeans algorithm over the input
        _, cluster_centers = _kmeans(X=x, num_clusters=k, distance=self.distance, device=self.device)
        # Reshape it to fit the SOM size
        return cluster_centers.view(*(size[:-1]), -1)


class RandomInitializer(SomInitializer):
    """Initializes SOM weights randomly."""

    def __call__(self, x: Tensor, size: Tuple, **kwargs):
        x_min = x.min(dim=0)[0]
        x_max = x.max(dim=0)[0]
        return (x_max - x_min) * torch.zeros(size).uniform_(0, 1) - x_min


som_initializers = {
    'random': RandomInitializer(),
    'kmeans': KMeansInitializer(),
    'kmeans_cosine': KMeansInitializer(distance='cosine'),
}
