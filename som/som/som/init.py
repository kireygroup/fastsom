"""
Initializers are used to define 
initial map weights for Self-Organizing Maps.
"""
import torch

from torch import Tensor
from kmeans_pytorch import kmeans as _kmeans
from functools import partial as _partial, reduce as _reduce
from typing import Tuple, Callable

from ..core import compose


class SomInitializer():
    "SOM weight initializer base class."

    def __call__(self, x: Tensor, size: Tuple, **kwargs) -> Tensor:
        raise NotImplementedError


class KMeansInitializer(SomInitializer):
    "Initializes SOM weights using KMeans."

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
        k = _reduce(lambda acc, x: x * acc, size[:-1])
        # Run the KMeans algorithm over the input
        _, cluster_centers = _kmeans(X=x, num_clusters=k, distance=self.distance, device=self.device)
        # Reshape it to fit the SOM size
        return cluster_centers.view(*(size[:-1]), -1)


class RandomInitializer(SomInitializer):
    "Initializes SOM weights randomly."

    def __call__(self, x: Tensor, size: Tuple, **kwargs):
        return torch.randn(size)


"""
SOM Initializers
"""
som_initializers = {
    'kmeans_euclidean': KMeansInitializer(),
    'kmeans_cosine': KMeansInitializer(distance='cosine'),
    'random': RandomInitializer(),
}

__all__ = [
    "som_initializers",
]
