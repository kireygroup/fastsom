import torch
import torch.nn.functional as F

from functools import reduce
from typing import Callable, Union, Tuple, List


__all__ = [
    "pnorm",
    "pdist",
    "pcosdist",
    "manhattan_dist",
    "grouped_distance",
]


def pcosdist(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates cosine distance of order `p` between `a` and `b`.

    Parameters
    ----------
    a : torch.Tensor
        The first tensor
    b : torch.Tensor
        The second tensor
    p : int default=2
        The order.
    """
    return -F.cosine_similarity(a, b, dim=-1)


def pnorm(a: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates the norm of order `p` of tensor `a`.

    Parameters
    ----------
    a : torch.Tensor
        The input tensor
    p : int default=2
        The order.
    """
    return a.abs().pow(p).sum(-1).pow(1 / p)


def pdist(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates the distance of order `p` between `a` and `b`.
    Assumes tensor shapes are compatible.

    Parameters
    ----------
    a : torch.Tensor
        The first tensor
    b : torch.Tensor
        The second tensor
    p : int default=2
        The order.
    """
    return pnorm(a - b, p=p)


def manhattan_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Manhattan distance (order 1 p-distance) between `a` and `b`.
    Assumes tensor shapes are compatible.

    Parameters
    ----------
    a : torch.Tensor
        The first tensor
    b : torch.Tensor
        The second tensor
    p : int default=2
        The order.
    """
    return pdist(a, b, p=1)


def grouped_distance(a: torch.Tensor, b: torch.Tensor, dist_fn: Callable, n_groups: Union[int, List[Tuple[int, int]]]) -> torch.Tensor:
    """
    Divides both `a` and `b` into smaller groups, then calls `dist_fn` on each pair of groups.
    Useful when computing distances over groups of embeddings, for example.

    Parameters
    ----------
    a : torch.Tensor
        The first Tensor
    b : torch.Tensor
        The second Tensor
    dist_fn : Callable
        The actual distance function
    groups : int
        The number of groups of equal size to be used.
    """
    # Split each tensor into `n_groups` chunks in the feature dimension
    zipped_chunks = zip(torch.chunk(a, n_groups, dim=-1), torch.chunk(b, n_groups, dim=-1))
    # Calculate distance for each pair, then concatenate back together
    dists = [dist_fn(a_group, b_group) for a_group, b_group in zipped_chunks]
    # Average over groups
    return reduce(lambda a, b: a+b, dists) / n_groups
