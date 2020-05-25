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
    "split_distance",
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
        The second tensorn_groups=3
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
    if isinstance(n_groups, int):
        # Split each tensor into `n_groups` chunks in the feature dimension
        zipped_chunks = zip(torch.chunk(a, n_groups, dim=-1), torch.chunk(b, n_groups, dim=-1))
    else:
        # takes a tensor and a tuple, and indexes the tensor with start:end on last (feature) dimension
        def mapper_fn(tens: torch.Tensor): return lambda group: tens[..., group[0]:group[1]]
        zipped_chunks = zip(map(mapper_fn(a), n_groups), map(mapper_fn(b), n_groups))
    # Calculate distance for each pair, then concatenate back together
    dists = [dist_fn(a_group, b_group) for a_group, b_group in zipped_chunks]
    # Average over groups
    return reduce(lambda a, b: a+b, dists) / (a.size(-1) / n_groups)


def split_distance(a: torch.Tensor, b: torch.Tensor, dist_fn_1: Callable, dist_fn_2: Callable, split_idx: int) -> torch.Tensor:
    """
    Splits both `a` and `b` in two parts at `split_idx`, then applies two
    different distance functions for each part.

    This is useful on a mixed dataset containing embeddings and continous values.

    Parameters
    ----------
    a : torch.Tensor
        The first Tensor
    b : torch.Tensor
        The second Tensor
    dist_fn_1 : Callable
        The distance function for the first chunks
    dist_fn_2 : Callable
        The distance function for the second chunks
    split_idx : int
        The split point for `a` and `b`
    """
    a1, a2 = a[..., :split_idx], a[..., split_idx:]
    b1, b2 = b[..., :split_idx], b[..., split_idx:]
    dist_1 = dist_fn_1(a1, b1)
    dist_2 = dist_fn_2(a2, b2)

    def normalize(a: torch.Tensor) -> torch.Tensor:
        return (a - a.min()) / (a.max() - a.min())
    dist_1 = normalize(dist_1)
    dist_2 = normalize(dist_2)
    return dist_1 + dist_2
