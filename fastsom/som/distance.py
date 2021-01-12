from functools import partial, reduce
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

__all__ = [
    "pnorm",
    "pdist",
    "pcosdist",
    "manhattan_dist",
    "grouped_distance",
    "split_distance",
    "MixedEmbeddingDistance",
    "MixedCategoricalDistance",
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


def grouped_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    dist_fn: Callable,
    n_groups: Union[int, List[int]],
) -> torch.Tensor:
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
    # Split `a` and `b` into chunks of equal size or groups of specified size
    split_fn = torch.chunk if isinstance(n_groups, int) else torch.split
    # Zip together pairs of chunks
    zipped_chunks = zip(split_fn(a, n_groups, dim=-1), split_fn(b, n_groups, dim=-1))
    # Calculate distance for each pair, then concatenate back together
    dists = [dist_fn(a_group, b_group) for a_group, b_group in zipped_chunks]
    # Average over groups
    n_chunks = n_groups if isinstance(n_groups, int) else len(n_groups)
    return reduce(lambda a, b: a + b, dists) / (a.size(-1) / n_chunks)


def split_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    dist_fn_1: Callable,
    dist_fn_2: Callable,
    split_idx: int,
) -> torch.Tensor:
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
    a1, a2 = torch.split(a, [split_idx, a.size(-1) - split_idx], dim=-1)
    b1, b2 = torch.split(b, [split_idx, b.size(-1) - split_idx], dim=-1)
    dist_1 = dist_fn_1(a1, b1)
    dist_2 = dist_fn_2(a2, b2)

    def normalize(a: torch.Tensor) -> torch.Tensor:
        return (a - a.min()) / (a.max() - a.min())

    dist_1 = normalize(dist_1)
    dist_2 = normalize(dist_2)
    return (dist_1 * a1.shape[-1] / a.shape[-1]) + (dist_2 * a2.shape[-1] / a.shape[-1])


class MixedEmbeddingDistance(Callable):
    """
    Splits input tensor into encoded-categorical and continuous features;
    then, calculates embedding distance on embeddings and euclidean distance
    on continuous features and returns the average.
    """

    def __init__(self, emb_szs: Dict[str, int]):
        self.emb_szs = emb_szs
        self.split_idx = sum(self.emb_szs.values())
        # Setup embedding and continuous distances
        self._embs_dist = partial(grouped_distance, dist_fn=pcosdist, n_groups=list(self.emb_szs.values()))
        self._cont_dist = pdist
        # Setup split distance with the above parameters
        self._dist_fn = partial(
            split_distance,
            split_idx=self.split_idx,
            dist_fn_1=self._embs_dist,
            dist_fn_2=self._cont_dist,
        )

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Check if `a` and `b` are splittable
        if a.shape[-1] + b.shape[-1] <= (2 * self.split_idx):
            return self._embs_dist(a, b)
        else:
            return self._dist_fn(a, b)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @property
    def __name__(self):
        return self.__class__.__name__


def mismatch_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Returns 0 where a == b and 1 where a != b.
    """
    return (a != b).long().float().sum(-1)


class MixedCategoricalDistance(Callable):
    """
    Mixed distance for categorical & continuous features.
    Uses mismatch distance for categoricals and p-distance of order 2 for continuous.
    """
    def __init__(self, split_idx: Optional[int] = None):
        self.split_idx = split_idx
        self._cats_dist = mismatch_distance
        self._cont_dist = pdist
        self._dist_fn = partial(
            split_distance,
            split_idx=self.split_idx,
            dist_fn_1=self._cats_dist,
            dist_fn_2=self._cont_dist,
        )

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._dist_fn(a, b)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @property
    def __name__(self):
        return f"{self.__class__.__name__}"
