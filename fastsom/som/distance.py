import torch


__all__ = [
    "pcosim",
    "pnorm",
    "pdist",
    "manhattan_dist",
]


def pcosim(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates the cosine similarity of order `p` between `a` and `b`.
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
    return (a * b).sum(-1) / pnorm(a, p=p) * pnorm(b, p=p)


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
