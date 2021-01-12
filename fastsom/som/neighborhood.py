from typing import Tuple

import torch

from .distance import pnorm

__all__ = [
    "neigh_gauss",
    "neigh_square",
    "neigh_rhomb",
    "neigh_diff_standard",
    "neigh_diff_toroidal",
]


def neigh_gauss(position_diff: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Gaussian neighborhood scaling function based on center-wise \
        diff `position_diff` and radius `sigma`.

    Parameters
    ----------
    position_diff : torch.Tensor
        The positional difference around some center.
    sigma : torch.Tensor
        The scaling radius.
    """
    v = pnorm(position_diff, p=2)
    return torch.exp(torch.neg(v.pow(2) / sigma.pow(2)))


def neigh_rhomb(position_diff: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Diamond-shaped neighborhood function based on center-wise diff `position_diff` and radius `sigma`.
    Note: Manhattan distance should be used with this function.

    Parameters
    ----------
    position_diff : torch.Tensor
        The positional difference around some center.
    sigma : torch.Tensor
        The scaling radius.
    """
    v = pnorm(position_diff, p=1)
    return torch.exp(torch.neg(torch.sqrt(v) / sigma))


def neigh_square(position_diff: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Square-shaped neighborhood scaling function based on center-wise diff `position_diff` and radius `sigma`.

    Parameters
    ----------
    position_diff : torch.Tensor
        The positional difference around some center.
    sigma : torch.Tensor
        The scaling radius.
    """
    v = (position_diff).abs().max(-1)[0]
    # v = pnorm(position_diff, p=2)
    return torch.exp(torch.neg(v.sqrt() / sigma))


def neigh_diff_standard(bmus: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Positional difference function.
    Computes index difference between Best Matching Units (`bmus`) and `positions`, where `positions` 
    are indices of elements inside the SOM grid.

    Parameters
    ----------
    bmus : torch.Tensor
        The list of Best Matching Units (2D indices)
    positions : torch.Tensor
        The 2D tensor of grid element indices 
    """
    return bmus - positions


def neigh_diff_toroidal(
    bmus: torch.Tensor, positions: torch.Tensor, map_size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Positional difference function.
    Computes toroidal (wraparound) difference between Best Matching Units (`bmus`) and `positions`, where `positions` 
    are indices of elements inside the SOM grid.

    Parameters
    ----------
    bmus : torch.Tensor
        The list of Best Matching Units (2D indices)
    positions : torch.Tensor
        The 2D tensor of grid element indices
    map_size : Tuple[int, int] = None
        The SOM size. Used to pick shortest distance between indices.
    """
    ms = torch.tensor(map_size)
    # Calculate diff between x coordinate and y coordinate
    dx = (bmus[..., 0] - positions[..., 0]).abs().unsqueeze(-1)
    dy = (bmus[..., 1] - positions[..., 1]).abs().unsqueeze(-1)

    # Calculate single coordinate toroidal diffs using map size
    dx = torch.cat([dx, ms[0] - dx], dim=-1).min(dim=-1, keepdim=True)[0]
    dy = torch.cat([dy, ms[1] - dy], dim=-1).min(dim=-1, keepdim=True)[0]

    # Aggregate back again
    return torch.cat([dx, dy], dim=-1)
