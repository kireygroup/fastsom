"""
This module contains core tensor operations.
"""


import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Callable

__all__ = [
    "index_tensor",
    "expanded_op",
    "idxs_2d_to_1d",
    "idxs_1d_to_2d",
]

def index_tensor(size: Tuple) -> Tensor:
    """
    Returns an index tensor of size `size`, 
    where each element contains its own index.

    """
    return torch.ones(*size).nonzero().view(*size, -1)


def expanded_op(a: Tensor, b: Tensor, fn: Callable, interleave: bool = False, device=torch.device("cuda")) -> Tensor:
    "Expands `a` and `b` to make sure their shapes match; then calls `fn`."
    N, M = a.shape[0], b.shape[0]

    # Allocate device space to store results
    res = torch.zeros(N, M).to(device=device)

    # Reshape A and B to enable distance calculation
    # # Optionally interleaves repeat method
    # _a = a.repeat_interleave(M, dim=0) if interleave else a.repeat(M, 1).to(device=device)
    # _b = b.view(-1, D).repeat(N, 1).to(device=device)

    _a = a.view(N, 1, -1).to(device=device)
    _b = b.expand(N, -1, -1).to(device=device)

    # Invoke the function over the two Tensors
    res = fn(_a, _b)

    # Cleanup device space
    use_cuda = _a.is_cuda
    del _a
    del _b
    if use_cuda:
        torch.cuda.empty_cache()

    return res


def idxs_2d_to_1d(idxs: np.ndarray, row_size: int) -> list:
    "Turns 2D indices to 1D"
    return torch.tensor([el[0] * row_size + el[1] for el in idxs])


def idxs_1d_to_2d(idxs: np.ndarray, col_size: int) -> list:
    "Turns 1D indices to 2D"
    return torch.tensor([[el // col_size, el % col_size] for el in idxs])