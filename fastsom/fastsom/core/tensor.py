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


def index_tensor(*size) -> Tensor:
    """
    Returns an index tensor of size `size`,
    where each element contains its own index.

    Parameters
    ----------
    size : Tuple
        The size of the matrix indexed by the index tensor.
        Note that the return tensor size will be `(*size, 2)`.

    Examples
    --------
    >>> index_tensor(2, 2)
    tensor([[[0, 0],
         [0, 1]],
         [[1, 0],
          [1, 1]]])
    """
    if isinstance(size[0], tuple):
        size = size[0]
    return torch.ones(*size).nonzero().view(*size, -1)


def expanded_op(a: Tensor, b: Tensor, fn: Callable, device=torch.device("cuda")) -> Tensor:
    """
    Expands tensors `a` and `b` to ensure operation compatibility; then calls `fn(a, b)`.

    Parameters
    ----------
    a : Tensor
        The first tensor.
    b : Tensor
        The second tensor.
    fn : Callable
        The function to be applied on expanded `a` and `b`.
    device : torch.device default=torch.device('cuda')

    Examples
    --------
    >>> a = torch.randn(5, 2)
    >>> b = torch.randn(12, 2)
    >>> c = expanded_op(a, b, lambda a, b: a + b)
    >>> c.shape
    torch.Size([5, 12, 2])
    """
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
    """
    Transforms an `np.ndarray` of indices from 2D to 1D by using `row_size`.

    Parameters
    ----------
    idxs : np.ndarray
        The matrix of 2D indices.
    row_size : int
        The 2D matrix row size.

    Examples
    --------
    >>> idxs = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    >>> idxs_1d_to_2d(idxs, 4)
    tensor([ 1,  5,  9, 13])
    """
    return torch.tensor([el[0] * row_size + el[1] for el in idxs])


def idxs_1d_to_2d(idxs: np.ndarray, col_size: int) -> list:
    """
    Transforms an `np.ndarray` of indices from 1D to 2D by using `col_size`.

    Parameters
    ----------
    idxs : np.ndarray
        The matrix of 1D indices.
    col_size : int
        The 2D matrix col size.

    Examples
    --------
    >>> idxs = np.array([ 1,  5,  9, 13])
    >>> idxs_1d_to_2d(idxs, 4)
    tensor([[0, 1],
        [1, 1],
        [2, 1],
        [3, 1]])
    """
    return torch.tensor([[el // col_size, el % col_size] for el in idxs])
