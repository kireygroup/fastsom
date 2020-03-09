"""
This module contains core tensor operations.
"""

from typing import Tuple, Callable

import torch
from torch import Tensor


def index_tensor(size: Tuple) -> Tensor:
    """
    Returns an index tensor of size `size`, 
    where each element contains its own index.

    """
    return torch.ones(*size).nonzero().view(*size, -1)


def expanded_op(a: Tensor, b: Tensor, fn: Callable, interleave: bool = False, device=torch.device("cuda")) -> Tensor:
    "Expands `a` and `b` to make sure their shapes match; then calls `fn`."
    N, D = a.shape
    M, _ = b.shape

    # Allocate device space to store results
    res = torch.zeros(N, M).to(device=device)

    # Reshape A and B to enable distance calculation
    # Optionally interleaves repeat method
    _a = a.repeat_interleave(M, dim=0) if interleave else a.repeat(M, 1).to(device=device)
    _b = b.view(-1, D).repeat(N, 1).to(device=device)

    # Invoke the function over the two Tensors
    res = fn(_a, _b)

    # Cleanup device space
    use_cuda = _a.is_cuda
    del _a
    del _b
    if use_cuda:
        torch.cuda.empty_cache()

    # Return the result
    return res
