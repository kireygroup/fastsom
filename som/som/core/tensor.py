import torch

from torch import Tensor as Tensor
from typing import Tuple as Tuple


def index_tensor(size: Tuple) -> Tensor:
    """
    Returns an index tensor of size `size`, 
    where each element contains its own index.

    """
    return torch.ones(*size).nonzero().view(*size, -1)
