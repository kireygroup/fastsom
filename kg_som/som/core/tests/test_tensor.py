import torch
import unittest

from nose2.tools import params
from awesome.core import index_tensor


class IndexTensorTest(unittest.TestCase):

    @params(
        ((2, 2), torch.tensor([[[0, 1], [1, 1]], [[1, 0], [1, 1]]])),
        ((2, 2, 2), tensor([[[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]], [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]]]))
    )
    def test_is_index_tensor(self, size: Tuple, expected: torch.Tensor):
        assert(expected == index_tensor(size))
