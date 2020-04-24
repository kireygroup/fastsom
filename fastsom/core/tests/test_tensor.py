import torch
import unittest
import numpy as np

from typing import Tuple
from nose2.tools import params
from fastsom import index_tensor, idxs_1d_to_2d, idxs_2d_to_1d


class IndexTensorTest(unittest.TestCase):

    @params(
        ((2, 2), torch.tensor([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])),
        ((2, 2, 2), torch.tensor([[[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]], [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]]]))
    )
    def test_is_index_tensor(self, size: Tuple, expected: torch.Tensor):
        assert((expected == index_tensor(size)).all())


class IdxsConversionTest(unittest.TestCase):
    
    @params(
        (
            np.array([[0, 1], [1, 1], [2, 1], [3, 1]]),
            4,
            torch.tensor([1, 5, 9, 13])
        ),
        (
            np.array([[1, 49], [45, 23]]),
            50,
            torch.tensor([99, 2273])
        )
    )
    def test_2d_idxs_to_1d(self, idxs: np.ndarray, row_size: int, expected: torch.Tensor):
        assert((expected == idxs_2d_to_1d(idxs, row_size)).all())
    
        
    @params(
        (
            np.array([1, 5, 9, 13]),
            4,
            torch.tensor([[0, 1], [1, 1], [2, 1], [3, 1]])
        ),
        (
            np.array([99, 2273]),
            50,
            torch.tensor([[1, 49], [45, 23]])
        )
    )
    def test_1d_idxs_to_2d(self, idxs: np.ndarray, col_size: int, expected: torch.Tensor):
        assert((expected == idxs_1d_to_2d(idxs, col_size)).all())