import unittest
import torch


class DistanceTest(unittest.TestCase):

    def test_distance(self):
        vars = 5
        cols = 4
        rows = 4
        a = torch.tensor([
            [0. for _ in range(vars)],
            [1. for _ in range(vars)],
            [2. for _ in range(vars)],
            [3. for _ in range(vars)],
            [4. for _ in range(vars)]
        ])
        b = torch.tensor([
            [a, a, a, a],
            [a, a, a, a],
            [a, a, a, a],
            [a, a, a, a]
        ])
