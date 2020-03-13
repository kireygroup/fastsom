"""
This module contains the base SOM module.
"""
from typing import Tuple, Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from fastai.torch_core import Module

from ..core import index_tensor, expanded_op
from .decorators import timeit

__all__ = [
    "SomFast",
]


SomSize2D = Tuple[int, int, int]


def neigh_gauss(v, sigma):
    return torch.exp(torch.neg(v / sigma))


def neigh_square(v, sigma):
    return torch.exp(torch.neg(v.pow(2) / sigma))


class SomFast(Module):
    """
    Self-Organizing Map implementation with a Fastai-like code organization.\n
    Uses linear decay for `alpha` and `sigma` parameters,
    gaussian neighborhood function and batchwise weight update.\n
    Uses PyTorch's `pairwise_distance` to run BMU calculations on the GPU.
    """

    def __init__(self, size: SomSize2D, alpha=0.003, dist_fn: Callable = F.pairwise_distance, neigh_fn: Callable = neigh_gauss) -> None:
        self.size = size
        self.lr = torch.tensor([1.0])
        self.alpha = torch.tensor([alpha])
        self.sigma = torch.tensor([max(size[:-1]) / 2.0])
        self.weights = torch.randn(size)
        self.dist_fn, self.neigh_fn = dist_fn, neigh_fn
        self.indices = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.after_weight_init()

    def after_weight_init(self):
        self.to_device()

    def parameters(self, recurse=True):
        "Returns an iterator over module parameters."
        return iter([self.lr, self.alpha, self.sigma, self.weights])

    def distance(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Calculates the pairwise distance between `x` and `w` by
        expanding them and passing them to `dist_fn`.\n
        \n
        Input\n
        x: [N, S]\n
        w: [rows, cols, S]\n
        \n
        Output\n
        d: [N, rows, cols]\n
        """
        return expanded_op(x, w.view(-1, x.shape[-1]), self.dist_fn, interleave=True, device=self.device).view(-1, *w.shape[:-1])

    def find_bmus(self, distances: Tensor) -> Tensor:
        """
        Finds the BMUs for a batch of distances.\n
        Input:\n
        `distances`:  [B, rows, cols]\n
        \n
        Output:\n
        `bmus`:       [B, 2]
        """
        # Retrieve size from input
        batch_size, _, cols = distances.shape
        # Calculate the argmin of the tensor
        min_idx = distances.view(batch_size, -1).argmin(-1)
        # Stack the argmin 2D indices
        bmus = torch.stack((min_idx / cols, min_idx % cols), dim=1)
        return self._to_device(bmus)

    def diff(self, x: Tensor, w: Tensor) -> Tensor:
        "Calculates the difference between `x` and `w`, expanding their sizes."
        return expanded_op(x, w.view(-1, x.shape[-1]), lambda a, b: a - b, interleave=True, device=self.device).view(-1, *w.shape)

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluates the distances between `x` and the map weights;
        then updates the map.\n

        `distances`     : [B, rows, cols]\n
        `bmu_indices`   : [B, 2]\n
        """
        if x.device != self.device:
            x = self._to_device(x)
        if self.weights.device != self.device:
            self.weights = self._to_device(self.weights)

        # Evaluate distances between each item in `x` and each neuron of the map
        distances = self.distance(x, self.weights)

        # Retrieve the tensor of BMUs for each element in batch
        bmu_indices = self.find_bmus(distances)

        # If in training, save outputs for backward step
        if self.training:
            self._diffs = self.diff(x, self.weights)
            self._bmus = bmu_indices

        return bmu_indices

    def backward(self, debug: bool = False) -> None:
        """
        Tensors:\n
        `weights`                           : neuron map\n
        `indices`                           : indices of each map neuron\n
        `distances`                         : distance of each input value from each map neuron\n
        `bmu_indices`                       : index of the BMU for each batch\n
        `diff`                              : difference between each neuron's position in the map and the BMU of each batch\n
        `bmu_distances`                     : euclidean form of `diff`\n
        `neighborhood_mult`                 : scaled gaussian multiplier for each neuron in the map and each batch\n
        \n
        `bmu_indices.view(...)`             : reshape of `bmu_indices` for broadcasting over `indices`\n
        `indices.unsqueeze(0).repeat(...)`  : reshape of `indices` for broadcasting over `bmu_indices`\n
        `neighborhood_mult * alpha * dist`  : weight delta for each neuron for each batch\n
        \n
        Shapes:\n
        `indices`                           : [rows, cols, 2]\n
        `distances`                         : [B, rows, cols]\n
        `diff`                              : [B, 2]\n
        `bmu_distances`                     : [B, rows, cols]\n
        `neighborhood_mult`                 : [B, rows, cols]\n
        `weights`                           : [rows, cols, N]\n
        `bmu_indices.view(...)`             : [B, 1, 1, 2]\n
        `indices.unsqueeze(0).repeat(...)`  : [B, rows, cols, 2]\n
        `neighborhood_mult * alpha * dist`  : [B, rows, cols]
        """
        # Retrieve the current batch outputs and batch size
        bmu_indices, elementwise_diffs = self.get_prev_batch_output()
        batch_size = bmu_indices.shape[0]

        # Calculate the distance between the BMU and each node of the network
        # First, create a tensor of indices of the same size as the weights map
        if self.indices is None:
            self.indices = self._to_device(index_tensor(self.size[:-1]))

        # Calculate the index-based difference from the bmu indices for each neuron
        diff = bmu_indices.view(batch_size, 1, 1, len(self.size)-1) - self.indices.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Then, calculate the euclidean distance of the BMU position with the other nodes
        # We also multiply it by `sigma` for scaling
        bmu_distances = (diff).pow(2).sum(-1).float().sqrt()

        # Let's use the distances to evaluate the neighborhood multiplier:
        neighborhood_mult = self.neigh_fn(bmu_distances, self.sigma)

        a = neighborhood_mult[..., None] * self.alpha * elementwise_diffs

        if debug:
            print(f'Decay: {self.lr}')
            print(f'Alpha: {self.alpha}')
            print(f'Sigma: {self.sigma}')
            print(f'Max neighborhood mult (should be 1.0): {neighborhood_mult.max()}')
            print(f'Min neighborhood mult (should be close to 0): {neighborhood_mult.min()}')
            print(f'Max `forward` diff: {elementwise_diffs.max()}')
            print(f'Max update factor: {a.max()}')

        # Now we need to evaluate the distance of each neuron with the input
        for delta in a:
            self.weights = self.weights + delta

    def get_prev_batch_output(self):
        "Retrieves the output of the previous batch in training"
        if self._diffs is not None and self._bmus is not None:
            return self._bmus, self._diffs
        raise RuntimeError(f'`{self.__class__.__name__}.get_prev_batch_output` should only be called during training')

    def __repr__(self):
        return f'{self.__class__.__name__}(size={self.size[:-1]}, neuron_size={self.size[-1]}, alpha={self.alpha}, sigma={self.sigma}), dist_fn={self.dist_fn}'

    def to_device(self) -> None:
        "Moves params and weights to the appropriate device."
        self.weights = self._to_device(self.weights)
        self.alpha = self._to_device(self.alpha)
        self.sigma = self._to_device(self.sigma)
        self.lr = self._to_device(self.lr)

    def _to_device(self, a: Tensor) -> Tensor:
        "Moves a tensor to the appropriate device"
        return a.to(device=self.device)
