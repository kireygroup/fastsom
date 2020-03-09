"""
"""
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Tuple, Collection, Callable
from fastai.torch_core import Module
from functools import reduce

from ..core import index_tensor


__all__ = [
    "SomSize2D",
    "pairwise_distance",
    "Som",
]


SomSize2D = Tuple[int, int, int]


def pairwise_distance(a: Tensor, b: Tensor) -> Tensor:
    "Calculates the pairwise distance between `a` and `b`."
    return (a - b).pow(2).sum(-1).sqrt()


class Som(Module):
    """
    Self-Organizing Map implementation with a Fastai-like code organization.\n
    Uses linear decay for `alpha` and `sigma` parameters,
    gaussian neighborhood function and batchwise weight update.\n
    Uses PyTorch's `pairwise_distance` to run BMU calculations on the GPU.
    """

    def __init__(self, size: SomSize2D, alpha=0.003, dist_fn=pairwise_distance) -> None:
        self.size = size
        self.lr = None
        self.alpha = alpha
        self.sigma = max(size[:-1]) / 2.0
        self.weights = torch.randn(size)
        self.training = False
        self.dist_fn = dist_fn
        self.indices = None
        self.use_cuda = torch.cuda.is_available()

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
        # Retrieve tensor dimensions
        batch_size, in_size = x.shape
        w_size = reduce(lambda a, b: a * b, w.shape[:-1])

        # Allocate GPU space to store results
        d = self._to_device(torch.zeros(batch_size, w_size))

        # Reshape X and W to enable distance calculation
        a = self._to_device(x.repeat(w_size, 1))
        b = self._to_device(w.view(-1, in_size).repeat(batch_size, 1))

        # Calculate the pairwise distance
        d = self.dist_fn(a, b)

        # Cleanup GPU space
        del a
        del b
        if self.use_cuda:
            torch.cuda.empty_cache()

        # Return the reshaped distance matrix
        return d.view(batch_size, *(w.shape[:-1]))

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

    def diff_old(self, x: Tensor, w: Tensor) -> Tensor:
        "Calculates the difference between `x` and `w`, matching their sizes."
        w_size = reduce(lambda a, b: a * b, w.shape[:-1])
        return (x.repeat(w_size, 1) - (w.view(w_size, -1).repeat(x.shape[0], *(1 for _ in range(len(x.shape)-1))))).view(-1, *w.shape)

    def diff(self, x: Tensor, w: Tensor) -> Tensor:
        "Calculates the difference between `x` and `w`, matching their sizes."
        # Retrieve tensor dimensions
        batch_size, in_size = x.shape
        w_size = reduce(lambda a, b: a * b, w.shape[:-1])

        d = self._to_device(torch.zeros(batch_size, w_size))

        # Reshape X and W to allow broadcasting
        a = self._to_device(x.repeat(w_size, 1))
        b = self._to_device(
            w.view(-1, in_size).repeat(batch_size, *(1 for _ in range(len(x.shape)-1))))

        # Calculate the pairwise distance
        d = a - b

        # Cleanup GPU space
        del a
        del b
        if self.use_cuda:
            torch.cuda.empty_cache()

        # Return the reshaped distance matrix
        return d.view(-1, *w.shape)

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluates the distances between `x` and the map weights;
        then updates the map.\n

        `distances`     : [B, rows, cols]\n
        `bmu_indices`   : [B, 2]\n
        """
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
        `x`                                 : batch data\n
        `weights`                           : neuron map\n
        `indices`                           : indices of each map neuron\n
        `distances`                         : distance of each input value from each map neuron\n
        `bmu_indices`                       : index of the BMU for each batch
        `diff`                              : difference between each neuron's position in the map and the BMU of each batch
        `bmu_distances`                     : euclidean form of `diff`
        `neighborhood_mult`                 : scaled gaussian multiplier for each neuron in the map and each batch\n
        \n
        `bmu_indices.view(...)`             : reshape of `bmu_indices` for broadcasting over `indices`\n
        `indices.unsqueeze(0).repeat(...)`  : reshape of `indices` for broadcasting over `bmu_indices`\n
        `neighborhood_mult * alpha * dist`  : weight delta for each neuron for each batch\n

        Shapes:\n
        `indices`                           : [rows, cols, 2]\n
        `distances`                         : [B, rows, cols]\n
        `diff`                              : [B, 2]
        `bmu_distances`                     : [B, rows, cols]
        `neighborhood_mult`                 : [B, rows, cols]\n
        `weights`                           : [rows, cols, N]\n
        `x`                                 : [B, N]\n
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

        # Let's use the distances to evaluate the Gaussian neighborhood multiplier:
        neighborhood_mult = torch.exp(torch.neg(bmu_distances / (self.sigma ** 2)))

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

    def _to_device(self, a: Tensor) -> Tensor:
        "Moves a tensor to the appropriate device"
        return a.cuda() if self.use_cuda else a.cpu()
