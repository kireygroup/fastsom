"""
This module contains the base SOM module.
"""
from typing import Tuple, Callable
from functools import partial
import torch
from torch import Tensor
import torch.nn.functional as F
from fastai.torch_core import Module

from .decorators import timeit
from ..core import index_tensor, expanded_op, ifnone 

__all__ = [
    "Som",
    "neigh_gauss",
    "neigh_square",
    "map_diff_standard",
    "map_diff_toroidal",
]


SomSize2D = Tuple[int, int, int]


def squared_dist(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).pow(2).sum(-1).sqrt()


def neigh_gauss(v: Tensor, sigma: Tensor) -> Tensor:
    return torch.exp(torch.neg(v / sigma))


def neigh_square(v: Tensor, sigma: Tensor) -> Tensor:
    return torch.exp(torch.neg(v.pow(2) / sigma))


def map_diff_standard(exp_bmus: Tensor, exp_weights: Tensor) -> Tensor:
    "Calculates the difference between the expanded versions of the batch BMU indices and `weights`."
    return exp_bmus - exp_weights


def map_diff_toroidal(exp_bmus: Tensor, exp_weights: Tensor, map_size: SomSize2D = None) -> Tensor:
    "Calculates the toroidal difference between the expanded versions of the batch BMU indices and `weights`."
    ms = torch.tensor(map_size[:-1])
    # Calculate diff between x coordinate and y coordinate
    dx = (exp_bmus[..., 0] - exp_weights[..., 0]).abs().unsqueeze(-1)
    dy = (exp_bmus[..., 1] - exp_weights[..., 1]).abs().unsqueeze(-1)

    # Calculate single coordinate toroidal diffs using map size
    dx = torch.cat([dx, ms[0] - dx], dim=-1).min(dim=-1, keepdim=True)[0]
    dy = torch.cat([dy, ms[1] - dy], dim=-1).min(dim=-1, keepdim=True)[0]

    # Aggregate back again
    return torch.cat([dx, dy], dim=-1)


class Som(Module):
    """
    Self-Organizing Map implementation with a Fastai-like code organization.\n
    Uses linear decay for `alpha` and `sigma` parameters,
    gaussian neighborhood function and batchwise weight update.\n
    Uses PyTorch's `pairwise_distance` to run BMU calculations on the GPU.
    """

    def __init__(
        self,
        size: SomSize2D,
        rec: list = None,
        alpha=0.003,
        dist_fn: Callable = squared_dist,
        neigh_fn: Callable = neigh_gauss,
        map_diff_fn: Callable = map_diff_toroidal,
    ) -> None:
        self.size = size
        self.rec = rec
        self.lr = torch.tensor([1.0])
        self.alpha = torch.tensor([alpha])
        self.sigma = torch.tensor([max(size[:-1]) / 2.0])
        self.weights = torch.randn(size)
        self.dist_fn, self.neigh_fn, self.map_diff_fn = dist_fn, neigh_fn, map_diff_fn
        self.indices = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.after_weight_init()

        if self.map_diff_fn.__name__ == 'map_diff_toroidal':
            self.map_diff_fn = partial(self.map_diff_fn, map_size=self.size)

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
        distances = self.distance(x, self.weights.view(-1, x.shape[-1]))

        # Retrieve the tensor of BMUs for each element in batch
        bmu_indices = self.find_bmus(distances)

        # If in training, save outputs for backward step
        if self.training:
            self._diffs = self.diff(x, self.weights.view(-1, x.shape[-1])).view(x.shape[0], self.size[0], self.size[1], x.shape[-1])
            self._bmus = bmu_indices

        return bmu_indices

    def backward(self, debug: bool = False) -> None:
        """
        Tensors:\n
        `weights`                           : neuron map\n
        `indices`                           : indices of each map neuron\n
        `map_position_diff`                 : distance of each input value from each map neuron\n
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
        `map_position_diff`                 : [B, 2]\n
        `bmu_distances`                     : [B, rows, cols]\n
        `neighborhood_mult`                 : [B, rows, cols]\n
        `weights`                           : [rows, cols, N]\n
        `bmu_indices.view(...)`             : [B, 1, 1, 2]\n
        `indices.unsqueeze(0).repeat(...)`  : [B, rows, cols, 2]\n
        `delta`                             : [B, rows, cols]
        """
        # Retrieve the current batch outputs and batch size
        bmu_indices, elementwise_diffs = self.get_prev_batch_output()
        batch_size = bmu_indices.shape[0]

        # Calculate the distance between the BMU and each node of the network
        # First, create a tensor of indices of the same size as the weights map
        if self.indices is None:
            self.indices = self._to_device(index_tensor(self.size[:-1]))

        # Calculate the index-based difference from the bmu indices for each neuron
        map_position_diff = self.map_diff_fn(bmu_indices.view(batch_size, 1, 1, len(self.size)-1), self.indices.unsqueeze(0).repeat(batch_size, 1, 1, 1))

        # Then, calculate the euclidean distance of the BMU position with the other nodes
        # We also multiply it by `sigma` for scaling
        bmu_distances = (map_position_diff).pow(2).sum(-1).float().sqrt()

        # Let's use the distances to evaluate the neighborhood multiplier:
        neighborhood_mult = self.neigh_fn(bmu_distances, self.sigma)
        delta = neighborhood_mult.unsqueeze(-1) * self.alpha * elementwise_diffs

        if debug:
            print(f'Decay: {self.lr}')
            print(f'Alpha: {self.alpha}')
            print(f'Sigma: {self.sigma}')
            print(f'Max neighborhood mult (should be 1.0): {neighborhood_mult.max()}')
            print(f'Min neighborhood mult (should be close to 0): {neighborhood_mult.min()}')
            print(f'Max `forward` diff: {elementwise_diffs.max()}')
            print(f'Max update factor: {a.max()}')

        # Update weights (divide by batch size to avoid exploding weights)
        self.weights += (delta.sum(dim=0) / batch_size)

        if self.rec is not None and len(self.rec) < 5:
            self.rec.append(dict(delta=delta.cpu(), n=neighborhood_mult.cpu(), ed=elementwise_diffs.cpu()))
 
    def find_bmus(self, distances: Tensor) -> Tensor:
        """
        Finds the BMUs for a batch of distances.\n
        Input:\n
        `distances`:  [B, rows*cols]\n
        \n
        Output:\n
        `bmus`:       [B, 2]
        """
        # Retrieve size from input
        batch_size = distances.shape[0]
        # Calculate the argmin of the tensor
        min_idx = distances.argmin(-1)
        # Stack the argmin 2D indices
        bmus = torch.stack([min_idx / self.size[1], min_idx % self.size[1]], dim=1)

        return self._to_device(bmus)

    def distance(self, a: Tensor, b: Tensor) -> Tensor:
        "Calculates the distance between `a` and `b`, expanding their sizes."
        return expanded_op(a, b, self.dist_fn, interleave=True, device=self.device)

    def diff(self, a: Tensor, b: Tensor) -> Tensor:
        "Calculates the difference between `a` and `b`, expanding their sizes."
        return expanded_op(a, b, lambda a, b: a - b, interleave=True, device=self.device)

    def get_prev_batch_output(self):
        "Retrieves the output of the previous batch in training"
        if self._diffs is not None and self._bmus is not None:
            return self._bmus, self._diffs
        raise RuntimeError(f'`{self.__class__.__name__}.get_prev_batch_output` should only be called during training')
    
    def after_weight_init(self):
        self.to_device()

    def parameters(self, recurse=True):
        "Returns an iterator over module parameters."
        return iter([self.lr, self.alpha, self.sigma, self.weights])
    
    def to_device(self, device: torch.device = None) -> None:
        "Moves params and weights to the appropriate device."
        self.weights = self._to_device(self.weights, device=device)
        self.alpha = self._to_device(self.alpha, device=device)
        self.sigma = self._to_device(self.sigma, device=device)
        self.lr = self._to_device(self.lr, device=device)

    def _to_device(self, a: Tensor, device: torch.device = None) -> Tensor:
        "Moves a tensor to the appropriate device"
        return a.to(device=ifnone(device, self.device))

    def __repr__(self):
        return f'{self.__class__.__name__}(\n\
            size={self.size[:-1]}, neuron_size={self.size[-1]}, alpha={self.alpha}, sigma={self.sigma}),\n\
            dist_fn={self.dist_fn.__name__}, neigh_fn={self.neigh_fn.__name__}, map_diff_fn={self.map_diff_fn})'
