
from typing import Callable, Tuple

import torch
from fastai.torch_core import Module

from fastsom.core import expanded, index_tensor
from fastsom.core.tensor import idxs_1d_to_2d, idxs_2d_to_1d

from ..log import has_logger
from .distance import pdist
from .neighborhood import neigh_diff_standard, neigh_gauss


class Som(Module):
    """
    Self-Organizing Map module.

    Parameters
    ----------
    size: Tuple[int, int, int],
        The 2D map size: (rows, cols, n_features)
    dist_fn: Callable = pdist
        The distance function used to compute records-to-codebook distances.
    neigh_fn: Callable = neigh_gauss
        The neighbourhood scaling function.
    neigh_diff_fn: Callable = neigh_diff_standard
        The neighbourhood difference function. Can be standard or toroidal.
    """

    def __init__(
        self,
        size: Tuple[int, int, int],
        dist_fn: Callable = pdist,
        neigh_fn: Callable = neigh_gauss,
        neigh_diff_fn: Callable = neigh_diff_standard,
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = torch.tensor(0.3)
        self.sigma = torch.tensor(max(size[:-1]) / 2.0)
        self.weights = torch.randn(self.size)
        self.map_indices = index_tensor(self.size[:-1]).view(-1, 2)
        # customizable functions
        self.dist_fn = dist_fn
        self.neigh_fn = neigh_fn
        self.neigh_diff_fn = neigh_diff_fn
        self._recorder = dict()

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        SOM forward pass.

        Does the following:
         1. Calculate distance between `xb` and `weights`
         2. Find BMU for each element in `xb`
         3. Return BMUs indices

        Parameters
        ----------
        xb : torch.Tensor
            The batch data
        """
        with torch.no_grad():
            self.to_device(device=xb.device)
            n_features = xb.shape[-1]
            # self.logger.debug(f"xb: {xb.shape}, weights: {self.weights.view(-1, n_features).shape}")
            distances = self.distance(xb, self.weights.view(-1, n_features))
            bmus = self.find_bmus(distances)
            # self.logger.debug(f"bmus: {bmus.shape}")
            # save batch data
            self._recorder["xb"] = xb.clone()
            self._recorder["bmus"] = bmus
        return bmus

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculates distance between `a` and `b` using this module's `dist_fn`.

        Parameters
        ----------
        a : torch.Tensor
            The first tensor
        b : torch.Tensor
            The second tensor
        """
        return expanded(a, b, self.dist_fn)

    def backward(self) -> None:
        """
        SOM backward pass.

        Does the following:

         1. Calculate index-distances of codebook elements and bmus
         2. Calculate neighbourhood scaling on index distances
         3. Update weights
        """
        with torch.no_grad():
            xb, bmus = self._recorder["xb"], self._recorder["bmus"]
            batch_size = xb.shape[0]
            n_features = xb.shape[-1]
            # diffs shape: [bs, row_sz, col_sz, n_features]
            elementwise_diffs = expanded(xb, self.weights.view(-1, n_features), lambda a, b: a - b).view(batch_size, self.size[0], self.size[1], n_features)
            # neigh shape: [bs, row_sz, col_zs, 1]
            neighbourhood_mults = self.neighborhood(bmus, self.sigma)
            self.weights += (self.alpha * neighbourhood_mults * elementwise_diffs).sum(0) / batch_size

    def find_bmus(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Find BMU for each batch in `distances`.

        Parameters
        ----------
        distances : torch.Tensor
            The data-to-codebook distances.
        """
        # distances shape: [bs, codebook_size]
        # argmin shape   : [bs, 1]
        min_idxs = distances.argmin(-1)
        # Distances are flattened, so we need to transform 1d indices into 2d map locations
        return idxs_1d_to_2d(min_idxs, self.size[1]).to(distances.device)

    def neighborhood(self, bmus: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates neighborhood multipliers for each BMU in `bmus` using this module's `neigh_diff_fn` and `neigh_fn`.

        Parameters
        ----------
        bmus : torch.Tensor
            The batch BMU indices
        sigma : torch.Tensor
            The neighborhoor radius
        """
        out_shape = (bmus.shape[0], self.size[0], self.size[1], 1)
        index_diff = expanded(bmus, self.map_indices, self.neigh_diff_fn)
        return self.neigh_fn(index_diff, sigma).view(out_shape)

    def parameters(self):
        return iter([self.weights, self.map_indices])

    def to_device(self, device: torch.device = None) -> None:
        """Moves params and weights to the appropriate device."""
        self.weights = self._to_device(self.weights, device=device)
        self.alpha = self._to_device(self.alpha, device=device)
        self.sigma = self._to_device(self.sigma, device=device)
        self.map_indices = self._to_device(self.map_indices, device=device)

    def _to_device(self, a: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """Moves a tensor to the appropriate device."""
        if a.device != device:
            a = a.to(device=device)
        return a

    def __repr__(self):
        return f"{self.__class__.__name__}(\n\
            size={self.size[:-1]}, neuron_size={self.size[-1]}, alpha={self.alpha}, sigma={self.sigma}),\n\
            dist_fn={self.dist_fn.__name__}, neigh_fn={self.neigh_fn.__name__}, neigh_diff_fn={self.neigh_diff_fn})"
