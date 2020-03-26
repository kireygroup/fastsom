"""
SOM statistics for various purpouses.
"""
import torch
import numpy as np
from torch import Tensor

from typing import Tuple

from kg_som.som import Som

__all__ = [
    "cluster_stats",
    "idxs_2d_to_1d",
    "idxs_1d_to_2d",
]


def cluster_stats(x: Tensor, som: Som) -> Tuple[float]:
    "Calculates cluster statistics for a Self-Organizing Map."
    som.eval()
    # Run model predictions (BMU indices) and convert them to 1d
    preds = idxs_2d_to_1d(som.forward(x.cuda()), som.weights.shape[0]).cuda()
    w = som.weights.view(-1, som.weights.shape[-1]).cuda()
    # Retrieve unique BMUs with count
    uniques, inverse, counts = preds.unique(dim=0, return_inverse=True, return_counts=True)
    # Calculate distance from each input and its BMU
    distances = (w[preds] - x.cuda()).pow(2).sum(-1).sqrt()
    max_distances = []
    # Get the max distance for each BMU cluster
    for b in uniques:
        idxs = (inverse == b).nonzero()
        if idxs.nelement() > 0:
            cluster_max_dist = distances[preds[idxs.squeeze(-1)]].max()
            max_distances.append(cluster_max_dist.cpu().numpy())
    # Calculate how many unused clusters were found
    empty_clusters_count = w.shape[0] - len(uniques)
    return counts.float().std().log().cpu().numpy(), np.mean(max_distances), float(empty_clusters_count)


def idxs_2d_to_1d(idxs: np.ndarray, row_size: int) -> list:
    "Turns 2D indices to 1D"
    return torch.tensor([el[0] * row_size + el[1] for el in idxs])


def idxs_1d_to_2d(idxs: np.ndarray, col_size: int) -> list:
    "Turns 1D indices to 2D"
    return torch.tensor([[el // col_size, el % col_size] for el in idxs])
