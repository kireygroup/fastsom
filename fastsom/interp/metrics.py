"""
SOM statistics for various purpouses.
"""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from fastsom.core import expanded, idxs_1d_to_2d, idxs_2d_to_1d
from fastsom.som import Som, pdist

__all__ = [
    "cluster_stats",
    "codebook_err",
    "mean_quantization_err",
    "topologic_err",
]


# TODO: Refactor this function
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
    return (
        counts.float().std().log().cpu().numpy(),
        np.mean(max_distances),
        float(empty_clusters_count),
    )


def codebook_err(pred_b: Tensor, yb: Tensor, som: Som = None) -> Tensor:
    "Counts the number of records not belonging to each cluster that are closer than that cluster's furthest record."
    xb = som._recorder["xb"]
    w = som.weights.view(-1, xb.shape[-1])
    distances = expanded(xb, w.to(device=xb.device), pdist)
    row_sz = som.size[0]
    preds = idxs_2d_to_1d(pred_b, row_sz)
    n_classes = som.size[0] * som.size[1]
    batch_size = xb.shape[0]
    count = 0
    # Inverse is a reverse index: it gives us a grouping over predictions
    # that we can use to index back into our distance tensor
    _, inverse = preds.unique(dim=0, return_inverse=True)
    for cluster_idx in range(n_classes):
        # Get distances where records were assigned to this cluster_idx
        cluster_distances = distances[(inverse == cluster_idx).nonzero().view(-1)]
        # If there is at least one element in this cluster
        if len(cluster_distances) > 0:
            # Grab the furthest element in the cluster
            max_cluster_distance = cluster_distances[:, cluster_idx].max()
            # Now retrieve the non-cluster distances:
            non_cluster_distances = distances[(inverse != cluster_idx).nonzero().view(-1)]
            # And check if any of these elements has a distance from cluster_idx less than max_dist
            count += (
                (non_cluster_distances[:, cluster_idx] < max_cluster_distance)
                .nonzero()
                .shape[0]
            )
    return torch.tensor(count / n_classes / batch_size).to(device=xb.device)


def mean_quantization_err(pred_b: Tensor, yb: Tensor, som: Som = None) -> Tensor:
    "Mean distance of each record from its respective BMU."
    xb = som._recorder["xb"]
    w = som.weights.view(-1, xb.shape[-1]).to(device=xb.device)
    row_sz = som.size[0]
    preds = idxs_2d_to_1d(pred_b, row_sz)
    return pdist(xb, w[preds], p=2).mean()


def topologic_err(pred_b: Tensor, yb: Tensor, som: Som = None, thresh: int = 4) -> Tensor:
    """
    Min vec distance of each record with every class and checks if the second-to-min value belongs in the first-best neighborhood.
    If not, it gets added as an error.
    """
    xb = som._recorder["xb"]
    # Calculate distance between each element in `xb` and each weight
    distances = expanded(xb, som.weights.view(-1, xb.shape[-1]), pdist)
    # Get the indices of the 2 closest element (BMU and 2nd-to-BMU)
    _, closest_2_indices = distances.topk(2, largest=False, sorted=True, dim=-1)
    col_sz = som.size[1]
    # Unflatten indices (from 1d to 2d)
    top_2_2d_idxs = idxs_1d_to_2d(closest_2_indices.cpu().numpy(), col_sz).float()
    # Calculate index-based euclidean distance between first and second closest weights for each record
    map_distances = pdist(top_2_2d_idxs[:, 0], top_2_2d_idxs[:, 1], p=2)
    # Count how many distances are above the threshold
    return (map_distances >= thresh).int().float().sum(-1).to(device=xb.device)
