"""
"""
import torch
import numpy as np
from torch import Tensor

from .decorators import timeit
from .som import Som
from .stats import idxs_2d_to_1d


__all__ = [
    "cluster_loss",
    "BackwardRedirectTensor",
]


def cluster_loss(preds: Tensor, x: Tensor, som: Som = None, device=torch.device("cpu")) -> Tensor:
    "Calculates cluster statistics for a Self-Organizing Map."
    # Run model predictions (BMU indices) and convert them to 1d
    # preds = idxs_2d_to_1d(preds, som.weights.shape[0]).to(device=device)
    # w = som.weights.view(-1, som.weights.shape[-1]).to(device=device)
    # # Retrieve unique BMUs with count
    # uniques, counts = preds.unique(dim=0, return_counts=True)
    # # Calculate distance from each input and its BMU
    # distances = (w[preds] - x.to(device=device)).pow(2).sum(-1).sqrt()
    # max_distances = []
    # # Get the max distance for each BMU cluster
    # for b in uniques:
    #     idxs = (preds == b).nonzero()
    #     if idxs.nelement() > 0:
    #         cluster_max_dist = distances[idxs.squeeze(-1)].max()
    #         max_distances.append(cluster_max_dist.cpu().numpy())
    # # Calculate how many unused clusters were found
    # empty_clusters_count = w.shape[0] - len(uniques)
    # loss = counts.float().std().log().cpu() + np.sqrt(np.mean(max_distances))
    # return BackwardRedirectTensor(loss, som.backward)
    return BackwardRedirectTensor(torch.tensor(0.0), som.backward)


class BackwardRedirectTensor(Tensor):
    "A Tensor that calls a given function instead of regular `backward`."
    @staticmethod
    def __new__(cls, x, redir_fn, *args, **kwargs):
        return super().__new__(cls, x.numpy(), *args, **kwargs)

    def __init__(self, x, redir_fn):
        super().__init__()
        self.redir_fn = redir_fn

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        # print(f'{self.__class__.__name__}.backward has been called')
        self.redir_fn()
