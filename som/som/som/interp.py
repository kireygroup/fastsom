"""
This file contains interpretation 
utilities for Self-Organizing Maps
"""
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import Tensor
from torch.utils.data import TensorDataset, BatchSampler
from typing import Optional, List, Collection, Union
from sklearn.decomposition import PCA
from fastprogress.fastprogress import progress_bar

from ..core import ifnone, listify
from ..datasets import UnsupervisedDataBunch, get_sampler


class SomInterpretation():
    "SOM interpretation class"

    def __init__(self, learn) -> None:
        self.learn = learn
        self.data: UnsupervisedDataBunch = learn.data
        self.map_size = learn.model.weights.shape
        self.w = learn.model.weights.clone().view(-1, self.map_size[-1]).cpu().numpy()
        self.pca = None

    @classmethod
    def from_learner(cls, learn):
        return cls(learn)

    def _get_train(self):
        return self.data.train_ds.tensors[0].cpu()

    def show_hitmap(self, data: Tensor = None, bs: int = 64) -> None:
        "Shows a hitmap with counts for each codebook unit over `data` (if provided) or on the train dataset."
        _, ax = plt.subplots(figsize=(10, 10))
        d = data if data is not None else self._get_train()
        bs = min(bs, len(d))
        sampler = BatchSampler(get_sampler('seq', TensorDataset(d, d), bs), batch_size=bs, drop_last=True)
        preds = torch.zeros(0, 2).cpu().long()

        for xb_slice in iter(sampler):
            preds = torch.cat([preds, self.learn.model(d[xb_slice]).cpu()], dim=0)

        out, counts = preds.unique(return_counts=True, dim=0)
        z = torch.zeros(self.map_size[:-1]).long()
        for i, c in enumerate(out):
            z[c[0], c[1]] += counts[i]

        sns.heatmap(z.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt='d')
        plt.show()

    def show_feature_heatmaps(self, dim: Optional[Union[int, Collection[int]]] = None, labels: Optional[List[str]] = None) -> None:
        "Shows a heatmap for each feature displaying its values over the codebook."
        if dim is not None:
            if isinstance(dim, list):
                dims = dim
            else:
                dims = [dim]
        else:
            dims = list(range(self._get_train().shape[-1]))
        cols = 4 if len(dims) > 4 else len(dims)
        rows = math.ceil(len(dims) / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
        labels = ifnone(labels, [f'Feature #{i}' for i in range(len(dims))])

        if len(dims) == 1:
            axs = [[axs]]
        elif rows == 1 or cols == 1:
            axs = [axs]

        for d in progress_bar(range(len(dims))):
            i = d // cols
            j = d % cols
            ax = axs[i][j]
            ax.set_title(labels[d])
            sns.heatmap(self.w[:, d].reshape(self.map_size[:-1]), ax=ax, annot=True)
        fig.show()

    def show_weights(self):
        "Shows a colored heatmap of the SOM weights."
        if self.pca is None:
            self.init_pca()
        if self.w.shape[-1] != 3:
            # Calculate the 3-layer PCA of the weights
            d = self.pca.transform(self.w).reshape(*self.map_size[:-1], 3)
        else:
            d = self.w.reshape(*self.map_size[:-1], 3)

        plt.imshow(((d - d.min(0)) / d.ptp(0) * 255).astype(int))

    def init_pca(self):
        "Initializes and fits the PCA instance."
        self.pca = PCA(n_components=3)
        self.pca.fit(self.w)


__all__ = [
    "SomInterpretation",
]
