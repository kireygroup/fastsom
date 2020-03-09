"""
This file contains interpretation 
utilities for Self-Organizing Maps
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import Tensor
from typing import Optional, List
from sklearn.decomposition import PCA

from .learn import SomLearner
from ..core import ifnone, listify
from ..datasets import UnsupervisedDataset


class SomInterpretation():
    "SOM interpretation class"

    def __init__(self, learn: SomLearner) -> None:
        self.learn = learn
        self.data: UnsupervisedDataset = learn.data
        self.map_size = learn.model.weights.shape
        self.w = learn.model.weights.clone().view(-1, self.map_size[-1]).cpu().numpy()
        self.pca = None

    @classmethod
    def from_learner(cls, learn: SomLearner):
        return cls(learn)

    def show_hitmap(self, data: Tensor = None) -> None:
        "Displays a hitmap"
        _, ax = plt.subplots(figsize=(10, 10))
        d = ifnone(data, self.data.train)
        preds = self.learn.predict(d)
        out, counts = preds.unique(return_counts=True, dim=0)
        z = torch.zeros(self.map_size[:-1]).long()
        for i, c in enumerate(out):
            z[c[0], c[1]] += counts[i]

        sns.heatmap(z.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt='d')
        plt.show()

    def show_feature_heatmaps(self, dim: Optional[int] = None, labels: Optional[List[str]] = None) -> None:
        "Displays a hitmap"
        dims = [dim] if dim is not None else list(range(self.data.train.shape[-1]))
        s = np.sqrt(0.0 + len(dims)).astype(int)
        rows, cols = (1, len(dims)) if s * s < len(dims) else (s, s)
        fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

        labels = ifnone(labels, [' ' for _ in range(len(dims))])

        if len(dims) == 1:
            axs = [[axs]]
        elif rows == 1 or cols == 1:
            axs = [axs]

        for d in dims:
            i = d % s
            j = d // s
            axs[i][j].set_title(labels[d])
            sns.heatmap(self.w[:, d].reshape(self.map_size[:-1]), ax=axs[i][j], annot=True)
        fig.show()

    def show_omni_heatmap(self):
        "Displays a colored heatmap"
        if self.pca is None:
            self.init_pca()

        if self.w.shape[-1] != 3:
            # Calculate the 3-layer PCA of the weights
            d = self.pca.transform(self.w).reshape(*self.map_size[:-1], 3)
        else:
            d = self.w.reshape(*self.map_size[:-1], 3)
        # Scale the 3 layers in [0, 255]
        # colors = 255 * (d - d.min()) / (d.max() - d.min())
        # colors = colors.astype(int)
        # Plot w/ colors
        plt.imshow(d)

    def init_pca(self):
        self.pca = PCA(n_components=3)
        self.pca.fit(self.w)


__all__ = [
    "SomInterpretation",
]
