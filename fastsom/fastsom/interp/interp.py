"""
This file contains interpretation
utilities for Self-Organizing Maps
"""
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import TensorDataset, BatchSampler
from typing import Optional, List, Collection, Union
from sklearn.decomposition import PCA
from fastprogress.fastprogress import progress_bar

from fastsom.core import ifnone, listify
from fastsom.datasets import UnsupervisedDataBunch, get_sampler


__all__ = [
    "SomInterpretation",
]


class SomInterpretation():
    """
    SOM interpretation utility.

    Displays various information about a trained Self-Organizing Map, such as
    topological weight distribution, features distribution and training set 
    distribution over the map.

    Parameters
    ----------
    learn : SomLearner
        The learner to be used for interpretation.
    """

    def __init__(self, learn) -> None:
        self.learn = learn
        self.data = learn.data
        self.pca = None
        self.w = learn.model.weights.clone().view(-1, learn.model.size[-1]).cpu()
        if self.data.normalizer is not None:
            self.w = self.data.denormalize(self.w).numpy()

    @classmethod
    def from_learner(cls, learn):
        """
        Creates a new instance of `SomInterpretation` from a `SomLearner`.\n

        Parameters
        ----------
        learn : SomLearner
            The learner to be used for interpretation.
        """
        return cls(learn)

    def _get_train(self):
        return self.data.train_ds.tensors[0].cpu()

    def _init_pca(self):
        "Initializes and fits the PCA instance."
        self.pca = PCA(n_components=3)
        self.pca.fit(self.w)

    def show_hitmap(self, data: Tensor = None, bs: int = 64, save: bool = False) -> None:
        """
        Shows a hitmap with counts for each codebook unit over the dataset.

        Parameters
        ----------
        data : Tensor default=None
            The dataset to be used for prediction; defaults to the training set if None.
        bs : int default=64
            The batch size to be used to run model predictions.
        save : bool default=False
            If True, saves the hitmap into a file.
        """
        _, ax = plt.subplots(figsize=(10, 10))
        d = data if data is not None else self._get_train()
        bs = min(bs, len(d))
        sampler = BatchSampler(get_sampler('seq', TensorDataset(d, d), bs), batch_size=bs, drop_last=True)
        preds = torch.zeros(0, 2).cpu().long()

        for xb_slice in iter(sampler):
            preds = torch.cat([preds, self.learn.model(d[xb_slice]).cpu()], dim=0)

        out, counts = preds.unique(return_counts=True, dim=0)
        z = torch.zeros(self.learn.model.size[:-1]).long()
        for i, c in enumerate(out):
            z[c[0], c[1]] += counts[i]

        sns.heatmap(z.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt='d')
        plt.show()

    def show_feature_heatmaps(self,
                              dim: Optional[Union[int, List[int]]] = None,
                              cat_labels: Optional[List[str]] = None,
                              cont_labels: Optional[List[str]] = None,
                              recategorize: bool = True,
                              save: bool = False) -> None:
        """
        Shows a heatmap for each feature displaying its value distribution over the codebook.

        Parameters
        ----------
        dim : Optional[Union[int, List[int]]] default=None
            Indices of features to be shown; defaults to all features.
        cat_labels : Optional[List[str]] default=None
            Categorical feature labels.
        cont_labels : Optional[List[str]] default=None
            Continuous feature labels.
        recategorize : bool default=True
            If True, converts back categorical features that were previously made continuous.
        save : bool default=False
            If True, saves the charts into a file.
        """
        n_variables = self._get_train().shape[-1]
        cat_labels = ifnone(cat_labels, [])
        cont_labels = ifnone(cont_labels, [])
        labels = cat_labels+cont_labels if len(cat_labels+cont_labels) > 0 else [f'Feature #{i}' for i in range(n_variables)]

        if dim is not None:
            if isinstance(dim, list):
                dims = dim
            else:
                dims = [dim]
        else:
            dims = list(range(len(labels)))

        cols = 4 if len(dims) > 4 else len(dims)
        rows = math.ceil(len(dims) / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

        # Optionally recategorize categorical variables
        if recategorize:
            w = torch.tensor(self.w)
            encoded_count = self.w.shape[-1] - len(cont_labels)
            cat = self.learn.data.cat_enc.make_categorical(w[:, :encoded_count])
            w = np.concatenate([cat, torch.tensor(self.w[:, encoded_count:])], axis=-1)
        else:
            w = self.w

        if len(dims) == 1:
            axs = [[axs]]
        elif rows == 1 or cols == 1:
            axs = [axs]

        for d in progress_bar(range(len(dims))):
            i = d // cols
            j = d % cols
            ax = axs[i][j]
            ax.set_title(labels[d])
            sns.heatmap(w[:, d].reshape(self.learn.model.size[:-1]), ax=ax, annot=True)
        fig.show()

    def show_weights(self, save: bool = False):
        """
        Shows a colored heatmap of the SOM codebooks.

        Parameters
        ----------
        save : bool default=False
            If True, saves the heatmap into a file.
        """
        if self.w.shape[-1] != 3:
            if self.pca is None:
                self._init_pca()
            # Calculate the 3-layer PCA of the weights
            d = self.pca.transform(self.w).reshape(*self.learn.model.size[:-1], 3)
        else:
            d = self.w.reshape(*self.learn.model.size[:-1], 3)

        # Rescale values into the RGB space (0, 255)
        def rescale(d): return ((d - d.min(0)) / d.ptp(0) * 255).astype(int)
        d = rescale(self.w)
        # Show weights
        plt.imshow(d.reshape(self.learn.model.size))
