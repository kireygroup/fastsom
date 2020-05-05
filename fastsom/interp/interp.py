"""
This file contains interpretation
utilities for Self-Organizing Maps.
"""
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import TensorDataset, BatchSampler
from typing import Optional, List, Union
from fastai.basic_data import DatasetType
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from fastprogress.fastprogress import progress_bar
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from fastsom.core import ifnone, idxs_2d_to_1d
from fastsom.datasets import get_sampler


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

    def show_weights(self, save: bool = False) -> None:
        """
        Shows a colored heatmap of the SOM codebooks.
        data = idxs_1d_to_2d(data, self.learn.model.size[1])

        Parameters
        ----------
        save : bool default=False
            If True, saves the heatmap into a file.
        """

        image_shape = (self.learn.model.size[0], self.learn.model.size[1], 3)
        if self.w.shape[-1] != 3:
            if self.pca is None:
                self._init_pca()
            # Calculate the 3-layer PCA of the weights
            d = self.pca.transform(self.w).reshape(*image_shape)
        else:
            d = self.w

        # Rescale values into the RGB space (0, 255)
        def rescale(d): return ((d - d.min(0)) / d.ptp(0) * 255).astype(int)
        d = rescale(d)
        # Show weights
        plt.figure(figsize=(10, 10))
        plt.imshow(d.reshape(image_shape))

    def show_preds(self, ds_type: DatasetType = DatasetType.Train, class_names: List[str] = None, n_bins: int = 5, save: bool = False, ) -> None:
        """
        Displays most frequent label for each map position in `ds_type` dataset.
        If labels are countinuous, binning on `n_bins` is performed.

        Parameters
        ----------
        ds_type : DatasetType default=DatasetType.Train
            The enum of the dataset to be used.
        n_bins : int default=5
            The number of bins to use when labels are continous.
        save : bool default=False
            Whether or not the output chart should be saved on a file.
        """
        if not self.learn.data.has_labels:
            raise RuntimeError('Unable to show predictions for a dataset that has no labels. \
                Please pass labels when creating the `UnsupervisedDataBunch` or use `interp.show_hitmap()`')
        # Run model predictions
        preds, labels = self.learn.get_preds(ds_type)

        # Check if labels are continuous
        continuous_labels = 'float' in str(labels.dtype)

        if continuous_labels and n_bins > 0:
            # Split labels into bins
            labels = KBinsDiscretizer(n_bins=n_bins, encode='ordinal').fit_transform(labels.numpy())
            labels = torch.tensor(labels)

        map_size = (self.learn.model.size[0], self.learn.model.size[1])

        # Data placeholder
        data = torch.zeros(map_size[0] * map_size[1])

        # Transform BMU indices to 1D for easier processing
        preds_1d = idxs_2d_to_1d(preds, map_size[0])
        unique_bmus = preds_1d.unique(dim=0)

        for idx, bmu in enumerate(unique_bmus):
            # Get labels corresponding to this BMU
            bmu_labels = labels[(preds_1d == bmu).nonzero()]

            if continuous_labels and n_bins <= 0:
                data[idx] = bmu_labels.mean()
            else:
                # Calculate unique label counts
                unique_labels, label_counts = bmu_labels.unique(return_counts=True)
                data[idx] = unique_labels[label_counts.argmax()]
            # TODO show percentages + class color
            # max_label = label_counts.max()
            # data[idx] = float("{:.2f}".format(max_label.float() / float(len(bmu_labels))))

        if not continuous_labels or n_bins > 0:
            # Legend labels
            unique_labels = labels.unique()
            class_names = ifnone(class_names, [str(label) for label in unique_labels.numpy()])
            # Color map
            colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_labels)))
            cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        else:
            palette = sns.palettes.SEABORN_PALETTES['deep6']
            cmap = ListedColormap(palette)

        f, ax = plt.subplots(figsize=(11, 9))
        # Plot the heatmap
        ax = sns.heatmap(data.view(map_size), annot=True, cmap=cmap, square=True, linewidths=.5)

        if not continuous_labels or n_bins > 0:
            # # Manually specify colorbar labelling after it's been generated
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks(unique_labels.numpy())
            colorbar.set_ticklabels(class_names)
        plt.show()
