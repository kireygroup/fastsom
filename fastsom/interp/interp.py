"""
This file contains interpretation
utilities for Self-Organizing Maps.
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional, List, Union, Tuple
from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from fastai.tabular import TabularDataBunch
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from fastprogress.fastprogress import progress_bar
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from fastsom.core import ifnone, idxs_2d_to_1d, find
from fastsom.data_block import ToBeContinuousProc


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

    def __init__(self, learn: Learner) -> None:
        self.learn = learn
        self.data = learn.data
        self.pca = None
        self.w = learn.model.weights.clone().view(-1, learn.model.size[-1]).cpu()
        # TODO: denormalization?
        self.w = self.learn.denormalize(self.w)

    @classmethod
    def from_learner(cls, learn: Learner):
        """
        Creates a new instance of `SomInterpretation` from a `SomLearner`.\n

        Parameters
        ----------
        learn : SomLearner
            The learner to be used for interpretation.
        """
        return cls(learn)

    @property
    def modelsize(self):
        return self.learn.model.weights.shape[:-1]

    def _get_train_batched(self, progress: bool = True):
        iter_fn = progress_bar if progress else iter
        for xb, yb in iter_fn(self.learn.data.train_dl):
            if not isinstance(xb, torch.Tensor):
                xb = xb[1]
            yield xb, yb

    def _init_pca(self):
        "Initializes and fits the PCA instance."
        self.pca = PCA(n_components=3)
        self.pca.fit(self.w)

    def show_hitmap(self, save: bool = False) -> None:
        """
        Shows a hitmap with counts for each codebook unit over the dataset.

        Parameters
        ----------
        save : bool default=False
            If True, saves the hitmap into a file.
        """
        _, ax = plt.subplots(figsize=(10, 10))
        preds = torch.zeros(0, 2).cpu().long()
        for xb, yb in self._get_train_batched():
            preds = torch.cat([preds, self.learn.model(xb).cpu()], dim=0)

        out, counts = preds.unique(return_counts=True, dim=0)
        z = torch.zeros(self.learn.model.size[:-1]).long()
        for i, c in enumerate(out):
            z[c[0], c[1]] += counts[i]

        sns.heatmap(z.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt='d')
        plt.show()

    def show_feature_heatmaps(
        self,
        feature_indices: Optional[Union[int, List[int]]] = None,
        cat_labels: Optional[List[str]] = None,
        cont_labels: Optional[List[str]] = None,
        recategorize: bool = True,
        figsize: Tuple[int, int] = (12, 12),
        save: bool = False
    ) -> None:
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
        xb, _ = next(self._get_train_batched(progress=False))
        n_variables = xb.shape[-1]
        if isinstance(self.learn.data, TabularDataBunch):
            tobecont_proc = find(self.learn.data.processor[0].procs, lambda p: isinstance(p, ToBeContinuousProc))
            if tobecont_proc is not None:
                cat_labels, cont_labels = tobecont_proc.original_cat_names, tobecont_proc.original_cont_names
            else:
                cat_labels, cont_labels = self.learn.data.cat_names, self.learn.data.cont_names
        else:
            cat_labels = ifnone(cat_labels, [])
            cont_labels = ifnone(cont_labels, [])
        labels = cat_labels+cont_labels if len(cat_labels+cont_labels) > 0 else [f'Feature #{i}' for i in range(n_variables)]
        feature_indices = list(range(len(labels))) if feature_indices is None else feature_indices if isinstance(feature_indices, list) else [feature_indices]

        # Optionally recategorize categorical variables
        if recategorize:
            w = self.learn.recategorize(self.w, denorm=False)
        else:
            w = self.w.numpy()
        # gather feature indices from weights
        w = np.take(w, feature_indices, axis=-1)

        # Initialize subplots
        cols = min(2, len(feature_indices))
        rows = max(1, len(feature_indices) // cols + (1 if len(feature_indices) % cols > 0 else 0))
        fig, axs = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        zipped_items = zip(range(len(feature_indices)), axs[:len(feature_indices)], np.split(w, w.shape[-1], axis=-1), labels)
        for i, ax, data, label in progress_bar(list(zipped_items)):
            ax.set_title(label)
            if data.dtype.kind == 'S' or data.dtype.kind == 'U':
                # TODO: apply colors to strings
                data = data.astype(str)
                numeric_data = (np.searchsorted(np.unique(data), data, side='left') + 1).reshape(self.modelsize)
                sns.heatmap(numeric_data, ax=ax, annot=data.reshape(self.modelsize), fmt='s')
            else:
                sns.heatmap(data.reshape(self.modelsize), ax=ax, annot=True)
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
            d = self.pca.transform(self.w.numpy()).reshape(*image_shape)
        else:
            d = self.w.numpy()

        # Rescale values into the RGB space (0, 255)
        def rescale(d): return ((d - d.min(0)) / d.ptp(0) * 255).astype(int)
        d = rescale(d)
        # Show weights
        plt.figure(figsize=(10, 10))
        plt.imshow(d.reshape(image_shape))

    def show_preds(
        self,
        ds_type: DatasetType = DatasetType.Train,
        class_names: List[str] = None,
        n_bins: int = 5,
        save: bool = False
    ) -> None:
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
        if not self.learn.has_labels:
            raise RuntimeError('Unable to show predictions for a dataset that has no labels. \
                Please pass labels when creating the `DataBunch` or use `interp.show_hitmap()`')
        # Run model predictions
        preds, labels = self.learn.get_preds(ds_type)

        # Check if labels are continuous
        continuous_labels = 'float' in str(labels.dtype)

        if continuous_labels and n_bins > 0:
            # Split labels into bins
            labels = KBinsDiscretizer(n_bins=n_bins, encode='ordinal').fit_transform(labels.unsqueeze(-1).numpy())
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
