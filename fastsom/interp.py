"""
This file contains interpretation
utilities for Self-Organizing Maps.
"""
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fastai.tabular.data import DataLoader
from fastprogress.fastprogress import progress_bar
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

from fastsom.core import idxs_2d_to_1d, ifnone
from fastsom.learn import SomLearner

__all__ = ["SomInterpretation"]


class SomInterpretation:
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

    def __init__(self, learn: SomLearner) -> None:
        self.learn = learn
        self.pca = None
        self.w = learn.model.weights.clone().view(-1, learn.model.size[-1]).cpu()
        # TODO: denormalization?
        self.w = self.learn.denormalize(self.w)

    @classmethod
    def from_learner(cls, learn: SomLearner):
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
        """Returns an iterator over the training set."""
        iter_fn = progress_bar if progress else iter
        for xb, yb in iter_fn(self.learn.dls.train):
            # If Tabular, grab the continuous part
            if not isinstance(xb, torch.Tensor):
                xb = xb[1]
            yield xb, yb

    def _init_pca(self):
        """Initializes and fits the PCA instance."""
        self.pca = PCA(n_components=3)
        self.pca.fit(self.w)

    def show_hitmap(self, ds_idx: int = 0, save: bool = False) -> None:
        """
        Shows a hitmap with counts for each codebook unit over the dataset.

        Parameters
        ----------
        ds_idx : int, default=0
            Dataset index. 0: train, 1: valid, etc.
        save : bool default=False
            If True, saves the hitmap into a file.
        """
        plt.ioff()
        _, ax = plt.subplots(figsize=(10, 10))
        preds, _ = self.learn.get_preds(ds_idx)
        out, counts = preds.unique(return_counts=True, dim=0)
        z = torch.zeros(self.learn.model.size[:-1]).long()
        for i, c in enumerate(out):
            z[c[0], c[1]] += counts[i]

        sns.heatmap(z.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt="d")
        plt.show()

    def show_feature_heatmaps(
        self,
        feature_indices: Optional[Union[int, List[int]]] = None,
        recategorize: bool = True,
        denorm: bool = False,
        figsize: Tuple[int, int] = (12, 12),
        save: bool = False,
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
        plt.ioff()
        # Transform feature indices to a list
        if isinstance(feature_indices, int):
            feature_indices = [feature_indices]
        elif feature_indices is None:
            n_features = self.learn.model.size[-1]
            feature_indices = list(range(n_features))
        # If the DataLoader is for Tabular, gather feature names
        if self.learn.is_tabular:
            cat_names, cont_names, encoded_cat_names = self.learn.get_feature_names()
            labels = encoded_cat_names + cont_names if len(encoded_cat_names) > 0 else cat_names + cont_names
            # Optionally recategorize categorical variables
            if len(encoded_cat_names) > 0 and recategorize:
                w = self.learn.recategorize(self.w, denorm=denorm)
            else:
                w = self.w.numpy()
        else:  # Otherwise, use given features indices as names
            labels = [f"Feature #{i}" for i in feature_indices]
            w = self.w.numpy()

        # gather feature indices from weights
        w = np.take(w, feature_indices, axis=-1)

        # Initialize subplots
        cols = min(2, len(feature_indices))
        rows = max(1, len(feature_indices) // cols + (1 if len(feature_indices) % cols > 0 else 0))
        fig, axs = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        zipped_items = zip(
            range(len(feature_indices)),
            axs[: len(feature_indices)],
            np.split(w, w.shape[-1], axis=-1),
            labels,
        )
        for i, ax, data, label in progress_bar(list(zipped_items)):
            ax.set_title(label)
            if data.dtype.kind in ["S", "U", "O"]:
                # TODO: apply colors to strings
                data = data.astype(str)
                numeric_data = (np.searchsorted(np.unique(data), data, side="left") + 1).reshape(self.modelsize)
                sns.heatmap(numeric_data, ax=ax, annot=data.reshape(self.modelsize), fmt="s")
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
        plt.ioff()
        image_shape = (self.learn.model.size[0], self.learn.model.size[1], 3)
        if self.w.shape[-1] != 3:
            if self.pca is None:
                self._init_pca()
            # Calculate the 3-layer PCA of the weights
            d = self.pca.transform(self.w.numpy()).reshape(*image_shape)
        else:
            d = self.w.numpy()

        # Rescale values into the RGB space (0, 255)
        def rescale(d):
            return ((d - d.min(0)) / d.ptp(0) * 255).astype(int)

        d = rescale(d)
        # Show weights
        plt.figure(figsize=(10, 10))
        plt.imshow(d.reshape(image_shape))

    def show_preds(
        self,
        dl: Optional[DataLoader] = None,
        class_names: List[str] = None,
        n_bins: int = 5,
        save: bool = False,
    ) -> None:
        """
        Displays most frequent label for each map position in `dl` dataset.
        If labels are countinuous, binning on `n_bins` is performed.

        Parameters
        ----------
        dl : DataLoader, default=None
            The dataloader to use for prediction. Defaults to the validation set,
            or training set in case validation set is empty.
        n_bins : int default=5
            The number of bins to use when labels are continous.
        save : bool default=False
            Whether or not the output chart should be saved on a file.
        """
        plt.ioff()
        if not self.learn.has_labels:
            raise RuntimeError(
                "Unable to show predictions for a dataset that has no labels. \
                Please pass labels when creating the `DataBunch` or use `interp.show_hitmap()`"
            )
        # Run model predictions
        dl = ifnone(dl, self.learn.dls.valid if len(self.learn.dls.loaders) > 1 else self.learn.dls.train)
        preds, labels = self.learn.get_preds(dl=dl)
        map_size = (self.learn.model.size[0], self.learn.model.size[1])
        # Data placeholder
        data = torch.zeros(map_size[0] * map_size[1])

        # Check if labels are continuous
        is_target_continuous = "float" in str(labels.dtype)
        # Discretize the target
        if is_target_continuous and n_bins > 0:
            labels = KBinsDiscretizer(n_bins=n_bins, encode="ordinal").fit_transform(labels.cpu().numpy())
            labels = torch.tensor(labels)

        # Transform predictions (2D BMU indices) to 1D for easier processing
        preds_1d = idxs_2d_to_1d(preds, map_size[0])
        unique_bmus = preds_1d.unique(dim=0)

        # Count predictions for each Best-Matching Unit
        for idx, bmu in enumerate(unique_bmus):
            # Get labels corresponding to this BMU
            bmu_labels = labels[(preds_1d == bmu).nonzero()]

            if is_target_continuous and n_bins <= 0:
                data[idx] = bmu_labels.mean()
            else:
                # Calculate unique label counts
                unique_labels, label_counts = bmu_labels.unique(return_counts=True)
                data[idx] = unique_labels[label_counts.argmax()]
            # TODO show percentages + class color
            # max_label = label_counts.max()
            # data[idx] = float("{:.2f}".format(max_label.float() / float(len(bmu_labels))))

        if not is_target_continuous or n_bins > 0:
            # Legend labels
            unique_labels = labels.unique()
            class_names = ifnone(class_names, [str(label) for label in unique_labels.numpy()])
            # Color map
            colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_labels)))
            cmap = LinearSegmentedColormap.from_list("Custom", colors, len(colors))
        else:
            palette = sns.palettes.SEABORN_PALETTES["deep6"]
            cmap = ListedColormap(palette)

        f, ax = plt.subplots(figsize=(11, 9))
        # Plot the heatmap
        ax = sns.heatmap(data.view(map_size), annot=True, cmap=cmap, square=True, linewidths=0.5)

        if not is_target_continuous or n_bins > 0:
            # # Manually specify colorbar labelling after it's been generated
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks(unique_labels.numpy())
            colorbar.set_ticklabels(class_names)
        plt.show()
