"""
This file contains visualization callbacks
for Self-Organizing Maps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from fastai.callback.core import Callback
from fastai.learner import Learner

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from fastsom.core import idxs_2d_to_1d
from ..log import get_logger


def get_xy(dls):
    # TabularDataLoaders
    x, y = [], []
    for batch in dls.train:
        x.append(torch.cat([batch[0], batch[1]], dim=-1))
        y.append(batch[1])
    return torch.cat(x, dim=0), torch.cat(y, dim=0)
    # TODO: other types


__all__ = [
    "SomVizCallback",
    "SomTrainingViz",
    "SomHyperparamsViz",
    "SomBmuViz",
]


class SomVizCallback(Callback):
    """Base class for SOM visualization callbacks."""

    pass


class SomTrainingViz(SomVizCallback):
    """`Callback` used to visualize an approximation of the SOM weight update."""

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

    def __init__(self, *args, update_on_batch: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self)
        self.dim = 2
        self.input_el_size = None
        self.data_color, self.weights_color = "#539dcc", "#e58368"
        self.pca, self.f, self.ax, self.scatter = None, None, None, None
        self.update_on_batch = update_on_batch

    def before_fit(self, **kwargs):
        """Initializes the PCA on the dataset and creates the plot."""
        self.model = self.learn.model
        if not self.model.training:
            return
        # Retrieve data
        data, _ = get_xy(self.learn.dls)
        data = data.cpu().numpy()
        self.input_el_size = data.shape[-1]
        # Init + fit the PCA
        self.pca = PCA(n_components=self.dim)
        self.logger.error(self.pca)
        d = self.pca.fit_transform(data)
        # Make the chart interactive
        plt.ion()
        self.f = plt.figure()
        self.ax = self.f.add_subplot(111, projection="3d" if self.dim == 3 else None)
        # Calculate PCA of the weights
        w = self.pca.transform(
            self.model.weights.view(-1, self.input_el_size).cpu().numpy()
        )
        # Plot weights
        self.scatter = self.ax.scatter(
            *tuple([el[i] for el in w] for i in range(self.dim)),
            c=self.weights_color,
            zorder=100
        )
        # Plot data
        self.ax.scatter(
            *tuple([el[i] for el in d] for i in range(self.dim)), c=self.data_color
        )
        self.f.show()

    def _update_plot(self):
        if not self.model.training:
            return
        w = self.pca.transform(
            self.model.weights.view(-1, self.input_el_size).cpu().numpy()
        )
        t = tuple([el[i] for el in w] for i in range(self.dim))
        self.scatter.set_offsets(np.c_[t])
        self.f.canvas.draw()

    def before_epoch(self, **kwargs):
        """Updates the plot if needed."""
        if not self.update_on_batch:
            self._update_plot()

    def before_batch(self, **kwargs):
        """Updates the plot if needed."""
        if self.update_on_batch:
            self._update_plot()

    def after_fit(self, **kwargs):
        """Cleanup after training."""
        if not self.model.training:
            return
        del self.pca


class SomHyperparamsViz(SomVizCallback):
    """
    Displays a lineplot for each SOM hyperparameter.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fig, self.plots = None, None
        self.alphas, self.sigmas = [], []

    def before_fit(self, **kwargs):
        """Initializes the plots."""
        n_epochs = kwargs["n_epochs"]
        self.model = self.learn.model
        plt.ion()
        self.fig, self.plots = plt.subplots(1, 2, figsize=(12, 10))
        self.plots = self.plots.flatten()

        self.plots[0].set_title("Alpha Hyperparameter")
        self.plots[0].set_xlabel("Epoch")
        self.plots[0].set_ylabel("Alpha")
        self.plots[0].set_xlim([0, n_epochs])

        self.plots[1].set_title("Sigma hyperparameter")
        self.plots[1].set_xlabel("Epoch")
        self.plots[1].set_ylabel("Sigma")
        self.plots[1].set_xlim([0, n_epochs])

        self.fig.show()

    def after_epoch(self, **kwargs):
        """Updates hyperparameters and plots."""
        if not self.model.training:
            return
        self.alphas.append(self.model.alpha.cpu().numpy())
        self.sigmas.append(self.model.sigma.cpu().numpy())
        self.plots[0].plot(self.alphas, c="#589c7e")
        self.plots[1].plot(self.sigmas, c="#4791c5")
        self.fig.canvas.draw()


class SomBmuViz(SomVizCallback):
    """
    Visualization callback for SOM training.
    Stores BMU locations for each batch and displays them on epoch end.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """

    def __init__(self, *args, update_on_batch: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fig, self.ax = None, None
        self.epoch_counts, self.total_counts = 0, 0
        self.update_on_batch = update_on_batch

    def before_fit(self, **kwargs):
        self.model = self.learn.model
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.total_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.fig = plt.figure()

    def after_batch(self, **kwargs):
        "Saves BMU hit counts for this batch."
        bmus = self.model._recorder["bmus"]
        unique_bmus, bmu_counts = idxs_2d_to_1d(bmus, self.model.size[0]).unique(
            dim=0, return_counts=True
        )
        self.epoch_counts[unique_bmus] += bmu_counts
        if self.update_on_batch:
            self._update_plot()

    def after_epoch(self, **kwargs):
        "Updates total BMU counter and resets epoch counter."
        if not self.update_on_batch:
            self._update_plot()
        self.total_counts += self.epoch_counts
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])

    def after_fit(self, **kwargs):
        "Cleanup after training."
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.total_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.fig = None
        self.ax = None

    def _update_plot(self, **kwargs):
        "Updates the plot."
        if not self.model.training:
            return
        imsize = self.model.size[:-1]
        if self.ax is None:
            self.ax = plt.imshow(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.show()
        else:
            self.ax.set_data(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.canvas.draw()
