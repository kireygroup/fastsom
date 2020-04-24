"""
This file contains visualization callbacks
for Self-Organizing Maps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from fastai.callback import Callback
from fastai.basic_train import Learner
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from ..core import idxs_2d_to_1d


__all__ = [
    "SomTrainingViz",
    "SomHyperparamsViz",
    "SomBmuViz",
]


class SomTrainingViz(Callback):
    """
    `Callback` used to visualize an approximation of the Som weight update.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

    def __init__(self, learn: Learner, update_on_batch: bool = False) -> None:
        self.dim = 2
        self.learn = learn
        self.model = learn.model
        self.data = learn.data.train_ds.tensors[0].clone().cpu().numpy()
        self.input_el_size = self.data.shape[-1]
        self.data_color, self.weights_color = '#539dcc', '#e58368'
        self.pca, self.f, self.ax, self.scatter = None, None, None, None
        self.update_on_batch = update_on_batch

    def on_train_begin(self, **kwargs):
        "Initializes the PCA on the dataset and creates the plot."
        # Init + fit the PCA
        self.pca = PCA(n_components=self.dim)
        self.pca.fit(self.data)
        # Make the chart interactive
        plt.ion()
        self.f = plt.figure()
        self.ax = self.f.add_subplot(111, projection='3d' if self.dim == 3 else None)
        w = self.pca.transform(self.model.weights.view(-1, self.input_el_size).cpu().numpy())

        # Plot weights
        self.scatter = self.ax.scatter(*tuple([el[i] for el in w] for i in range(self.dim)), c=self.weights_color, zorder=100)
        # Plot data
        d = self.pca.transform(self.data)
        self.ax.scatter(*tuple([el[i] for el in d] for i in range(self.dim)), c=self.data_color)
        self.f.show()

    def _update_plot(self):
        w = self.pca.transform(self.model.weights.view(-1, self.input_el_size).cpu().numpy())
        t = tuple([el[i] for el in w] for i in range(self.dim))
        self.scatter.set_offsets(np.c_[t])
        self.f.canvas.draw()

    def on_epoch_begin(self, **kwargs):
        """Updates the plot if needed."""
        if not self.update_on_batch:
            self._update_plot()

    def on_batch_begin(self, **kwargs):
        """Updates the plot if needed."""
        if self.update_on_batch:
            self._update_plot()

    def on_train_end(self, **kwargs):
        """Cleanup after training."""
        del self.pca


class SomHyperparamsViz(Callback):
    """
    Displays a lineplot for each SOM hyperparameter.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """

    def __init__(self, learn: Learner) -> None:
        self.learn = learn
        self.model = learn.model
        self.fig, self.plots = None, None
        self.alphas, self.sigmas = [], []

    def on_train_begin(self, **kwargs):
        """Initializes the plots."""
        n_epochs = kwargs['n_epochs']
        plt.ion()
        self.fig, self.plots = plt.subplots(2, figsize=(15, 5))

        self.plots[0].set_title('Alpha Hyperparameter')
        self.plots[0].set_xlabel('Epoch')
        self.plots[0].set_ylabel('Alpha')
        self.plots[0].set_xlim([0, n_epochs])

        self.plots[1].set_title('Sigma hyperparameter')
        self.plots[1].set_xlabel('Epoch')
        self.plots[1].set_ylabel('Sigma')
        self.plots[1].set_xlim([0, n_epochs])

        self.fig.show()

    def on_epoch_end(self, **kwargs):
        """Updates hyperparameters and plots."""
        self.alphas.append(self.model.alpha.cpu().numpy())
        self.sigmas.append(self.model.sigma.cpu().numpy())
        self.alphas_plt.plot(self.alphas, c='#589c7e')
        self.sigmas_plt.plot(self.sigmas, c='#4791c5')
        self.fig.canvas.draw()


class SomBmuViz(Callback):
    """
    Visualization callback for SOM training.
    Stores BMU locations for each batch and displays them on epoch end.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """

    def __init__(self, learn: Learner, update_on_batch: bool = False) -> None:
        self.learn = learn
        self.model = learn.model
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.total_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.fig, self.ax = None, None
        self.update_on_batch = update_on_batch

    def on_train_begin(self, **kwargs):
        self.fig = plt.figure()

    def on_batch_end(self, **kwargs):
        "Saves BMU hit counts for this batch."
        bmus = self.model._recorder['bmus']
        unique_bmus, bmu_counts = idxs_2d_to_1d(bmus, self.model.size[0]).unique(dim=0, return_counts=True)
        self.epoch_counts[unique_bmus] += bmu_counts
        if self.update_on_batch:
            self._update_plot()

    def on_epoch_end(self, **kwargs):
        "Updates total BMU counter and resets epoch counter."
        if not self.update_on_batch:
            self._update_plot()
        self.total_counts += self.epoch_counts
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])

    def on_train_end(self, **kwargs):
        "Cleanup after training."
        self.epoch_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.total_counts = torch.zeros(self.model.size[0] * self.model.size[1])
        self.fig = None
        self.ax = None

    def _update_plot(self, **kwargs):
        "Updates the plot."
        imsize = self.model.size[:-1]
        if self.ax is None:
            self.ax = plt.imshow(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.show()
        else:
            self.ax.set_data(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.canvas.draw()
