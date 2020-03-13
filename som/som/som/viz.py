"""
This file contains visualization callbacks
for Self-Organizing Maps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING
from torch import Tensor
from fastai.callback import Callback, CallbackHandler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from .som import Som
from .stats import cluster_stats


__all__ = [
    "SomScatterVisualizer",
    "SomStatsVisualizer",
]


class SomScatterVisualizer(Callback):
    "`Learner` callback used to visualize an approximation of the Som weight update."
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

    def __init__(self, model: Som, data: Tensor, dim: int = 2, data_color: str = '#539dcc', weights_color: str = '#e58368') -> None:
        self.model = model
        self.data = data.clone().cpu().numpy()
        self.input_el_size = data.shape[-1]
        self.dim = dim
        self.data_color, self.weights_color = data_color, weights_color
        self.pca, self.f, self.ax, self.scatter = None, None, None, None

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
        # self.ax.set_xlim([-1, 1])
        # self.ax.set_ylim([-1, 1])
        # Plot data
        d = self.pca.transform(self.data)
        self.ax.scatter(*tuple([el[i] for el in d] for i in range(self.dim)), c=self.data_color)
        self.f.show()

    def on_epoch_begin(self, **kwargs):
        "Updates the plot."
        w = self.pca.transform(self.model.weights.view(-1, self.input_el_size).cpu().numpy())
        t = tuple([el[i] for el in w] for i in range(self.dim))
        self.scatter.set_offsets(np.c_[t])
        self.f.canvas.draw()

    def on_train_end(self, **kwargs):
        "Cleanup after training"
        del self.pca


class SomStatsVisualizer(Callback):
    "Accumulates and displays SOM statistics for each epoch"

    def __init__(self, model: Som, data: Tensor, plot_hyperparams: bool = False) -> None:
        self.model, self.data = model, data.clone().cpu()
        self.bmu_count = model.weights.shape[0] * model.weights.shape[1]
        self.vars, self.means, self.alphas, self.sigmas = [], [], [], []
        self.fig, self.vars_plt, self.means_plt, self.alphas_plt, self.sigmas_plt = None, None, None, None, None
        self.plot_hyperparams = plot_hyperparams

    def on_train_begin(self, **kwargs):
        "Initializes the plot"
        n_epochs = kwargs['n_epochs']
        plt.ion()
        subplots_size = (2, 2) if self.plot_hyperparams else (1, 2)
        self.fig, plots = plt.subplots(*subplots_size, figsize=(15, 5))
        plots = plots.flatten() if self.plot_hyperparams else plots
        self.vars_plt, self.means_plt = plots[0], plots[1]
        self.vars_plt.set_title('Log Cluster Count Variance')
        self.vars_plt.set_xlabel('Epoch')
        self.vars_plt.set_ylabel('Variance')
        self.vars_plt.set_xlim([0, n_epochs])

        self.means_plt.set_title('Mean of Max Distances from BMU')
        self.means_plt.set_xlabel('Epoch')
        self.means_plt.set_ylabel('Mean Distance')
        self.means_plt.set_xlim([0, n_epochs])

        if self.plot_hyperparams:
            self.alphas_plt, self.sigmas_plt = plots[2], plots[3]
            self.alphas_plt.set_title('Alpha Hyperparameter')
            self.alphas_plt.set_xlabel('Epoch')
            self.alphas_plt.set_ylabel('Alpha')
            self.alphas_plt.set_xlim([0, n_epochs])

            self.sigmas_plt.set_title('Sigma hyperparameter')
            self.sigmas_plt.set_xlabel('Epoch')
            self.sigmas_plt.set_ylabel('Sigma')
            self.sigmas_plt.set_xlim([0, n_epochs])

        self.fig.show()

    def on_epoch_end(self, **kwargs):
        "Updates statistics and plot"
        cluster_count_std, max_dist_mean, _ = cluster_stats(self.data, self.model)

        self.vars.append(cluster_count_std)
        self.means.append(max_dist_mean)
        if self.plot_hyperparams:
            self.alphas.append(self.model.alpha)
            self.sigmas.append(self.model.sigma)
        self._update_plot()

    def _update_plot(self):
        "Updates the plot"
        self.vars_plt.plot(self.vars, c='#589c7e')
        self.means_plt.plot(self.means, c='#4791c5')
        if self.plot_hyperparams:
            self.alphas_plt.plot(self.alphas, c='#589c7e')
            self.sigmas_plt.plot(self.sigmas, c='#4791c5')
        self.fig.canvas.draw()

    def _2d_idxs_to_1d(self, idxs: np.ndarray, row_size: int) -> list:
        "Turns 2D indices to 1D"
        return [el[0] * row_size + el[1] for el in idxs]

    def _1d_idxs_to_2d(self, idxs: np.ndarray, col_size: int) -> list:
        "Turns 1D indices to 2D"
        return [[el // col_size, el % col_size] for el in idxs]
