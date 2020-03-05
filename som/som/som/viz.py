"""
This file contains visualization callbacks
for Self-Organizing Maps.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import Tensor
from fastai.callback import Callback
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from .som import Som


class SomScatterVisualizer(Callback):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    def __init__(self, model: Som, data: Tensor, dim: int = 2, data_color: str = '#69F0AE', weights_color: str = '#FF5722') -> None:
        self.model = model
        self.data = data.clone().cpu().numpy()
        self.input_el_size = data.shape[-1]
        self.dim = dim
        self.epoch = 0
        self.data_color, self.weights_color = data_color, weights_color

    def on_train_begin(self):
        "Initializes the plot when the training begins"
        self.prepare_plot()

    def on_epoch_end(self):
        "Updates the plot after each epoch"
        self.epoch += 1
        self.update_plot()

    def on_train_end(self):
        "Cleanup after training"
        del self.pca
        del self.data

    def prepare_plot(self) -> None:
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

    def update_plot(self) -> None:
        "Updates the plot."
        w = self.pca.transform(self.model.weights.view(-1, self.input_el_size).cpu().numpy())
        t = tuple([el[i] for el in w] for i in range(self.dim))
        self.scatter.set_offsets(np.c_[t])
        self.f.canvas.draw()


class SomStatsVisualizer(Callback):
    "Accumulates and displays SOM statistics for each epoch"
    def __init__(self, n_epochs: int, learn) -> None:
        self.epoch, self.n_epochs = 0, n_epochs
        self.learn, self.data = learn, learn.data.train.clone().cpu()
        self.bmu_count = learn.model.weights.shape[0] * learn.model.weights.shape[1]
        self.vars, self.means = [[] for _ in range(self.bmu_count)], [[] for _ in range(self.bmu_count)]
        self.fig, self.vars_plt, self.means_plt = None, None, None
        
    def on_train_begin(self):
        "Prepares the plot"
        self._prepare_plot()
    
    def on_epoch_end(self):
        "Updates statistics and plot"
        self.epoch += 1
        # Gather predictions over the dataset
        preds = self.learn.predict(self.data)
        row_size, _, w_size = self.learn.model.weights.shape
        # Turn indices in the map to 1D
        preds = torch.tensor(self._2d_idxs_to_1d(preds.cpu().numpy().astype(int), row_size))
        # Access BMU weights
        w = self.learn.model.weights.view(-1, w_size).cpu()
        # Evaluate unique BMUs and inverse access indices
        uniques, inverse = preds.unique(dim=0, return_inverse=True)
        # Calculate euclidean distances between each input and its BMU
        d = (w[preds] - self.data.cpu()).pow(2).sum(1).sqrt()
        # For each cluster, eval stats
        for b in range(self.bmu_count):
            if b in uniques:
                idxs = (inverse == b).nonzero()
                var = w[preds[idxs.squeeze(-1)]].std()
                mean = d[preds[idxs.squeeze(-1)]].mean()
                self.vars[b].append(var.numpy())
                self.means[b].append(mean.numpy())
        self._update_plot()
    
    def _prepare_plot(self):
        "Initializes the plot"
        plt.ion()
        self.fig, (self.vars_plt, self.means_plt) = plt.subplots(1, 2, figsize=(15,5))
        self.vars_plt.set_title('Cluster Variance')
        self.vars_plt.set_xlabel('Epoch')
        self.vars_plt.set_ylabel('Variance')
        self.vars_plt.set_xlim([0, self.n_epochs])
        
        self.means_plt.set_title('Cluster Mean Distance')
        self.means_plt.set_xlabel('Epoch')
        self.means_plt.set_ylabel('Mean Distance')
        self.means_plt.set_xlim([0, self.n_epochs])
        
        self.fig.show()
    
    def _update_plot(self):
        "Updates the plot"
        # plt.legend(range(self.dists[0]))
        for b in range(self.bmu_count):
            self.vars_plt.plot(self.vars[b], label=f'Cluster {b}')
            self.means_plt.plot(self.means[b], label=f'Cluster {b}')
        self.fig.canvas.draw()
    
    def _2d_idxs_to_1d(self, idxs: np.ndarray, row_size: int) -> list:
        "Turns 2D indices to 1D"
        return [el[0] * row_size + el[1] for el in idxs]
        

# TODO: heatmap + hitmap
"""
HitMap                  : x -> predict -> position -> count by position
HeatMap                 : x -> per ciascuna variabile, plot di una casella con il valore di ogni peso (ROWSxCOLS)
HeatMap Omnicomprensiva : x -> PCA a 3 e uso quei 3 su 0,255 per colorazione cella

"""
__all__ = [
    "SomScatterVisualizer",
    "SomStatsVisualizer",
]
