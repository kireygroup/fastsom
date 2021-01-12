"""
This file contains visualization callbacks
for Self-Organizing Maps.
"""

import enum
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from fastai.callback.core import Callback
from fastai.torch_core import ifnone
from fastcore.meta import delegates
from sklearn.decomposition import PCA

from fastsom.core import idxs_2d_to_1d
from fastsom.log import has_logger

from .decorators import no_export, training_only
from .plotly_utils import scatter, show_figure

__all__ = [
    "PCABasedVisualizationCallback",
    "SomTrainingVisualizationCallback",
    "SomTrainingVisualizationCallback2",
    "SomHyperparamsVisualizationCallback",
    "SomBmuVisualizationCallback",
    "get_xy",
    "SOM_TRAINING_VIZ",
    "get_visualization_callbacks",
]


def get_xy(dls):
    """Grabs data as tensors from a DataLoaders instance."""
    x, y = [], []
    for batch in dls.train:
        x.append(torch.cat(list(batch)[:-1], dim=-1) if len(batch) > 2 else batch[0])
        y.append(batch[-1])
    return torch.cat(x, dim=0), torch.cat(y, dim=0)


class PCABasedVisualizationCallback(Callback):
    """Base class for a callback that can perform 2D/3D PCA operations for visualization purposes."""

    def __init__(self, is_3d: bool = True):
        super().__init__()
        self.is_3d = is_3d
        self.pca = PCA(n_components=3 if is_3d else 2)

    def do_pca(self, data: Union[torch.Tensor, np.ndarray], do_train: bool = False):
        """Performs PCA over `data`. Trains the PCA model if `do_train` is `True`."""
        data = data if isinstance(data, np.ndarray) else data.cpu().numpy()
        data = data if len(data.shape) < 3 else data.reshape(-1, data.shape[-1])
        if do_train:
            return self.pca.fit_transform(data)
        else:
            return self.pca.transform(data)


@no_export
class SomTrainingVisualizationCallback(PCABasedVisualizationCallback):
    """Visualize SOM weights VS. dataset during training using Plotly."""
    @delegates(PCABasedVisualizationCallback.__init__)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_pca    : np.ndarray      = None
        self.weight_pca   : np.ndarray      = None
        self.train_trace  : go.Scatter      = None
        self.weight_trace : go.Scatter      = None
        self.fig          : go.FigureWidget = None
        self.__show       : bool            = False

    def before_fit(self, **kwargs):
        self.train_pca = ifnone(self.train_pca, self.do_pca(get_xy(self.learn.dls)[0], do_train=True))
        self.train_trace = scatter(self.train_pca, name='Training data', mode='markers', marker_color='#539dcc', marker_size=1.5 if self.is_3d else 4)
        self.weight_pca = self.do_pca(self.learn.model.weights)
        self.weight_trace = scatter(self.weight_pca, name='SOM weights', mode='markers', marker_color='#e58368', marker_size=3 if self.is_3d else 6)
        expl_var = str(tuple(map(lambda pct: f'{pct:.0f}%', self.pca.explained_variance_ratio_ * 100)))[1:-1]
        layout = go.Layout(title=f"SOM Visualization ({expl_var} explained variance)")
        self.fig = go.FigureWidget([self.train_trace, self.weight_trace], layout=layout)
        self.fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")

    @training_only
    def before_train(self):
        if not self.__show:
            show_figure(self.fig)
            self.__show = True
        self.weight_pca = self.do_pca(self.learn.model.weights)
        with self.fig.batch_update():
            self.fig.data[1].x = self.weight_pca[:, 0]
            self.fig.data[1].y = self.weight_pca[:, 1]
            if self.is_3d:
                self.fig.data[1].z = self.weight_pca[:, 2]

    def after_fit(self):
        self.__show = False


@no_export
class SomTrainingVisualizationCallback2(PCABasedVisualizationCallback):
    """Visualize SOM weights VS. dataset during training using Matplotlib."""
    @delegates(PCABasedVisualizationCallback.__init__)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig            : plt.figure      = None
        self.train_axis     : plt.axis        = None
        self.weight_scatter : plt.scatter     = None
        self.train_pca      : np.ndarray      = None
        self.weight_pca     : np.ndarray      = None
        self.__show         : bool            = False

    def before_fit(self):
        self.train_pca = ifnone(self.train_pca, self.do_pca(get_xy(self.learn.dls)[0], do_train=True))
        self.weight_pca = self.do_pca(self.learn.model.weights)
        plt.ion()
        self.fig = plt.figure()
        self.train_axis = self.fig.add_subplot(111, projection='3d' if self.is_3d else None)
        # Draw weights
        self.weight_scatter = self.train_axis.scatter(*tuple([self.weight_pca[:, i] for i in range(self.weight_pca.shape[-1])]), c="#e58368", zorder=100)
        # Draw training data (once)
        self.train_axis.scatter(*tuple([self.train_pca[:, i] for i in range(self.train_pca.shape[-1])]), c="#539dcc")

    @training_only
    def before_train(self):
        if not self.__show:
            self.fig.show()
            self.__show = True
        # Read new weight values, compute PCA and update them on the scatter plot
        w = self.do_pca(self.learn.model.weights.view(-1, self.learn.model.weights.shape[-1]))
        w = tuple(w[:, i] for i in range(w.shape[-1]))
        self.weight_scatter.set_offsets(np.c_[w])
        self.fig.canvas.draw()

    def after_fit(self):
        self.__show = False


@no_export
class SomHyperparamsVisualizationCallback(Callback):
    """
    Displays a lineplot for each SOM hyperparameter.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """
    def __init__(self):
        self.fig    : plt.figure            = None
        self.plots  : np.ndarray            = None
        self.alphas : List[torch.Tensor]    = []
        self.sigmas : List[torch.Tensor]    = []
        self.__show : bool                  = False

    def before_fit(self, **kwargs):
        """Initializes the plots."""
        n_epochs = self.learn.n_epoch
        plt.ion()
        self.fig, self.plots = plt.subplots(1, 2, figsize=(16, 6))
        self.alphas, self.sigmas = [], []
        self.plots = self.plots.flatten()

        self.plots[0].set_title("Alpha Hyperparameter")
        self.plots[0].set_xlabel("Epoch")
        self.plots[0].set_ylabel("Alpha")
        self.plots[0].set_xlim([0, n_epochs])

        self.plots[1].set_title("Sigma hyperparameter")
        self.plots[1].set_xlabel("Epoch")
        self.plots[1].set_ylabel("Sigma")
        self.plots[1].set_xlim([0, n_epochs])

    @training_only
    def after_train(self, **kwargs):
        """Updates hyperparameters and plots."""
        if not self.__show:
            self.fig.show()
            self.__show = True
        self.alphas.append(self.learn.model.alpha.cpu().numpy())
        self.sigmas.append(self.learn.model.sigma.cpu().numpy())
        self.plots[0].plot(self.alphas, c="#589c7e")
        self.plots[1].plot(self.sigmas, c="#4791c5")
        self.fig.canvas.draw()

    def after_fit(self, **kwargs):
        self.__show = False


@no_export
class SomBmuVisualizationCallback(Callback):
    """
    Visualization callback for SOM training.
    Stores BMU locations for each batch and displays them on epoch end.

    Parameters
    ----------
    learn : Learner
        The `Learner` instance.
    """
    def __init__(self, update_on_batch: bool):
        self.fig, self.ax = None, None
        self.epoch_counts, self.total_counts = 0, 0
        self.update_on_batch = update_on_batch
        self.__show = False

    def before_fit(self, **kwargs):
        self.epoch_counts = torch.zeros(self.learn.model.size[0] * self.learn.model.size[1])
        self.total_counts = torch.zeros(self.learn.model.size[0] * self.learn.model.size[1])
        self.fig = plt.figure()

    @training_only
    def after_batch(self, **kwargs):
        "Saves BMU hit counts for this batch."
        bmus = self.learn.model._recorder["bmus"]
        unique_bmus, bmu_counts = idxs_2d_to_1d(bmus, self.learn.model.size[0]).unique(dim=0, return_counts=True)
        self.epoch_counts[unique_bmus] += bmu_counts
        if self.update_on_batch:
            self._update_plot()

    @training_only
    def after_train(self, **kwargs):
        "Updates total BMU counter and resets epoch counter."
        if not self.update_on_batch:
            self._update_plot()
        self.total_counts += self.epoch_counts
        self.epoch_counts = torch.zeros(self.learn.model.size[0] * self.learn.model.size[1])

    def after_fit(self, **kwargs):
        "Cleanup after training."
        self.__show = False
        self.epoch_counts = torch.zeros(self.learn.model.size[0] * self.learn.model.size[1])
        self.total_counts = torch.zeros(self.learn.model.size[0] * self.learn.model.size[1])
        self.fig = None
        self.ax = None

    def _update_plot(self, **kwargs):
        "Updates the plot."
        if not self.__show:
            self.fig.show()
            self.__show = True
        imsize = self.learn.model.size[:-1]
        if self.ax is None:
            self.ax = plt.imshow(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.show()
        else:
            self.ax.set_data(self.epoch_counts.view(imsize).cpu().numpy())
            self.fig.canvas.draw()


class SOM_TRAINING_VIZ(enum.Enum):
    """Enumerator class for SOM training visualization."""
    WEIGHTS_2D      = 'weights-2d'
    CODEBOOK_2D     = 'codebook-2d'
    WEIGHTS_3D      = 'weights-3d'
    CODEBOOK_3D     = 'codebook-3d'
    BMUS            = 'bmus'
    HYPERPARAMS     = 'hyperparams'


def get_visualization_callbacks(names: List[SOM_TRAINING_VIZ], use_epochs: bool = True) -> List[Callback]:
    """Maps names to visualization callbacks."""
    cbs = []
    if SOM_TRAINING_VIZ.WEIGHTS_2D in names or SOM_TRAINING_VIZ.CODEBOOK_2D in names:
        cbs.append(SomTrainingVisualizationCallback(is_3d=False))
    if SOM_TRAINING_VIZ.WEIGHTS_3D in names or SOM_TRAINING_VIZ.CODEBOOK_3D in names:
        cbs.append(SomTrainingVisualizationCallback(is_3d=True))
    if SOM_TRAINING_VIZ.BMUS in names:
        cbs.append(SomBmuVisualizationCallback(update_on_batch=not use_epochs))
    if SOM_TRAINING_VIZ.HYPERPARAMS in names:
        cbs.append(SomHyperparamsVisualizationCallback())
    return cbs
