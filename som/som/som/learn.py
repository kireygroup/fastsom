"""
This file contains Learners and Callbacks
used to train Self-Organizing Maps.

"""
from typing import Tuple, Optional, List
from torch import Tensor
from fastai.callback import Callback
from fastai.basic_train import Learner
from fastai.train import *

from .som import Som
from .init import som_initializers
from .viz import SomScatterVisualizer, SomStatsVisualizer
from .callbacks import ProgressBarHelper, SomLinearDecayHelper, SomEarlyStoppingHelper

from ..core import ifnone
from ..datasets import UnsupervisedDataset


__all__ = [
    "SomLearner",
    "som_learner",
    "create_som",
]


def create_som(data: UnsupervisedDataset, map_size: Tuple[int, int] = (10, 10), **model_kwargs) -> Som:
    "Creates a new SOM of the given size."
    in_size = data.train.shape[-1]
    return Som((*map_size, in_size), **model_kwargs)


class SomLearner():
    "Handles the training loop for a Som subclass."

    def __init__(self, data: UnsupervisedDataset, model: Som, cbs: Optional[List[Callback]] = None) -> None:
        self.data = data
        self.model = model
        self.cbs = dict(ifnone(cbs, []))

    def fit(self, epochs: int, visualize: bool = True, visualize_dim: int = 2, max_batches: int = 100, plot_hyperparams: bool = False, debug: bool = False) -> None:
        "Trains the model for `epochs` epochs, optionally visualizing what's going on"
        # Initialize callbacks
        self.cbs['som-progress'] = ProgressBarHelper()
        self.cbs['som-decay'] = SomLinearDecayHelper(self.model)
        self.cbs['som-early-stop'] = SomEarlyStoppingHelper(self.model)
        if visualize:
            self.cbs['som-stats'] = SomStatsVisualizer(self, plot_hyperparams=plot_hyperparams)
            self.cbs['som-scatter-viz'] = SomScatterVisualizer(self.model, self.data.train, dim=visualize_dim)
        bs = self.data.bs
        x = self.data.train

        # Calculate number of iterations (optionally setting a max if random batching is enabled)
        max_batches = max_batches if self.data.random_batching else 1 if bs is None or bs >= x.shape[0] else x.shape[0] // bs

        # Train start
        self._callback('on_train_begin', n_epochs=epochs)
        self.model.to_device()
        # Epoch start
        try:
            for epoch in range(epochs):
                self._callback('on_epoch_begin', epoch=epoch)
                # Batch start
                for batch in range(max_batches):
                    self._callback('on_batch_begin', batch=batch)
                    self.model.train()
                    # Forward pass (find BMUs)
                    self.model.forward(self.data.grab_batch())
                    # Backward pass (update weights)
                    self.model.backward(debug=debug)
                    self._callback('on_batch_end', batch=batch)
                self._callback('on_epoch_end', epoch=epoch)
            self.model.eval()
            self._callback('on_train_end')
        except KeyboardInterrupt as e:
            print(e)

    def predict(self, x: Tensor) -> Tensor:
        "Runs model inference over `x`."
        self.model.eval()
        return self.model.forward(self.model._to_device(x))

    def _callback(self, callback_method: str, **kwargs):
        "Invokes `callback_method` on each callback."
        for callback in self.cbs.values():
            getattr(callback, callback_method)(**kwargs)


def som_learner(data: UnsupervisedDataset,
                map_size: Tuple[int, int] = (10, 10),
                init: str = 'kmeans_euclidean',
                model: Optional[Som] = None,
                alpha=0.3,
                **learn_kwargs) -> SomLearner:
    "Creates a SOM Learner (and its model, if not provided)."
    model = model if model is not None else create_som(data, map_size=map_size, alpha=alpha)
    data.normalize()
    if init is not None:
        model.weights = som_initializers[init](data.train, (*map_size, data.train.shape[-1]))
    # TODO initialize model parameters depending on init method

    return SomLearner(data, model, **learn_kwargs)
