"""
This file contains Learners and Callbacks
used to train Self-Organizing Maps.

"""
from typing import Tuple, Optional, List
from torch import Tensor
from fastai.callback import Callback

from fastprogress.fastprogress import progress_bar

from .som import Som
from .init import som_initializers
from .viz import SomScatterVisualizer, SomStatsVisualizer

from ..core import ifnone
from ..datasets import UnsupervisedDataset


class SomLinearDecayHelper(Callback):
    "Callback for `SomLearner, used to linearly decrease training parameters."

    def __init__(self, model: Som, decay: float = 0.997) -> None:
        self.model, self.decay = model, decay

    def on_train_begin(self, **kwargs):
        self.model.lr = 1.0

    def on_epoch_begin(self, **kwargs):
        "Prints the current epoch & sets the epoch inside the SOM."
        self.model.lr = self.model.lr * self.decay                  # LR decreses after each epoch
        self.model.alpha = self.model.alpha * self.model.lr         # Alpha decreses after each epoch
        self.model.sigma = self.model.sigma * self.model.lr         # Sigma decreses after each epoch


class SomEarlyStoppingHelper(Callback):
    "Early stopping helper for Self-Organizing Maps."

    def __init__(self, model: Som, tol=0.0001) -> None:
        self.tol = tol
        self.model = model
        self.prev_weights = None

    def on_epoch_end(self, **kwargs):
        "Checks if training should stop."
        if self.prev_weights is not None:
            d = self.prev_weights - self.model.weights
            if d.max() < self.tol:
                raise KeyboardInterrupt(f'Early Stopping due to weight update below tolerance of {self.tol}')
        self.prev_weights = self.model.weights


class ProgressBarHelper(Callback):
    "Displays a progress bar that shows the epoch progress."

    def __init__(self, n_epochs: int) -> None:
        self.n_epochs = n_epochs
        self.epoch = -1
        self.pbar = progress_bar(range(n_epochs))

    def on_train_begin(self, **kwargs):
        self.pbar.update(0)

    def on_epoch_begin(self, **kwargs):
        self.epoch += 1
        self.pbar.update(self.epoch + 1)


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
        self.cbs['som-progress'] = ProgressBarHelper(epochs)
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
        self.model.weights = self.model._to_device(self.model.weights)
        # Epoch start
        for epoch in range(epochs):
            self._callback('on_epoch_begin', epoch=epoch)
            # Batch start
            for batch in range(max_batches):
                self._callback('on_batch_begin', batch=batch)
                self.model.training = True
                # Forward pass (find BMUs)
                self.model.forward(self.data.grab_batch())
                # Backward pass (update weights)
                self.model.backward(debug=debug)
                self._callback('on_batch_end', batch=batch)
            self._callback('on_epoch_end', epoch=epoch)
        self.model.training = False
        self._callback('on_train_end')

    def predict(self, x: Tensor) -> Tensor:
        "Runs model inference over `x`."
        self.model.training = False
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

    return SomLearner(data, model, **learn_kwargs)


__all__ = [
    "SomLearner",
    "som_learner",
    "create_som",
    "SomLinearDecayHelper",
    "ProgressBarHelper",
]
