"""
This module contains various Learner Callbacks.
"""

from fastprogress.fastprogress import progress_bar
from fastai.callback import Callback
from .som import Som

__all__ = [
    "SomLinearDecayHelper",
    "SomEarlyStoppingHelper",
    "ProgressBarHelper",
]


class SomLinearDecayHelper(Callback):
    "Callback for `SomLearner`, used to linearly decrease training parameters."

    def __init__(self, model: Som, decay: float = 0.997) -> None:
        self.model, self.decay = model, decay

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
            if d.abs().max() < self.tol:
                raise KeyboardInterrupt(f'Early Stopping due to weight update below tolerance of {self.tol}')
        self.prev_weights = self.model.weights


class ProgressBarHelper(Callback):
    "Displays a progress bar that shows the epoch progress."

    def __init__(self) -> None:
        self.pbar = None

    def on_train_begin(self, **kwargs):
        self.pbar = progress_bar(range(kwargs['n_epochs']))

    def on_epoch_end(self, **kwargs):
        self.pbar.update(kwargs['epoch'])
