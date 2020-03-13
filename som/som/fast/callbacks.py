"""
Callbacks for SOM.
"""

import numpy as np
from fastprogress.fastprogress import progress_bar
from fastai.callback import Callback
from .som import SomFast

__all__ = [
    "SomTrainingPhaseCallback",
]


class SomTrainingPhaseCallback(Callback):
    "Callback for `SomLearner`, used to switch between rough and finetuning training phase parameters."

    def __init__(self, model: SomFast, finetune_epoch_pct: float = 0.4, lr: tuple = (0.09, 0.03)) -> None:
        self.model, self.finetune_epoch_pct, self.lr = model, finetune_epoch_pct, lr
        self._finetune_start = None     # First finetuning epoch
        self._rough_radiuses = None     # Rough training radiuses
        self._finetune_radiuses = None  # Finetune training radiuses

    def on_train_begin(self, **kwargs):
        "Sets the finetune training iteration start, as well as the radius decays for the two phases."
        n_epochs = kwargs['n_epochs']
        self._finetune_start = int(n_epochs * (1.0 - self.finetune_epoch_pct))
        self._rough_radiuses = self._get_radiuses(3.0, 6.0, self._finetune_start)
        self._finetune_radiuses = self._get_radiuses(12.0, 25.0, n_epochs - self._finetune_start)

    def on_epoch_begin(self, **kwargs):
        "Prints the current epoch & sets the epoch inside the SOM."
        epoch = kwargs['epoch']
        if epoch < self._finetune_start:
            # Use rough train parameters
            self.model.alpha = self.lr[0]
            self.model.sigma = self._rough_radiuses[epoch]
        else:
            # Use finetune train parameters
            self.model.alpha = self.lr[1]
            self.model.sigma = self._finetune_radiuses[epoch - self._finetune_start]

    def _get_radiuses(self, initial_div: float, end_div: float, n_epochs: int):
        "Calculates initial and final radius given map size and multipliers."
        map_max_dim = max(0.0 + self.model.weights.shape[0], 0.0 + self.model.weights.shape[1]) / 2
        initial_radius = max(1, np.ceil(map_max_dim / initial_div).astype(int))
        final_radius = max(1, np.ceil(initial_radius / end_div))
        return np.linspace(int(initial_radius), int(final_radius), num=n_epochs)
