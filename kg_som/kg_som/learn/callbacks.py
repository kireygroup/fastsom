"""
Callbacks for SOM.
"""

import numpy as np
import torch
from torch import Tensor
from fastprogress.fastprogress import progress_bar
from fastai.callback import Callback

from ..som import Som

__all__ = [
    "SomTrainingPhaseCallback",
]


def update_all_neigh_fn(v, sigma) -> Tensor:
    ""
    return torch.ones_like(v).to(device=v.device)



class SomTrainingPhaseCallback(Callback):
    "Callback for `SomLearner`, used to switch between rough and finetuning training phase parameters."

    def __init__(self, model: Som, init_method: str, finetune_epoch_pct: float = 0.4, lr: tuple = (0.09, 0.03)) -> None:
        self.model, self.finetune_epoch_pct, self.lr = model, finetune_epoch_pct, lr
        self._finetune_start = None     # First finetuning epoch
        self._rough_radiuses = None     # Rough training radiuses
        self._finetune_radiuses = None  # Finetune training radiuses
        self.init_method = init_method
        self._backup_neigh_fn = model.neigh_fn

    def _parameters(self):
        "Returns parameters for each training phase based on initialization method."
        if self.init_method == 'random':
            rough_radius_divider_start = 1.0
            rough_radius_divider_end = 6.0
            fine_radius_divider_start = 12.0
            fine_radius_divider_end = 25.0
            # rough_neigh_fn = update_all_neigh_fn
            rough_neigh_fn = self._backup_neigh_fn
        elif self.init_method.split('_')[0] == 'kmeans':
            rough_radius_divider_start = 8.0
            rough_radius_divider_end = 4.0
            fine_radius_divider_start = 36.0
            fine_radius_divider_end = np.inf  # force radius to 1
            rough_neigh_fn = self._backup_neigh_fn

        return rough_radius_divider_start, rough_radius_divider_end, fine_radius_divider_start, fine_radius_divider_end, rough_neigh_fn

    def on_train_begin(self, **kwargs):
        "Sets the finetune training iteration start, as well as the radius decays for the two phases."
        n_epochs = kwargs['n_epochs']
        # Retrieve parameters based on the codebook initialization method
        rough_radius_s, rough_radius_e, fine_radius_s, fine_radius_e, rough_neigh_fn = self._parameters()
        self._rough_neigh_fn = rough_neigh_fn
        # Save radiuses for each epoch into an array for both rough and finetune phases
        self._finetune_start = int(n_epochs * (1.0 - self.finetune_epoch_pct))
        self._rough_radiuses = self._get_radiuses(rough_radius_s, rough_radius_e, self._finetune_start)
        self._finetune_radiuses = self._get_radiuses(fine_radius_s, fine_radius_e, n_epochs - self._finetune_start)

    def on_epoch_begin(self, **kwargs):
        "Prints the current epoch & sets the epoch inside the SOM."
        epoch = kwargs['epoch']
        if epoch < self._finetune_start:
            # Use rough train parameters
            self.model.alpha = self.lr[0]
            self.model.sigma = self._rough_radiuses[epoch]
            self.model.neigh_fn = self._rough_neigh_fn
        else:
            # Use finetune train parameters
            self.model.alpha = self.lr[1]
            self.model.sigma = self._finetune_radiuses[epoch - self._finetune_start]
            self.model.neigh_fn = self._backup_neigh_fn

    def _get_radiuses(self, initial_div: float, end_div: float, n_epochs: int):
        "Calculates initial and final radius given map size and multipliers."
        map_max_dim = max(0.0 + self.model.weights.shape[0], 0.0 + self.model.weights.shape[1])
        initial_radius = max(1, np.ceil(map_max_dim / initial_div).astype(int))
        final_radius = max(1, np.ceil(initial_radius / end_div))
        return np.linspace(int(initial_radius), int(final_radius), num=n_epochs)
