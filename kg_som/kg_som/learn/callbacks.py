"""
Callbacks for SOM.
"""

import numpy as np
import torch
from torch import Tensor
from fastprogress.fastprogress import progress_bar
from fastai.callback import Callback
from fastai.basic_train import Learner
from typing import Union, Tuple, Collection

from ..som import Som, neigh_gauss, neigh_square
from ..core import ifindict

__all__ = [
    "SomTrainer",
    "TwoPhaseSomTrainer",
    "ExperimentalSomTrainer",
    "LinearDecaySomTrainer",
    "OneCycleEmulatorSomTrainer",
]


def update_all_neigh_fn(v, sigma) -> Tensor:
    ""
    return torch.ones_like(v).to(device=v.device)


class SomTrainer(Callback):
    "Base class for SOM train strategy callbacks."
    @classmethod
    def from_model(cls, model: Som, init_method: str, lr: Collection[float], *args, **kwargs):
        return cls(model, init_method, lr, *args, **kwargs)


class TwoPhaseSomTrainer(SomTrainer):
    "Callback for `SomLearner`, used to switch between rough and finetuning training phase parameters."

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model, self.init_method, self.lr = model, init_method, lr
        self.finetune_epoch_pct = ifindict(kwargs, 'finetune_epoch_pct', 0.4)
        self._finetune_start = None                     # First finetuning epoch
        self._rough_radiuses = None                     # Rough training radiuses
        self._finetune_radiuses = None                  # Finetune training radiuses
        self._backup_neigh_fn = model.neigh_fn

    def _parameters(self):
        "Returns parameters for each training phase based on initialization method."
        if self.init_method == 'random':
            rough_radius_divider_start = 1.0
            rough_radius_divider_end = 6.0
            fine_radius_divider_start = 12.0
            fine_radius_divider_end = 25.0
            # neigh_fn = update_all_neigh_fn
            rough_neigh_fn = neigh_gauss
            fine_neigh_fn = neigh_gauss
        
        elif self.init_method.split('_')[0] == 'kmeans':
            rough_radius_divider_start = 8.0
            rough_radius_divider_end = 4.0
            fine_radius_divider_start = 36.0
            fine_radius_divider_end = np.inf  # force radius to 1
            rough_neigh_fn = neigh_gauss
            fine_neigh_fn = neigh_gauss

        return rough_radius_divider_start, rough_radius_divider_end, rough_neigh_fn, fine_radius_divider_start, fine_radius_divider_end, fine_neigh_fn

    def on_train_begin(self, **kwargs):
        "Sets the finetune training iteration start, as well as the radius decays for the two phases."
        n_epochs = kwargs['n_epochs']
        # Retrieve parameters based on the codebook initialization method
        rough_radius_s, rough_radius_e, rough_neigh_fn, fine_radius_s, fine_radius_e, fine_neigh_fn = self._parameters()

        self._rough_neigh_fn = rough_neigh_fn
        self._fine_neigh_fn = fine_neigh_fn
        # Save radiuses for each epoch into an array for both rough and finetune phases
        self._finetune_start = int(n_epochs * (1.0 - self.finetune_epoch_pct))
        self._rough_radiuses = self._get_radiuses(rough_radius_s, rough_radius_e, self._finetune_start)
        self._finetune_radiuses = self._get_radiuses(fine_radius_s, fine_radius_e, n_epochs - self._finetune_start)

    def on_epoch_begin(self, **kwargs):
        "Updates hyperparameters."
        epoch = kwargs['epoch']
        if epoch < self._finetune_start:
            # Use rough train parameters
            self.model.alpha = torch.tensor(self.lr[0])
            self.model.sigma = torch.tensor(self._rough_radiuses[epoch])
            # self.model.neigh_fn = self._rough_neigh_fn
        else:
            # Use finetune train parameters
            self.model.alpha = torch.tensor(self.lr[1])
            self.model.sigma = torch.tensor(self._finetune_radiuses[epoch - self._finetune_start])
            # self.model.neigh_fn = self._fine_neigh_fn

    def _get_radiuses(self, initial_div: float, end_div: float, n_epochs: int):
        "Calculates initial and final radius given map size and multipliers."
        map_max_dim = max(0.0 + self.model.weights.shape[0], 0.0 + self.model.weights.shape[1])
        initial_radius = max(1, np.ceil(map_max_dim / initial_div).astype(int))
        final_radius = max(1, np.ceil(initial_radius / end_div))
        return np.linspace(int(initial_radius), int(final_radius), num=n_epochs)


class ExperimentalSomTrainer(SomTrainer):
    "SOM Trainer callback "

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model, self.init_method, self.lr = model, init_method, lr
        self.min_lr, self.mid_lr, self.max_lr = lr[0], lr[1], lr[2]
        self.max_sigma = np.max(model.size[:-1]) / 2
        self.min_sigma = 1.0
        self.epoch, self.n_epochs = -1, -1
        self.alphas = []
        self.sigmas = []
        self.neigh_fns = []
        self.iter = 0
        self.update_on_batch = ifindict(kwargs, 'update_on_batch', False)

    def on_train_begin(self, **kwargs):
        self.n_epochs = kwargs['n_epochs']
        self.iter = 0
        iterations = len(self.learn.data.train_dl) * self.n_epochs if self.update_on_batch else self.n_epochs
        phase_1_iters = int(iterations * 0.4)
        phase_2_iters = int(iterations * 0.2)
        phase_3_iters = int(iterations * 0.4)

        # alphas_1 = np.linspace(self.min_lr, self.min_lr, num=phase_1_iters)
        # alphas_2 = np.linspace(self.min_lr, self.mid_lr, num=phase_2_iters)
        # alphas_3 = np.linspace(self.mid_lr, self.max_lr, num=phase_3_iters)

        alphas_1 = [self.lr[0] for _ in range(phase_1_iters)]
        alphas_2 = [self.lr[1] for _ in range(phase_2_iters)]
        alphas_3 = [self.lr[2] for _ in range(phase_3_iters)]

        sigmas_1 = np.linspace(self.max_sigma, self.max_sigma, num=phase_1_iters)
        sigmas_2 = np.linspace(self.max_sigma, self.min_sigma, num=phase_2_iters)
        sigmas_3 = np.linspace(self.min_sigma, self.min_sigma, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_2, sigmas_3], axis=0)

        phase_1_neighs = [neigh_gauss for _ in range(phase_1_iters)]
        phase_2_neighs = [neigh_square for _ in range(phase_2_iters)]
        phase_3_neighs = [neigh_square for _ in range(phase_3_iters)]

        self.neigh_fns = phase_1_neighs + phase_2_neighs + phase_3_neighs

    def on_batch_begin(self, **kwargs):
        if self.update_on_batch:
            self.step()

    def on_epoch_begin(self, **kwargs):
        if not self.update_on_batch:
            self.model.alpha = torch.tensor(self.alphas[self.iter])
            self.model.sigma = torch.tensor(self.sigmas[self.iter])
            # self.model.neigh_fn = self.neigh_fns[self.iter]
            self.iter += 1


class LinearDecaySomTrainer(SomTrainer):
    "Training callback for self-organizing maps. Updates alpha and sigma parameters with linear correction."

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model = model
        self.max_lr = torch.tensor(lr) if isinstance(lr, float) else torch.tensor(lr[0])
        self.sigma = torch.tensor(np.max(model.size[:-1])) / 2
        self.n_epochs = -1

    def on_train_begin(self, **kwargs):
        "Saves the epoch count."
        self.n_epochs = kwargs['n_epochs']

    def on_epoch_begin(self, **kwargs):
        "Updates hyperparameters."
        epoch = kwargs['epoch']
        decay = 1.0 - epoch / self.n_epochs
        self.model.alpha = self.max_lr * decay
        self.model.sigma = self.sigma * decay


class OneCycleEmulatorSomTrainer(SomTrainer):
    "Training callback that emulates the One Cycle LR growth."

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model = model
        self.max_lr = lr[0]
        self.min_lr = lr[1]
        self.max_sigma = np.max(model.size[:-1])
        self.min_sigma = 1.0
        self.epoch, self.n_epochs = -1, -1
        self.alphas = []
        self.sigmas = []
        self.iter = 0
        self.update_on_batch = ifindict(kwargs, 'update_on_batch', False)

    def on_train_begin(self, **kwargs):
        self.n_epochs = kwargs['n_epochs']
        self.iter = 0
        iterations = len(self.learn.data.train_dl) * self.n_epochs if self.update_on_batch else self.n_epochs
        phase_1_iters = int(iterations * 0.35)
        phase_2_iters = phase_1_iters
        phase_3_iters = iterations - 2 * phase_1_iters

        alphas_1 = np.linspace(self.min_lr, self.max_lr, num=phase_1_iters)
        alphas_2 = np.linspace(self.max_lr, self.min_lr, num=phase_2_iters)
        alphas_3 = np.linspace(self.min_lr, self.min_lr / 100, num=phase_3_iters)

        sigmas_1 = np.linspace(self.max_sigma, self.max_sigma, num=phase_1_iters + phase_2_iters)
        sigmas_3 = np.linspace(self.max_sigma, self.min_sigma, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_3], axis=0)

    def on_batch_begin(self, **kwargs):
        self.epoch = kwargs['epoch']
        self.model.alpha = torch.tensor(self.alphas[self.iter])
        self.model.sigma = torch.tensor(self.sigmas[self.iter])
        self.iter += 1
