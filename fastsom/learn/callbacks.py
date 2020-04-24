"""
Callbacks for SOM.
"""
import numpy as np
import torch
from torch import Tensor
from fastai.callback import Callback
from typing import Collection, Callable, Dict
from functools import partial

from ..som import Som, neigh_gauss
from ..core import ifindict

__all__ = [
    "SomTrainer",
    "TwoPhaseSomTrainer",
    "ExperimentalSomTrainer",
    "LinearDecaySomTrainer",
    "OneCycleEmulatorSomTrainer",
    "SomEarlyStoppingCallback",
    "lr_split",
]


def lr_split(lr: Collection[float], lr_phases: Collection[float]) -> Collection[float]:
    """

    """
    if len(lr) >= len(lr_phases):
        return lr[:len(lr_phases)]
    else:
        last_lr = lr[-1]
        return lr + [last_lr * lr_phase_mult for lr_phase_mult in lr_phases[len(lr):]]


def update_all_neigh_fn(v, sigma) -> Tensor:
    """Experimental. Updates the whole SOM map with multiplier = 1."""
    return torch.ones_like(v).to(device=v.device)


class SomTrainer(Callback):
    """Base class for SOM training strategy callbacks."""

    def __init__(self):
        self.lr_phases = 1

    @classmethod
    def from_model(cls, model: Som, init_method: str, lr: Collection[float], *args, **kwargs):
        """Creates a new instance from model."""
        return cls(model, init_method, lr, *args, **kwargs)


class TwoPhaseSomTrainer(SomTrainer):
    """
    Callback for `SomLearner`, used to switch between rough and finetuning training phase parameters.

    This was heavily inspired by the training strategy used in https://github.com/sevamoo/SOMPY.
    """

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model, self.init_method = model, init_method
        self.finetune_epoch_pct = ifindict(kwargs, 'finetune_epoch_pct', 0.4)
        self._finetune_start = None                     # First finetuning epoch
        self._rough_radiuses = None                     # Rough training radiuses
        self._finetune_radiuses = None                  # Finetune training radiuses
        self._backup_neigh_fn = model.neigh_fn
        lr_phases = [1, 0.2]
        self.lr = lr_split(lr, lr_phases)

    def _parameters(self):
        """Returns parameters for each training phase based on initialization method."""
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
        """Sets the finetune training iteration start, as well as the radius decays for the two phases."""
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
        """Updates hyperparameters."""
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
    """
    Experimental SOM callback for hyperparameter scaling.
    Seems to work well in practice when compared with traditional linear decay.

    Parameters
    ----------
    model : Som
        The SOM model.
    init_method : str
        The weight initialization strategy to be used. Available values are 'random' and 'kmeans'.
    lr : Collection[float]
        A collection of LR values. For this Trainer, it should contain three values, one for each phase.
    update_on_batch : bool default=False
        Whether to update hyperparameters on each batch. If false, update is performed on each epoch.
    """

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model, self.init_method = model, init_method
        self.max_sigma = np.max(model.size[:-1]) / 2
        self.min_sigma = 1.0
        self.epoch, self.n_epochs = -1, -1
        self.alphas = []
        self.sigmas = []
        self.iter = 0
        self.update_on_batch = ifindict(kwargs, 'update_on_batch', False)
        lr_phases = [1, 0.5, 0.3]
        self.lr = lr_split(lr, lr_phases)

    def on_train_begin(self, **kwargs):
        """Setup per-iteration values based on the number of epochs."""
        self.n_epochs = kwargs['n_epochs']
        self.iter = 0
        # Setup iterations based on either epochs or batches, as configured
        iterations = len(self.learn.data.train_dl) * self.n_epochs if self.update_on_batch else self.n_epochs
        phase_1_iters = int(round(iterations * 0.16))
        phase_2_iters = int(round(iterations * 0.5))
        phase_3_iters = int(round(iterations * 0.34))

        alphas_1 = [self.lr[0] for _ in range(phase_1_iters)]
        alphas_2 = [self.lr[1] for _ in range(phase_2_iters)]
        alphas_3 = [self.lr[2] for _ in range(phase_3_iters)]

        sigmas_1 = np.linspace(self.max_sigma, self.max_sigma, num=phase_1_iters)
        sigmas_2 = np.linspace(self.max_sigma, self.min_sigma, num=phase_2_iters)
        sigmas_3 = np.linspace(self.min_sigma, self.min_sigma, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_2, sigmas_3], axis=0)

    def step(self):
        """Advances one step in the training schedule."""
        self.model.alpha = torch.tensor(self.alphas[self.iter])
        self.model.sigma = torch.tensor(self.sigmas[self.iter])
        self.iter += 1

    def on_batch_begin(self, **kwargs):
        """Advances one step if the trainer is configured to update on each batch."""
        if self.update_on_batch:
            self.step()

    def on_epoch_begin(self, **kwargs):
        """Advances one step if the trainer is configured to update on each epoch."""
        if not self.update_on_batch:
            self.step()


class LinearDecaySomTrainer(SomTrainer):
    """
    Training callback for self-organizing maps. Updates alpha and sigma parameters with linear correction.
    """

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model = model
        self.max_lr = torch.tensor(lr) if isinstance(lr, float) else torch.tensor(lr[0])
        self.sigma = torch.tensor(np.max(model.size[:-1])) / 2
        self.n_epochs = -1

    def on_train_begin(self, **kwargs):
        """Saves the epoch count."""
        self.n_epochs = kwargs['n_epochs']

    def on_epoch_begin(self, **kwargs):
        """Updates hyperparameters."""
        epoch = kwargs['epoch']
        decay = 1.0 - epoch / self.n_epochs
        self.model.alpha = self.max_lr * decay
        self.model.sigma = self.sigma * decay


class OneCycleEmulatorSomTrainer(SomTrainer):
    """Training callback that emulates the One Cycle LR growth."""

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model = model
        self.max_sigma = np.max(model.size[:-1])
        self.min_sigma = 1.0
        self.epoch, self.n_epochs = -1, -1
        self.alphas = []
        self.sigmas = []
        self.iter = 0
        self.update_on_batch = ifindict(kwargs, 'update_on_batch', False)
        lr_phases = [1, 0.3]
        self.lr = lr_split(lr, lr_phases)

    def on_train_begin(self, **kwargs):
        self.n_epochs = kwargs['n_epochs']
        self.iter = 0
        iterations = len(self.learn.data.train_dl) * self.n_epochs if self.update_on_batch else self.n_epochs
        phase_1_iters = int(iterations * 0.35)
        phase_2_iters = phase_1_iters
        phase_3_iters = iterations - 2 * phase_1_iters

        max_lr = self.lr[0]
        min_lr = self.lr[1]

        alphas_1 = np.linspace(min_lr, max_lr, num=phase_1_iters)
        alphas_2 = np.linspace(max_lr, min_lr, num=phase_2_iters)
        alphas_3 = np.linspace(min_lr, min_lr / 100, num=phase_3_iters)

        sigmas_1 = np.linspace(self.max_sigma, self.max_sigma, num=phase_1_iters + phase_2_iters)
        sigmas_3 = np.linspace(self.max_sigma, self.min_sigma, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_3], axis=0)

    def on_batch_begin(self, **kwargs):
        self.epoch = kwargs['epoch']
        self.model.alpha = torch.tensor(self.alphas[self.iter])
        self.model.sigma = torch.tensor(self.sigmas[self.iter])
        self.iter += 1


default_early_stopping_thresholds = dict(
    topologic_err=10.0,
    codebook_err=0.02,
)


class SomEarlyStoppingCallback(Callback):
    """Performs early stopping checks based on given metric-wise thresholds."""

    def __init__(self, thresh: Dict[str, float] = default_early_stopping_thresholds):
        self.metric_names = []
        self.thresh = thresh

    def _get_func_name(self, func: Callable) -> str:
        "Returns function name, optionally unwrapping partial function applications."
        while isinstance(func, partial):
            func = func.func
        if isinstance(func, Callable):
            return func.__name__
        raise ValueError('`func` is not a function')

    def on_train_begin(self, **kwargs):
        metrics = ifindict(kwargs, 'metrics', [])
        self.metric_names = [self._get_func_name(m) for m in metrics]

    def on_epoch_end(self, **kwargs):
        "Performs early stopping checks."
        last_metric_values = ifindict(kwargs, 'last_metrics', [])
        should_stop = [metric_value < self.thresh[metric_name] for metric_name, metric_value in zip(self.metric_names, last_metric_values)]

        return {
            'stop_training': all(should_stop)
        }


class UntilWhenSomTrainer(SomTrainer):
    """
    Experimental.
    """

    def __init__(self, model: Som, init_method: str, lr: Collection[float], *args, **kwargs) -> None:
        self.model, self.init_method, self.lr = model, init_method, lr
        self.finetune_epoch_pct = ifindict(kwargs, 'finetune_epoch_pct', 0.4)
        self._finetune_start = None                     # First finetuning epoch
        self._rough_radiuses = None                     # Rough training radiuses
        self._finetune_radiuses = None                  # Finetune training radiuses
        self._backup_neigh_fn = model.neigh_fn
