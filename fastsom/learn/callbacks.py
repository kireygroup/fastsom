"""
Callbacks for SOM.
"""
import numpy as np
import torch
from fastai.callback import Callback

from ..som import Som

__all__ = [
    "SomTrainer",
    "LinearDecaySomTrainer",
    "TwoPhaseSomTrainer",
    "ExperimentalSomTrainer",
]


class SomTrainer(Callback):
    """Base class for SOM training strategies."""

    def __init__(self, model: Som):
        pass


class LinearDecaySomTrainer(SomTrainer):
    """
    Linear decay for Self-Organizing Maps.

    Downscales both LR and radius based on the current and total epoch count.
    More stable than other approaches, but usually requires longer training.

    Parameters
    ----------
    model : Som
        The SOM model
    """

    def __init__(self, model: Som):
        self.model = model
        self.alpha = self.model.alpha.cpu().numpy()
        self.sigma = self.model.sigma.cpu().numpy()
        self.n_epochs = None

    def on_train_begin(self, **kwargs):
        self.n_epochs = kwargs['n_epochs']

    def on_epoch_begin(self, **kwargs):
        epoch = kwargs['epoch']
        decay = 1.0 - epoch / self.n_epochs
        self.model.alpha = self.alpha * decay
        self.model.sigma = self.sigma * decay


class TwoPhaseSomTrainer(SomTrainer):
    """
    Rough training / fine tuning trainer for Self-Organizing Maps.

    Divides training in two phases:

     - Phase 1 (50% epochs): LR max -> max/10, radius max -> max/6
     - Phase 2 (50% epochs): LR max/20 -> max/100, radius max/12 -> max/25

    Inspired by hyperparameter scaling done in https://github.com/sevamoo/SOMPY

    Parameters
    ----------
    model : Som
        The SOM model
    """

    def __init__(self, model: Som) -> None:
        self.model = model
        self.alpha = model.alpha.cpu().numpy()
        self.sigma = model.sigma.cpu().numpy()
        self.sigmas, self.alphas = [], []
        self.n_epochs = None

    def on_train_begin(self, **kwargs):
        # Initialize parameters for each epoch
        self.n_epochs = kwargs['n_epochs']
        # 50% rough training, 50% finetuning
        rough_pct = 0.5
        rough_epochs = int(rough_pct * self.n_epochs)
        finet_epochs = self.n_epochs - rough_epochs
        # Linear decaying radii for each phase
        rough_sigmas = np.linspace(self.sigma, max(self.sigma / 6.0, 1.0), num=rough_epochs)
        finet_sigmas = np.linspace(max(self.sigma / 12.0, 1.0), max(self.sigma / 25.0, 1.0), num=finet_epochs)
        # Linear decaying alpha
        rough_alphas = np.linspace(self.alpha, self.alpha / 10.0, num=rough_epochs)
        finet_alphas = np.linspace(self.alpha / 20.0, self.alpha / 100.0, num=finet_epochs)
        self.sigmas = np.concatenate([rough_sigmas, finet_sigmas], axis=0)
        self.alphas = np.concatenate([rough_alphas, finet_alphas], axis=0)

    def on_epoch_begin(self, **kwargs):
        # Update parameters
        epoch = kwargs['epoch']
        self.model.alpha = torch.tensor(self.alphas[epoch])
        self.model.sigma = torch.tensor(self.sigmas[epoch])


class ExperimentalSomTrainer(SomTrainer):
    """
    Experimental SOM training callback.

    Divides training in 3 phases with the following hyperparameter values*:

     - Phase 1 (15% epochs): LR max, radius max
     - Phase 2 (50% epochs): LR 1/2, radius max -> 1
     - Phase 3 (35% epochs): LR 1/6, radius = 1

    * arrows indicate start->end linear scaling

    Parameters
    ----------
    model : Som
        The SOM model
    """

    def __init__(self, model: Som):
        self.model = model
        self.alpha = self.model.alpha.cpu().numpy()
        self.sigma = self.model.sigma.cpu().numpy()
        self.alphas, self.sigmas = [], []

    def on_train_begin(self, **kwargs):
        n_epochs = kwargs['n_epochs']
        phase_1_iters = int(round(n_epochs * 0.16))
        phase_2_iters = int(round(n_epochs * 0.5))
        phase_3_iters = int(round(n_epochs * 0.34))

        alphas_1 = np.linspace(self.alpha, self.alpha, num=phase_1_iters)
        alphas_2 = np.linspace(self.alpha / 2, self.alpha / 2, num=phase_2_iters)
        alphas_3 = np.linspace(self.alpha / 6, self.alpha / 6, num=phase_3_iters)

        sigmas_1 = np.linspace(self.sigma, self.sigma, num=phase_1_iters)
        sigmas_2 = np.linspace(self.sigma, 1.0, num=phase_2_iters)
        sigmas_3 = np.linspace(1.0, 1.0, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_2, sigmas_3], axis=0)

    def on_epoch_begin(self, **kwargs):
        epoch = kwargs['epoch']
        self.model.alpha = torch.tensor(self.alphas[epoch])
        self.model.sigma = torch.tensor(self.sigmas[epoch])
