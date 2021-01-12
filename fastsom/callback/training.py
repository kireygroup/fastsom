"""
Callbacks for SOM.
"""
import numpy as np
import torch
from fastai.callback.core import Callback

__all__ = [
    "SomTrainer",
    "LinearDecaySomTrainer",
    "TwoPhaseSomTrainer",
    "ExperimentalSomTrainer",
]


class SomTrainer(Callback):
    """Base class for SOM training strategies."""
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

    def before_fit(self, **kwargs):
        self.initial_alpha = self.initial_alpha if hasattr(self, 'initial_alpha') else self.learn.model.alpha
        self.initial_sigma = self.initial_sigma if hasattr(self, 'initial_sigma') else self.learn.model.sigma
        self.alpha = self.initial_alpha.clone()
        self.sigma = self.initial_sigma.clone()

    def before_train(self, **kwargs):
        decay = 1.0 - self.learn.epoch / self.learn.n_epoch
        self.model.alpha = self.alpha * decay
        self.model.sigma = self.sigma * decay
        # self.logger.debug(f'alpha: {self.learn.model.alpha}; sigma: {self.learn.model.sigma}')


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

    def before_fit(self, **kwargs):
        # Initialize parameters for each epoch
        self.initial_alpha = self.initial_alpha if hasattr(self, 'initial_alpha') else self.learn.model.alpha
        self.initial_sigma = self.initial_sigma if hasattr(self, 'initial_sigma') else self.learn.model.sigma
        self.alpha = self.initial_alpha.clone()
        self.sigma = self.initial_sigma.clone()
        self.sigmas, self.alphas = [], []
        # 50% rough training, 50% finetuning
        rough_pct = 0.5
        rough_epochs = int(rough_pct * self.learn.n_epoch)
        finet_epochs = self.learn.n_epoch - rough_epochs
        # Linear decaying radii for each phase
        rough_sigmas = np.linspace(self.sigma, max(self.sigma / 6.0, 1.0), num=rough_epochs)
        finet_sigmas = np.linspace(max(self.sigma / 12.0, 1.0), max(self.sigma / 25.0, 1.0), num=finet_epochs)
        # Linear decaying alpha
        rough_alphas = np.linspace(self.alpha, self.alpha / 10.0, num=rough_epochs)
        finet_alphas = np.linspace(self.alpha / 20.0, self.alpha / 100.0, num=finet_epochs)
        self.sigmas = np.concatenate([rough_sigmas, finet_sigmas], axis=0)
        self.alphas = np.concatenate([rough_alphas, finet_alphas], axis=0)

    def before_train(self, **kwargs):
        # Update parameters
        self.learn.model.alpha = torch.tensor(self.alphas[self.learn.epoch])
        self.learn.model.sigma = torch.tensor(self.sigmas[self.learn.epoch])
        # self.logger.debug(f'alpha: {self.learn.model.alpha}; sigma: {self.learn.model.sigma}')


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
    alphas = []
    sigmas = []

    def before_fit(self, **kwargs):
        self.initial_alpha = self.initial_alpha if hasattr(self, 'initial_alpha') else self.learn.model.alpha
        self.initial_sigma = self.initial_sigma if hasattr(self, 'initial_sigma') else self.learn.model.sigma
        self.alpha = self.initial_alpha.clone()
        self.sigma = self.initial_sigma.clone()
        self.bs = []

        phase_1_iters = int(round(self.learn.n_epoch * 0.16))
        phase_2_iters = int(round(self.learn.n_epoch * 0.50))
        phase_3_iters = int(round(self.learn.n_epoch * 0.34))

        alphas_1 = np.linspace(self.alpha, self.alpha, num=phase_1_iters)
        alphas_2 = np.linspace(self.alpha / 2, self.alpha / 2, num=phase_2_iters)
        alphas_3 = np.linspace(self.alpha / 6, self.alpha / 6, num=phase_3_iters)

        sigmas_1 = np.linspace(self.sigma, self.sigma, num=phase_1_iters)
        sigmas_2 = np.linspace(self.sigma, 1.0, num=phase_2_iters)
        sigmas_3 = np.linspace(1.0, 1.0, num=phase_3_iters)

        self.alphas = np.concatenate([alphas_1, alphas_2, alphas_3], axis=0)
        self.sigmas = np.concatenate([sigmas_1, sigmas_2, sigmas_3], axis=0)

        bs_1 = [self.learn.dls.bs for _ in range(phase_1_iters)]
        bs_2 = [max([8, self.learn.dls.bs // 2]) for _ in range(phase_2_iters)]
        bs_3 = [max([1, self.learn.dls.bs // 6]) for _ in range(phase_3_iters)]

        self.bs = np.concatenate([bs_1, bs_2, bs_3], axis=0).astype(int)

    def before_train(self, **kwargs):
        self.learn.model.alpha = torch.tensor(self.alphas[self.learn.epoch])
        self.learn.model.sigma = torch.tensor(self.sigmas[self.learn.epoch])
        # self.logger.debug(f'alpha: {self.learn.model.alpha}; sigma: {self.learn.model.sigma}')
