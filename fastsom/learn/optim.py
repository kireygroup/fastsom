"""
Experimental.
Do not use!
"""
import numpy as np
import torch
from fastai.optimizer import Optimizer

__all__ = [
    "SplashOptimizer",
    "SomOptimizer",
]


class SplashOptimizer(Optimizer):
    "Optimizer used `zero_grad`. But, it failed!"

    def __init__(self, params, train_bn=True, **defaults):
        super().__init__([torch.tensor([0])], defaults)
        self.hypers = [{"lr": np.random.rand() * 100}]

    def __call__(self, *args, **kwargs):
        return

    def zero_grad(self):
        return

    def step(self, *args, **kwargs):
        return

    def set_hypers(self, *args, **kwargs):
        return


class SomOptimizer(Optimizer):
    "Optimizer used to update `alpha` and `sigma` params."

    def __init__(self, params, **defaults):
        defaults = dict(momentum=0.1)
        super().__init__(params, defaults)

    def __call__(self, *args, **kwargs):
        print(f"{self.__class__.__name__} has been called")

    def zero_grad(self):
        return

    def step(self, closure=None):
        return
