"""
Experimental.
Do not use
"""
from torch import Tensor
from torch.optim import Optimizer
from fastai.callback import OptimWrapper

__all__ = [
    "SplashOptimizer",
    "SomOptimizer",
]


class SplashOptimizer(Optimizer):
    "Optimizer used `zero_grad`. But, it failed!"

    def __init__(self, params):
        defaults = dict(momentum=0.1, lr=0.1)
        super(SplashOptimizer, self).__init__(params, defaults)

    def __call__(self, *args, **kwargs):
        return

    def zero_grad(self):
        return

    def step(self, closure):
        return


class SomOptimizer(Optimizer):
    "Optimizer used to update `alpha` and `sigma` params."

    def __init__(self, params, **defaults):
        defaults = dict(momentum=0.1)
        super().__init__(params, defaults)

    def __call__(self, *args, **kwargs):
        print(f'{self.__class__.__name__} has been called')

    def zero_grad(self):
        return

    def step(self, closure=None):
        return
        # print(self.param_groups)
        # print(self.defaults)
        # alpha, sigma, w = self.param_groups
        # alpha = alpha / self.defaults.lr
        # sigma = sigma / self.defaults.lr
