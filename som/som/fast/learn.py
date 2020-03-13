"""
"""
from typing import Optional, Callable, Collection, Union
from functools import partial
from fastai.basic_train import Learner
from fastai.train import *
from fastai.callback import Callback, CallbackHandler
from fastai.basic_data import DataBunch
from fastai.core import Floats, defaults, is_listy
from fastai.torch_core import to_detach
from fastprogress.fastprogress import master_bar, progress_bar

from .optim import SomOptimizer
from .loss import cluster_loss
from .callbacks import SomTrainingPhaseCallback
from ..som.som import Som
from ..som.callbacks import SomLinearDecayHelper, SomEarlyStoppingHelper
from ..som.viz import SomScatterVisualizer, SomStatsVisualizer
from ..core import listify, ifnone
from ..som.init import RandomInitializer


__all__ = [
    "SomLearnerFast"
]


class SomLearnerFast(Learner):
    "`Learner` subclass for Self-Organizing Maps."

    def __init__(
            self, data: DataBunch, model: Som,
            opt_func: Callable = SomOptimizer,
            loss_func: Optional[Callable] = cluster_loss,
            metrics: Collection[Callable] = None,
            visualize: bool = False,
            callbacks: Collection[Callable] = None,
            **learn_kwargs):
        self.visualize = visualize
        additional_callbacks = [
            SomTrainingPhaseCallback(model),
            SomEarlyStoppingHelper(model),
        ]
        if visualize:
            additional_callbacks += [
                SomScatterVisualizer(model, data.train_ds.tensors[0]),
                SomStatsVisualizer(model, data.train_ds.tensors[0], plot_hyperparams=True),
            ]
        callbacks = additional_callbacks if callbacks is None else callbacks + additional_callbacks
        model.weights = RandomInitializer()(data.train_ds.tensors[0], model.weights.shape)
        super().__init__(
            data, model,
            opt_func=opt_func,
            loss_func=partial(loss_func, som=model) if loss_func is not None else None,
            metrics=metrics,
            callbacks=callbacks,
            **learn_kwargs
        )

    # def fit(self, epochs: int, lr: Union[Floats, slice] = defaults.lr,
    #         wd: Floats = None, callbacks: Collection[Callback] = None,) -> None:
    #     lr = self.lr_range(lr)
    #     print(lr)
    #     super(Learner, self).fit(epochs, lr=lr, wd=wd, callbacks=callbacks)
