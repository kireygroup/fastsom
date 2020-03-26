"""
"""
import torch
from typing import Optional, Callable, Collection, Union, Tuple
from functools import partial
from fastai.basic_train import Learner
from fastai.train import *
from fastai.callback import Callback, CallbackHandler
from fastai.basic_data import DataBunch
from fastai.core import Floats, defaults, is_listy
from fastai.torch_core import to_detach
from fastai.tabular import TabularDataBunch
from fastprogress.fastprogress import master_bar, progress_bar

from .callbacks import SomTrainingPhaseCallback
from .initializers import som_initializers
from .loss import cluster_loss
from .optim import SomOptimizer

from ..core import listify, ifnone
from ..datasets import UnsupervisedDataBunch
from ..interp import SomScatterVisualizer, SomStatsVisualizer
from ..som import Som


__all__ = [
    "SomLearner",
]


def to_unsupervised_databunch(self, **kwargs) -> UnsupervisedDataBunch:
    "Transforms a `TabularDataBunch` into an `UnsupervisedDataBunch`"
    train_ds = torch.cat([torch.cat([el[0].data[0].float(), el[0].data[1]]).unsqueeze(0) for el in [*self.train_ds]], dim=0)
    valid_ds = torch.cat([torch.cat([el[0].data[0].float(), el[0].data[1]]).unsqueeze(0) for el in [*self.valid_ds]], dim=0)
    bs = getattr(kwargs, 'bs', 20)
    return UnsupervisedDataBunch(train_ds, valid=valid_ds, bs=bs, **kwargs)


TabularDataBunch.to_unsupervised_databunch = to_unsupervised_databunch


class SomLearner(Learner):
    "`Learner` subclass for Self-Organizing Maps."

    def __init__(
            self, data: DataBunch,
            model: Som,
            opt_func: Callable = SomOptimizer,
            loss_func: Optional[Callable] = cluster_loss,
            metrics: Collection[Callable] = None,
            visualize: bool = False,
            callbacks: Collection[Callable] = None,
            init_weights: str = 'random',
            finetune_epoch_pct: float = 0.4,
            lr: Tuple[float, float] = (0.09, 0.03),
            **learn_kwargs):

        train_ds = data.train_ds.tensors[0] if hasattr(data.train_ds, 'tensors') else torch.tensor(data.train_ds, dtype=float)

        # Initialize the model weights
        initializer = som_initializers[init_weights]
        model.weights = initializer(train_ds, model.weights.shape)    

        # Setup additional mandatory callbacks
        additional_callbacks = [
            SomTrainingPhaseCallback(model, init_weights, finetune_epoch_pct=finetune_epoch_pct, lr=lr),
            # SomEarlyStoppingHelper(model),
        ]

        # Setup visualization callbacks
        if visualize:
            additional_callbacks += [
                SomScatterVisualizer(model, train_ds),
                SomStatsVisualizer(model, train_ds, plot_hyperparams=True, plot_stats=False),
            ]

        # Add user-defined callbacks
        callbacks = additional_callbacks if callbacks is None else callbacks + additional_callbacks

        super().__init__(
            data, model,
            opt_func=opt_func,
            loss_func=partial(loss_func, som=model) if loss_func is not None else None,
            metrics=metrics,
            callbacks=callbacks,
            **learn_kwargs
        )
