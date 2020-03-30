"""
"""
import torch
from torch.utils.data import Dataset
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
    "tabular_ds_to_lists",
]


def tabular_ds_to_lists(ds: Dataset):
    x_cat = torch.cat([el[0].data[0].long().unsqueeze(0) for el in ds], dim=0)
    x_cont = torch.cat([el[0].data[1].float().unsqueeze(0) for el in ds], dim=0)
    return x_cat, x_cont


def to_unsupervised_databunch(self, bs: Optional[int] = None, **kwargs) -> UnsupervisedDataBunch:
    "Transforms a `TabularDataBunch` into an `UnsupervisedDataBunch`"
    train_x_cat, train_x_cont = tabular_ds_to_lists(self.train_ds)
    valid_x_cat, valid_x_cont = tabular_ds_to_lists(self.valid_ds)

    train_ds = torch.cat([train_x_cat.float(), train_x_cont], dim=1) if len(self.train_ds) > 0 else None
    # valid_ds = torch.cat([valid_x_cat.float(), valid_x_cont], dim=1) if len(self.valid_ds) > 0 else None
    valid_ds = torch.tensor([])
   
    bs = ifnone(bs, self.batch_size)
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
