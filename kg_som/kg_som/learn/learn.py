"""
This module defines a Fastai `Learner` subclass used to train Self-Organizing Maps.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Collection, Union, Tuple, List, Type, Dict
from functools import partial
from fastai.basic_train import Learner
from fastai.train import *
from fastai.callback import Callback, CallbackHandler
from fastai.basic_data import DataBunch
from fastai.core import Floats, defaults, is_listy
from fastai.torch_core import to_detach
from fastprogress.fastprogress import master_bar, progress_bar

from .callbacks import SomTrainer, TwoPhaseSomTrainer
from .initializers import som_initializers
from .loss import SomLoss
from .optim import SomOptimizer

from ..core import listify, ifnone, setify
from ..datasets import UnsupervisedDataBunch
from ..interp import SomScatterVisualizer, SomStatsVisualizer, SomBmuVisualizer, mean_quantization_err
from ..som import Som


__all__ = [
    "SomLearner",
]


def visualization_callbacks(visualize: List[str], model: Som, *viz_args) -> List[Callback]:
    "Builds a list of visualization callbacks."
    cbs = []
    s_visualize = setify(visualize)

    if 'weights' in s_visualize:
        cbs.append(SomScatterVisualizer(model, *viz_args))
    if 'hyperparams' in s_visualize:
        cbs.append(SomStatsVisualizer(model, *viz_args, plot_hyperparams=True, plot_stats=False))
    if 'bmus' in s_visualize:
        cbs.append(SomBmuVisualizer(model))
    return cbs


class SomLearner(Learner):
    "`Learner` subclass for Self-Organizing Maps."

    def __init__(
            self, data: DataBunch,
            model: Som,
            visualize: List[str] = [],
            init_weights: str = 'random',
            lr: Collection[float] = (0.09, 0.03),
            metrics: Collection[Callable] = None,
            callbacks: Collection[Callable] = None,
            loss_func: Callable = mean_quantization_err,
            trainer: Type[SomTrainer] = TwoPhaseSomTrainer,
            trainer_args: Dict = dict(),
            opt_func: Callable = SomOptimizer,
            **learn_kwargs):

        train_ds = data.train_ds.tensors[0] if hasattr(data.train_ds, 'tensors') else torch.tensor(data.train_ds, dtype=float)

        # Initialize the model weights
        initializer = som_initializers[init_weights]
        model.weights = initializer(train_ds, model.weights.shape)

        # Add callbacks
        callbacks = ifnone(callbacks, [])
        callbacks += visualization_callbacks(visualize, model, train_ds)
        callbacks += [trainer.from_model(model, init_weights, lr, **trainer_args)]

        # Setup loss function
        loss_func = ifnone(loss_func, mean_quantization_err)
        # Wrap the loss function with SomLoss if needed
        loss_fn = loss_func if isinstance(loss_func, SomLoss) else SomLoss(loss_func, model)

        # Pass model reference to metrics
        metrics = list(map(lambda fn: partial(fn, som=model), metrics)) if metrics is not None else []

        super().__init__(
            data, model,
            opt_func=opt_func,
            loss_func=loss_fn,
            metrics=metrics,
            callbacks=callbacks,
            **learn_kwargs
        )

    # def fit_one_cycle(self, cyc_len:int, max_lr:float = 1.0,
    #                 moms:Tuple[float,float]=(0.95,0.85), div_factor:float=25., pct_start:float=0.3, final_div:float=None,
    #                 wd:float=None, callbacks:Optional[List[Callback]]=None, tot_epochs:int=None, start_epoch:int=None)->None:
    #     "Fit a model following the 1cycle policy."
    #     callbacks = listify(callbacks)
    #     callbacks.append(OneCyclePolicyEmulator(self, self.model, max_lr=max_lr))
    #     self.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)