"""
This module defines a Fastai `Learner` subclass used to train Self-Organizing Maps.
"""
import torch
import pandas as pd
import numpy as np

from typing import Optional, Callable, Collection, List, Type, Dict, Tuple
from functools import partial
from fastai.basic_train import Learner
from fastai.basic_data import DataBunch
from fastai.tabular import TabularDataBunch
from fastai.train import *
from fastai.callback import Callback

from .callbacks import SomTrainer, ExperimentalSomTrainer
from .loss import SomLoss
from .optim import SomOptimizer

from ..core import ifnone, setify, index_tensor, find
from ..datasets import get_xy, ToBeContinuousProc
from ..interp import SomTrainingViz, SomHyperparamsViz, SomBmuViz, mean_quantization_err
from ..som import Som


__all__ = [
    "SomLearner",
]


def visualization_callbacks(visualize: List[str], visualize_on: str, learn: Learner) -> List[Callback]:
    """Builds a list of visualization callbacks."""
    cbs = []
    visualize_on = ifnone(visualize_on, 'epoch')
    s_visualize = setify(visualize)

    if 'weights' in s_visualize:
        cbs.append(SomTrainingViz(learn, update_on_batch=(visualize_on == 'batch')))
    if 'hyperparams' in s_visualize:
        cbs.append(SomHyperparamsViz(learn))
    if 'bmus' in s_visualize:
        cbs.append(SomBmuViz(learn, update_on_batch=(visualize_on == 'batch')))
    return cbs


class UnifyDataCallback(Callback):
    def on_batch_begin(self, **kwargs):
        x_cat, x_cont = kwargs['last_input']
        return {'last_input': x_cont}


class SomLearner(Learner):
    """
    Learner subclass used to train Self-Organizing Maps.

    All keyword arguments not listed below are forwarded to the `Learner` parent class.

    Parameters
    ----------
    data : UnsupervisedDataBunch
        Contains train and validations datasets, along with sampling and normalization utils.
    model : Som default=None
        The Self-Organizing Map model.
    size : Tuple[int, int] default=(10, 10)
        The map size to use if `model` is None.
    lr : float
        The learning rate to be used for training.
    trainer : Type[SomTrainer] default=ExperimentalSomTrainer
        The class that should be used to define SOM training behaviour such as hyperparameter scaling.
    callbacks : Collection[Callback] default=None
        A list of custom Fastai Callbacks.
    loss_func : Callable default=mean_quantization_err
        The loss function (actually a metric, since SOMs are unsupervised)
    metrics : Collection[Callable] default=None
        A list of metric functions to be evaluated after each iteration.
    visualize : List[str] default=[]
        A list of elements to be visualized while training. Available values are 'weights', 'hyperparams' and 'bmus'.
    visualize_on: str default='epoch'
        Determines when visualizations should be updated ('batch' / 'epoch').
    init_weights : str default='random'
        SOM weight initialization strategy. Defaults to random sampling in the train dataset space.
    """

    def __init__(
            self,
            data: DataBunch,
            model: Som = None,
            size: Tuple[int, int] = (10, 10),
            lr: float = 0.6,
            trainer: Type[SomTrainer] = ExperimentalSomTrainer,
            callbacks: List[Callback] = [],
            loss_func: Callable = mean_quantization_err,
            metrics: Collection[Callable] = None,
            visualize: List[str] = [],
            visualize_on: str = 'epoch',
            **learn_kwargs
    ) -> None:
        x, _ = get_xy(data)
        n_features = x.shape[-1]
        # Create a new Som using the size, if needed
        model = model if model is not None else Som((size[0], size[1], n_features))
        # Pass the LR to the model
        model.alpha = torch.tensor(lr)
        # Wrap the loss function
        loss_func = SomLoss(loss_func, model)
        # Initialize the trainer with the model
        callbacks.append(trainer(model))
        # Pass model reference to metrics
        metrics = list(map(lambda fn: partial(fn, som=model), metrics)) if metrics is not None else []
        if 'opt_func' not in learn_kwargs:
            learn_kwargs['opt_func'] = SomOptimizer
        super().__init__(data, model, callbacks=callbacks, loss_func=loss_func, metrics=metrics, **learn_kwargs)
        # Add visualization callbacks
        self.callbacks += visualization_callbacks(visualize, visualize_on, self)
        # Add optional data compatibility callback
        if isinstance(data, TabularDataBunch):
            self.callbacks.append(UnifyDataCallback())
        self.callbacks = list(set(self.callbacks))

    def codebook_to_df(self, recategorize: bool = False) -> pd.DataFrame:
        """
        Exports the SOM model codebook as a Pandas DataFrame.

        Parameters
        ----------
        recategorize: bool = False default=False
            Thether to apply backwards transformation of encoded categorical features. Only works with `TabularDataBunch`.
        """
        # Clone model weights
        w = self.model.weights.clone().cpu()
        w = w.view(-1, w.shape[-1])

        # TODO: Change with our tabular subclass
        if True or isinstance(self.data, TabularDataBunch) and recategorize:
            # TODO: retrieve denormalized data
            # Optional(?) feature recategorization
            # encoding_proc = find(self.data.processor[0].procs, lambda proc: isinstance(proc, ToBeContinuousProc))
            encoding_proc = self.data.processor[0].procs[-2]
            if encoding_proc is None:
                raise ValueError('Attribute recategorize=True, but no proc of type ToBeContinuousProc was found')
            cont_names, cat_names = encoding_proc.original_cont_names, encoding_proc.original_cat_names
            encoded_cat_names = encoding_proc.cat_names
            print(len(cont_names), len(cat_names))
            cat_features = encoding_proc.apply_backwards(w[:, :len(encoded_cat_names)])
            print(cat_features.shape)
            cont_features = w[:, len(encoded_cat_names):]
            w = np.concatenate([cat_features, cont_features], axis=-1)
            columns = cat_names+cont_names
        else:
            # TODO: retrieve column names in some way for other types of DataBunch
            w = w.numpy()
            columns = list(map(lambda i: f'Feature #{i+1}', range(w.shape[-1])))
        # Create the DataFrame
        df = pd.DataFrame(data=w, columns=columns)
        # Add SOM rows/cols coordinates into the `df`
        coords = index_tensor(self.model.size[:-1]).cpu().view(-1, 2).numpy()
        df['som_row'] = coords[:, 0]
        df['som_col'] = coords[:, 1]
        return df
