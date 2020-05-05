"""
This module defines a Fastai `Learner` subclass used to train Self-Organizing Maps.
"""
from torch import Tensor
import torch
import pandas as pd
import numpy as np

from typing import Optional, Callable, Collection, List, Type, Dict, Tuple
from functools import partial
from fastai.basic_train import Learner
from fastai.basic_data import DataBunch
from fastai.train import *
from fastai.callback import Callback

from .callbacks import SomTrainer, ExperimentalSomTrainer
from .initializers import som_initializers
from .loss import SomLoss
from .optim import SomOptimizer

from ..core import ifnone, setify, index_tensor
from ..datasets import UnsupervisedDataBunch
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
        n_features = data.train_ds.tensors[0].shape[-1]
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
        super().__init__(data, model, callbacks=callbacks, loss_func=loss_func, metrics=metrics, **learn_kwargs)
        # Add visualization callbacks
        self.callbacks += visualization_callbacks(visualize, visualize_on, self)
        self.callbacks = list(set(self.callbacks))

    def codebook_to_df(self, cat_values: Optional[Dict[str, Dict[int, str]]] = None, cat_as_str: bool = True) -> pd.DataFrame:
        """
        Exports the SOM model codebook as a Pandas DataFrame.

        Parameters
        ----------
        cat_values: Dict[str, Dict[int, str] default=None
            Nested dict of per-feature value-to-string mappings.
        cat_as_str : bool default=True
        Examples
        --------
        >>> codebook_to_df(cat_values=dict(feature_a=dict(0='Feature A - Value Zero', 1='Feature A - Value One')))
        """
        # Clone model weights
        w = self.model.weights.clone().cpu()
        w = w.view(-1, w.shape[-1])

        # Denormalization step
        w = self.data.denormalize(w)

        # Optional feature recategorization
        if self.data.cat_enc is not None:
            cont_count = len(self.data.cat_enc.cont_names)
            encoded_count = w.shape[-1] - cont_count
            cat = self.data.make_categorical(w[:, :encoded_count])
            # Transform categories back into strings
            if cat_as_str and cat_values is not None:
                cat = np.array([[cat_values[self.data.cat_enc.cat_names[idx]][el] for el in col] for idx, col in enumerate(cat.transpose())])
                cat = cat.transpose()
            w = np.concatenate([cat, w[:, encoded_count:]], axis=-1)
        else:
            w = w.numpy()

        # Create the DataFrame
        df = pd.DataFrame(data=w, columns=self.data.cat_enc.cat_names+self.data.cat_enc.cont_names)

        # Add SOM rows/cols coordinates into the `df`
        coords = index_tensor(self.model.size[:-1]).cpu().view(-1, 2).numpy()
        df['som_row'] = coords[:, 0]
        df['som_col'] = coords[:, 1]

        # Make cat features categories again
        for cat in self.data.cat_enc.cat_names:
            if not cat_as_str:
                df[cat] = df[cat].astype(int)
            df[cat] = df[cat].astype('category')

        return df
