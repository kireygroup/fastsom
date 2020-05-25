"""
This module defines a Fastai `Learner` subclass used to train Self-Organizing Maps.
"""
import torch
import pandas as pd
import numpy as np

from typing import Callable, Collection, List, Type, Tuple, Dict
from functools import partial
from fastai.basic_train import Learner
from fastai.basic_data import DataBunch
from fastai.tabular import TabularDataBunch, Normalize
from fastai.train import *
from fastai.callback import Callback

from .callbacks import SomTrainer, ExperimentalSomTrainer
from .loss import SomLoss
from .optim import SomOptimizer

from ..core import ifnone, setify, index_tensor, find
from ..data_block import get_xy, ToBeContinuousProc, Vectorize
from ..interp import SomTrainingViz, SomHyperparamsViz, SomBmuViz, mean_quantization_err
from ..som import Som, grouped_distance, split_distance, pcosdist


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
    """
    Callback for `SomLearner`, automatically added when the
    data class is a `TabularDataBunch`.
    Filters out categorical features, forwarding continous
    features to the SOM model.
    """
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
            # If categorical features are encoded as embeddings, we need to ensure the correct distance function is used
            vec_proc = find(data.processor[0].procs, lambda p: isinstance(p, Vectorize))
            if vec_proc is not None:
                chunked_emb_dist = partial(grouped_distance, dist_fn=pcosdist, n_groups=vec_proc._ft.vector_size)
                chunked_emb_dist.__name__ = f'{grouped_distance.__name__}(dist_fn={pcosdist.__name__}, n_groups={vec_proc._ft.vector_size})'
                if vec_proc._is_mixed():
                    # TODO: check split idx
                    split_dist = partial(split_distance,
                                         split_idx=len(vec_proc._out_cat_names),
                                         dist_fn_1=chunked_emb_dist,
                                         dist_fn_2=self.model.dist_fn)
                    split_dist.__name__ = f'{split_distance.__name__}(\
                                            dist_fn_1={chunked_emb_dist.__name__},\
                                            dist_fn_2={self.model.dist_fn.__name__})'
                    self.model.dist_fn = split_dist
                else:
                    self.model.dist_fn = chunked_emb_dist
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

        if isinstance(self.data, TabularDataBunch):    
            if recategorize:
                # Feature recategorization
                encoding_proc = find(self.data.processor[0].procs, lambda proc: isinstance(proc, ToBeContinuousProc))
                if encoding_proc is None:
                    raise ValueError('Attribute recategorize=True, but no proc of type ToBeContinuousProc was found')
                cont_names, cat_names = encoding_proc.original_cont_names, encoding_proc.original_cat_names
                encoded_cat_names = encoding_proc.cat_names
                cat_features = encoding_proc.apply_backwards(w[..., :len(encoded_cat_names)])
                cont_features = w[..., len(encoded_cat_names):]
            else:
                cont_names, cat_names = self.data.cont_names, self.data.cat_names
                cont_features = w[..., len(cat_names):]
                cat_features = w[..., :len(cat_names)]
            # Continuous features denormalization
            normalize_proc = find(self.data.processor[0].procs, lambda proc: isinstance(proc, Normalize))
            if normalize_proc is not None:
                cont_features = denormalize(cont_features, normalize_proc.means, normalize_proc.stds)

            df = pd.DataFrame(data=np.concatenate([cat_features, cont_features], axis=-1), columns=cat_names+cont_names)
            df[cont_names] = df[cont_names].astype(float)
            df[cat_names] = df[cat_names].astype(str)    
        else:
            # TODO: retrieve column names in some way for other types of DataBunch
            w = w.numpy()
            columns = list(map(lambda i: f'Feature #{i+1}', range(w.shape[-1])))
            df = pd.DataFrame(data=w, columns=columns)
        # Create the DataFrame
        # for col in df.columns:
            # df[col] = df[col].astype(get_type(pd.api.types.infer_dtype(df[col])))
        # Add SOM rows/cols coordinates into the `df`
        coords = index_tensor(self.model.size[:-1]).cpu().view(-1, 2).numpy()
        df['som_row'] = coords[:, 0]
        df['som_col'] = coords[:, 1]
        return df


def denormalize(data: torch.Tensor, means: Dict[str, float], stds: Dict[str, float]) -> torch.Tensor:
    """
    """
    means = torch.tensor(list(means.values()))
    stds = torch.tensor(list(stds.values()))
    return stds * data + means


def get_type(t: str) -> Type:
    print(t)
    if t == 'string':
        return str
    if t == 'integer':
        return np.int64
    if t == 'float':
        return float
