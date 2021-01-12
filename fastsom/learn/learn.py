"""
This module defines a Fastai `Learner` subclass used to train Self-Organizing Maps.
"""
import pathlib
from functools import partial
from typing import Callable, Collection, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.tabular.data import Normalize, TabularDataLoaders, TabularProc
from fastai_category_encoders import CategoryEncode
from fastcore.meta import delegates
from typing_extensions import Literal

from fastsom.callback import (ExperimentalSomTrainer, SomTrainer,
                              get_visualization_callbacks, get_xy)
from fastsom.core import find, ifnone, index_tensor
from fastsom.log import has_logger
from fastsom.metrics import mean_quantization_err
from fastsom.som import MixedEmbeddingDistance, Som

from .initializers import som_initializers
from .loss import SomLoss
from .optim import SplashOptimizer
from .utils import denorm_with_proc

StrOrPath = Union[str, pathlib.Path]
TensorOrNumpyArray = Union[torch.Tensor, np.ndarray]
CategoryEncode.order = 100


__all__ = [
    "SomLearner",
    "ForwardContsCallback",
    "UnifyDataCallback",
    "StrOrPath",
    "TensorOrNumpyArray",
    "split_torch_numpy",
]


def split_torch_numpy(x: TensorOrNumpyArray, sections: List[int], dim: int = 0):
    """Splits `x` in PyTorch fashion, even if it is a Numpy Array."""
    if isinstance(x, np.ndarray):
        sections = np.cumsum(sections)[:-1]
        return np.split(x, sections, axis=dim)
    else:
        return torch.split(x, sections, dim=dim)


class ForwardContsCallback(Callback):
    """
    Callback for `SomLearner`, automatically added when the
    data class is `TabularDataLoaders`.
    Filters out categorical features, forwarding continous
    features to the SOM model.
    Note that if a categorical encoder is used, categoricals
    will be encoded to continuous features, and as such will
    be forwarded by this callback.
    """
    def before_batch(self):
        x_cat, x_cont = self.xb
        self.learn.xb = [x_cont]


class UnifyDataCallback(Callback):
    """
    Callback for `SomLearner`, automatically added when the
    data class is a `TabularDataLoaders`.
    Merges categorical and continuous features into a single tensor.
    """

    def before_batch(self):
        x_cat, x_cont = self.xb
        self.learn.xb = [torch.cat([x_cat.float(), x_cont], dim=-1)]


class SomLearner(Learner):
    """
    Learner subclass used to train Self-Organizing Maps.

    All keyword arguments not listed below are forwarded to the `Learner` parent class.

    Parameters
    ----------
    dls : DataLoaders
        Fast.ai data object.
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
    """
    @delegates(Learner)
    def __init__(
        self,
        dls: DataLoaders,
        model: Som = None,
        size: Tuple[int, int] = (10, 10),
        init_weights: Literal['kmeans', 'kmeans_cosine', 'random'] = 'random',
        lr: float = 0.6,
        trainer: Type[SomTrainer] = ExperimentalSomTrainer,
        cbs: List[Callback] = [],
        loss_func: Callable = mean_quantization_err,
        metrics: Collection[Callable] = None,
        visualize: List[str] = [],
        visualize_on: str = "epoch",
        **kwargs,
    ) -> None:
        n_inputs = dls.train._n_inp
        xb = dls.train.one_batch()[0:n_inputs]
        n_features = sum(map(lambda x: x.shape[-1], xb if n_inputs > 1 else [xb]))
        # Create a new Som using the size, if needed
        model = ifnone(model, Som((size[0], size[1], n_features)))
        # Pass the LR to the model
        model.alpha = torch.tensor(lr)
        # Optionally initialize weights
        if init_weights is not None and init_weights in som_initializers:
            init = som_initializers[init_weights]
            num_bmus = size[0] * size[1]
            model.weights = init(get_xy(dls)[0], num_bmus).view(size[0], size[1], n_features)
        # Wrap the loss function
        loss_func = SomLoss(loss_func, model)
        # Pass model reference to metrics
        metrics = (list(map(lambda fn: partial(fn, som=model), metrics)) if metrics is not None else [])
        if "opt_func" not in kwargs:
            kwargs["opt_func"] = SplashOptimizer
        super().__init__(dls, model, cbs=cbs, loss_func=loss_func, metrics=metrics, **kwargs)
        # Add visualization callbacks
        self.add_cbs(get_visualization_callbacks(visualize, visualize_on))
        # Add training callback
        self.add_cbs(trainer())
        # Adjust SOM distance function + optional data pipelining callbacks
        if isinstance(dls, TabularDataLoaders):
            self.__maybe_adjust_model_dist_fn()

    def __maybe_adjust_model_dist_fn(self):
        """Changes the SOM distance function if the data type requires it."""
        encoder = self._find_categorical_encoder()
        # If no categorizer is provided, append the appropriate callback
        if encoder is None:
            self.add_cb(ForwardContsCallback())
            # if len(self.dls.cat_names) > 0:
            #     self.logger.warning("Categorical variables will NOT be passed to the SOM unless encoded with the `CategoryEncode` proc.")
        elif encoder.uses_embeddings:
            self.model.dist_fn = MixedEmbeddingDistance(encoder.get_emb_szs())
            self.add_cb(UnifyDataCallback())
        else:
            # TODO: handle other category encoders.
            # For now, just ignore categoricals,
            # expecting them to be encoded as continuous.
            self.add_cb(ForwardContsCallback())

    def recategorize(self, data: torch.Tensor, return_names: bool = False, denorm: bool = False) -> np.ndarray:
        """Recategorizes `data`, optionally returning cat/cont names."""
        if not self.is_tabular:
            raise ValueError("Recategorization is available only when using TabularDataLoaders")
        encoder = self._find_categorical_encoder()
        if encoder is None:
            raise ValueError("No CategoryEncode transform found while applying recategorization.")
        # Retrieve original & encoded feature names
        cont_names, cat_names = list(encoder.encoder.cont_names), list(encoder.encoder.cat_names)
        encoded_cat_names = list(encoder.encoder.get_feature_names())
        if denorm:
            data = self.denormalize(data)
        cats, conts = self.split_cats_conts(data)
        ret = encoder.encoder.inverse_transform(pd.DataFrame(cats.cpu().numpy(), columns=encoded_cat_names)).values
        if data.shape[-1] > len(encoded_cat_names):
            ret = np.concatenate([ret, conts.cpu().numpy()], axis=-1)
        return ret if not return_names else (ret, cat_names, cont_names)

    def codebook_to_df(self, recategorize: bool = False, denorm: bool = True) -> pd.DataFrame:
        """
        Exports the SOM model codebook as a Pandas DataFrame.

        Parameters
        ----------
        recategorize: bool = False
            Whether to apply backwards transformation of encoded categorical features. Only works with `TabularDataLoaders`.
        denorm: bool = True
            Whether data should be de-normalized.
        """
        # Clone model weights
        w = self.model.weights.clone().cpu()
        w = w.view(-1, w.shape[-1])
        if self.is_tabular:
            if self._find_categorical_encoder() is not None and recategorize:
                w, cat_names, cont_names = self.recategorize(w, return_names=True, denorm=denorm)
                cats, conts = self.split_cats_conts(w)
            else:
                cont_names, cat_names = list(self.dls.cont_names), list(self.dls.cat_names)
                cats, conts = self.split_cats_conts(self.denormalize(w)).cpu().numpy()
            # cat_features, cont_features = w[..., : len(cat_names)], w[..., len(cat_names):]
            w = np.concatenate([cats, conts], axis=-1)
            df = pd.DataFrame(data=w, columns=cat_names + cont_names)
            df[cont_names] = df[cont_names].astype(float)
            df[cat_names] = df[cat_names].astype(str)
        else:
            # TODO: retrieve column names in some way for other types of DataLoaders
            w = w.numpy()
            columns = list(map(lambda i: f"Feature #{i+1}", range(w.shape[-1])))
            df = pd.DataFrame(data=w, columns=columns)
        # Add SOM rows/cols coordinates into the `df`
        coords = index_tensor(self.model.size[:-1]).view(-1, 2).cpu().numpy()
        df["som_row"] = coords[:, 0]
        df["som_col"] = coords[:, 1]
        return df

    def export(self, file: StrOrPath = "export.pkl", destroy: bool = False):
        """Exports the Learner to file, removing unneeded callbacks."""
        cbs = list(self.cbs)
        self.cbs = list(filter(lambda cb: not getattr(cb, 'is_exportable', False), self.cbs))
        super().export(file, destroy)
        if not destroy:
            self.cbs = cbs

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalizes `data`."""
        if self.is_tabular:
            normalize_proc = self._find_normalizer()
            if normalize_proc is not None:
                if data.shape[-1] > len(normalize_proc.means):
                    cats, conts = self.split_cats_conts(data)
                    # conts, cats = (data[..., :len(normalize_proc.means)], data[..., len(normalize_proc.means):])
                    return torch.cat([cats, denorm_with_proc(conts, normalize_proc)], dim=-1)
                else:
                    return denorm_with_proc(data, normalize_proc)
        # TODO: implement for other DataLoaders types
        return data

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Returns three lists containing the categorical, continuous and encoded-categorical feature names.
        Only works if the associated DataLoaders is a TabularDataLoaders.
        Also, in case no encoding transform is associated with the dataloaders, the third list will be empty.
        """
        if self.is_tabular:
            cat_encoder = self._find_categorical_encoder()
            if cat_encoder is None:
                cat_names         = self.dls.cat_names
                cont_names        = self.dls.cont_names
                cat_names_encoded = []
            else:
                cat_names         = cat_encoder.encoder.cat_names
                cont_names        = cat_encoder.encoder.cont_names
                cat_names_encoded = cat_encoder.encoder.get_feature_names()
            return cat_names, cont_names, cat_names_encoded
        else:
            raise RuntimeError('Cannot get feature names of non-tabular dataloader')

    def split_cats_conts(self, x: TensorOrNumpyArray) -> Tuple[TensorOrNumpyArray, TensorOrNumpyArray]:
        """Splits a Tensor `x` into two parts, i.e. categorical and continuous features."""
        if self.is_tabular:
            cat_names, cont_names, cat_names_encoded = self.get_feature_names()
            if x.shape[-1] == len(cat_names) + len(cont_names):
                return split_torch_numpy(x, list(map(len, [cat_names, cont_names])), dim=-1)
            elif len(cat_names_encoded) > 0 and x.shape[-1] == len(cat_names_encoded) + len(cont_names):
                return split_torch_numpy(x, list(map(len, [cat_names_encoded, cont_names])), dim=-1)
            else:
                raise RuntimeError('Tensor size mismatch: cannot split Tensor with either encoded or non-encoded categorical features')
        else:
            raise RuntimeError('Cannot split features into categorical and continuous when self.dls is not a TabularDataLoaders')

    def _find_proc(self, proc_cls: Type[TabularProc]) -> Optional[TabularProc]:
        """
        If `self.dls` is a `TabularDataLoaders`, this method searches
        the data transforms for an instance of `proc_cls`.
        """
        if self.is_tabular:
            return find(self.dls.procs, lambda p: isinstance(p, proc_cls))
        else:
            raise RuntimeError('Cannot look for CategoryEncode procs in a non-tabular dataloaders')

    def _find_categorical_encoder(self) -> Optional[CategoryEncode]:
        return self._find_proc(CategoryEncode)

    def _find_normalizer(self) -> Optional[Normalize]:
        return self._find_proc(Normalize)

    @property
    def has_labels(self):
        return self.dls.train.ys.shape[-1] > 0

    @property
    def is_tabular(self):
        return isinstance(self.dls, TabularDataLoaders)
