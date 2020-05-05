"""
This module mimics Fastai dataset utilities for unsupervised data.

"""
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from fastai.basic_data import DataBunch
from fastai.tabular import TabularDataBunch, FillMissing, Categorify, Normalize, TabularList
from typing import Union, Optional, List, Callable, Tuple, Collection

from .normalizers import get_normalizer
from .samplers import SamplerType, get_sampler, SamplerTypeOrString
from .cat_encoders import CatEncoder, CatEncoderTypeOrString

from ..core import ifnone


__all__ = [
    "UnsupervisedDataBunch",
    "pct_split",
    "build_dataloaders",
]


def pct_split(x: Tensor, valid_pct: float = 0.2):
    """
    Returns a tuple of (train, valid) indices that randomly split `x` with `valid_pct`.

    Parameters
    ----------
    x : Tensor
        The tensor to be split.
    valid_pct : float default=0.2
        The validation data percentage.
    """
    sep = int(len(x) * (1.0 - valid_pct))
    perm = torch.randperm(len(x))
    return perm[:sep], perm[sep:]


TrainData = Union[
    Tensor,
    TensorDataset,
    Tuple[Tensor, Tensor],
    torch.utils.data.DataLoader,
]
ValidData = Union[TrainData, float]


def build_dataloaders(
        train: TrainData,
        valid: ValidData,
        sampler: SamplerTypeOrString,
        bs: int,
) -> Tuple[DataLoader, DataLoader, bool]:
    """
    Transforms `train` and `valid` into `DataLoader` instances.

    Parameters
    ----------
    train: Union[Tensor, TensorDataset, Tuple[Tensor, Tensor], torch.utils.data.DataLoader]
        The training dataset. If a single `Tensor` is provided, it will be replicated as target.
    valid: Union[float, Tensor, TensorDataset, Tuple[Tensor, Tensor], torch.utils.data.DataLoader]
        The validation dataset or split percentage over training data.
    sampler: SamplerTypeOrString
        The sampler to be used to build the `DataLoader`.
    bs: int
        The batch size.
    """
    train_type, valid_type = type(train), type(valid)
    has_labels = not isinstance(train, Tensor) and ((not isinstance(train, Tuple)) or len(train) > 1)
    train = (train, train) if not has_labels else train
    if isinstance(train, Tuple):
        if isinstance(valid, float):
            train_idxs, valid_idxs = pct_split(train[0], valid_pct=valid)
            valid = (train[0][valid_idxs], train[1][valid_idxs])
            train = (train[0][train_idxs], train[1][train_idxs])
        elif valid is None:
            valid = (torch.tensor([]), torch.tensor([]))
        elif not has_labels:
            valid = (valid, valid)

        # Build TensorDatasets
        train = TensorDataset(train[0], train[1])
        valid = TensorDataset(valid[0], valid[1])

    if isinstance(train, TensorDataset):
        train_smp = get_sampler(sampler, train, bs)
        valid_smp = get_sampler(sampler, valid, bs)
        train = DataLoader(train, sampler=train_smp, batch_size=bs)
        valid = DataLoader(valid, sampler=valid_smp, batch_size=bs)

    if isinstance(train, DataLoader) and isinstance(valid, DataLoader):
        return train, valid, has_labels

    raise ValueError(f'Unxpected train / valid data pair of type: {train_type} {valid_type}')


class UnsupervisedDataBunch(DataBunch):
    """
    `DataBunch` subclass without mandatory labels.
    If labels are not provided, they will be stubbed.

    All keyword args not listed below will be passed to the parent class.

    Parameters
    ----------
    train: Union[Tensor, TensorDataset, Tuple[Tensor, Tensor], torch.utils.data.DataLoader]
        The training dataset / DataLoader or a Tuple in the form (train, labels). If a single `Tensor` is provided, labels will be stubbed.
    valid: Union[float, Tensor, TensorDataset, Tuple[Tensor, Tensor], torch.utils.data.DataLoader]
        The validation dataset / DataLoader or split percentage over training data.
    bs : int default=64
        The batch size.
    sampler : SamplerTypeOrString default=SamplerType.SEQUENTIAL
        The sampler to be used. Can be `seq`, 'random' or 'shuffle'.
    normalizer : str default='var'
    tfms : Optional[List[Callable]] default=None
        Additional Fastai transforms. These will be forwarded to the DataBunch.
    cat_enc : Optional[CatEncoder] default=None
        The categorical encoder to be used, if any.
    """

    def __init__(
            self,
            train: TrainData,
            valid: Optional[ValidData] = None,
            bs: int = 64,
            sampler: SamplerTypeOrString = SamplerType.SEQUENTIAL,
            tfms: Optional[List[Callable]] = None,
            cat_enc: Optional[CatEncoder] = None,
            normalizer: str = 'var',
            **kwargs):
        self.cat_enc = cat_enc
        self.normalizer = None

        # Build DataLoaders for train / validation data by checking given input types
        train_dl, valid_dl, has_labels = build_dataloaders(train, valid, sampler, bs)

        # Keep track of labels availability
        self.has_labels = has_labels

        # Initialize Fastai's DataBunch
        super().__init__(
            train_dl,
            valid_dl,
            device=torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            dl_tfms=tfms,
            **kwargs
        )

        # Normalize data
        if normalizer is not None:
            self.normalize(normalizer)

    @classmethod
    def from_tabular_databunch(
            cls,
            data: TabularDataBunch,
            bs: Optional[int] = None,
            normalizer: Optional[str] = 'var',
            cat_enc: Union[CatEncoderTypeOrString, CatEncoder] = "onehot"):
        """
        Creates a new UnsupervisedDataBunch from a DataFrame.

        Parameters
        ----------
        data : TabularDataBunch
            The source TabularDataBunch.
        bs: Optional[int] default=None
            The batch size. Defaults to the source databunch batch size if not provided.
        normalizer: Optional[str] default='var'
            The optional normalization strategy to be used.
        cat_enc : Union[CatEncoderTypeOrString, CatEncoder] default='onehot'
            Categorical encoder.
        """
        return data.to_unsupervised_databunch(bs=bs, cat_enc=cat_enc)

    @classmethod
    def from_df(
            cls,
            df: pd.DataFrame,
            cat_names: List[str],
            cont_names: List[str],
            dep_vars: Optional[str] = None,
            bs: int = 128,
            valid_pct: float = 0.2,
            normalizer: Optional[str] = 'var',
            cat_enc: Union[CatEncoderTypeOrString, CatEncoder] = "onehot"):
        """
        Creates a new UnsupervisedDataBunch from a DataFrame.

        Parameters
        ----------
        df : pd.Dataframe
            The source DataFrame.
        cat_names : List[str]
            Categorical feature names.
        cont_names : List[str]
            Continuous feature names.
        dep_vars : Union[str, List[str]]
            The target variable(s).
        bs: int default=128
            The batch size.
        valid_pct : float default=0.2
            Validation split percentage.
        normalizer: Optional[str] default='var'
            The optional normalization strategy to be used.
        cat_enc : Union[CatEncoderTypeOrString, CatEncoder] default='onehot'
            Categorical encoder.
        """
        procs = [FillMissing, Categorify]
        dep_vars = dep_vars if dep_vars is None or isinstance(dep_vars, Collection) else [dep_vars]
        tabular_data = TabularList.from_df(df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs).split_by_rand_pct(valid_pct)
        if dep_vars is not None:
            tabular_data = tabular_data.label_from_df(cols=dep_vars)
        else:
            tabular_data = tabular_data.label_empty()
        tabular_data = tabular_data.databunch(bs=bs, num_workers=0)
        return tabular_data.to_unsupervised_databunch(bs=bs, cat_enc=cat_enc)

    def normalize(self, normalizer: str = "var") -> None:
        """
        Uses `normalizer` to normalize both train and validation data.

        Parameters
        ----------
        normalizer : str default='var'
            The normalizer to be used. Available values are 'var', 'minmax' or 'minmax-1'.
        """
        save_stats = self.normalizer is None
        self.normalizer = ifnone(self.normalizer, get_normalizer(normalizer))

        train_x, train_y = self.train_ds.tensors
        norm_train_x = self.normalizer.normalize(train_x, save=save_stats)
        self.train_ds.tensors = (norm_train_x, train_y)

        if self.valid_ds is not None and len(self.valid_ds) > 1:
            valid_x, valid_y = self.valid_ds.tensors
            norm_valid_x = self.normalizer.normalize_by(train_x, valid_x)
            self.valid_ds.tensors = (norm_valid_x, valid_y)

    def denormalize(self, data: Tensor) -> Tensor:
        """
        Denormalizes a `Tensor` using the stored normalizer.
        Falls back to simply returning input data if no normalizer is available.
        """
        if self.normalizer is None:
            return data
        return self.normalizer.denormalize(data)

    def make_categorical(self, t: Tensor) -> np.ndarray:
        """Transforms a Tensor `t` of encoded categorical variables into their original categorical form."""
        return self.cat_enc.make_categorical(t)
