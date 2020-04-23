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
from typing import Union, Optional, List, Callable

from .normalizers import get_normalizer
from .samplers import SamplerType, get_sampler, SamplerTypeOrString
from .cat_encoders import CatEncoder, CatEncoderTypeOrString

from ..core import ifnone


__all__ = [
    "UnsupervisedDataBunch",
    "pct_split",
]


def pct_split(x: Tensor, valid_pct: float = 0.2):
    """Splits a dataset in `train` and `valid` by using `pct`."""
    sep = int(len(x) * (1.0 - valid_pct))
    perm = x[torch.randperm(len(x))]
    return perm[:sep], perm[sep:]


TensorOrDataLoader = Union[torch.Tensor, torch.utils.data.DataLoader]
TensorOrDataLoaderOrSplitPct = Union[TensorOrDataLoader, float]


class UnsupervisedDataBunch(DataBunch):
    """
    `DataBunch` subclass without target data.

    All keyword args not listed below will be passed to the parent class.

    Parameters
    ----------
    train : TensorOrDataLoader
        The training data.
    valid : Optional[TensorOrDataLoaderOrSplitPct] default=None
        The validation data. Can be passed as a PyTorch Tensor / DataLoader or as a percentage of the training set.
    bs : int default=64
        The batch size.
    sampler : SamplerTypeOrString default=SamplerType.SEQUENTIAL
        The sampler to be used. Can be `seq`, 'random' or 'shuffle'.
    tfms : Optional[List[Callable]] default=None
        Additional Fastai transforms. These will be forwarded to the DataBunch.
    cat_enc : Optional[CatEncoder] default=None
        The categorical encoder to be used, if any.
    """

    def __init__(
            self, train: TensorOrDataLoader,
            valid: Optional[TensorOrDataLoaderOrSplitPct] = None,
            bs: int = 64,
            sampler: SamplerTypeOrString = SamplerType.SEQUENTIAL,
            tfms: Optional[List[Callable]] = None,
            cat_enc: Optional[CatEncoder] = None,
            **kwargs):
        self.cat_enc = cat_enc
        self.normalizer = None

        if isinstance(train, DataLoader):
            train_dl, valid_dl = train, valid
        else:
            if isinstance(valid, float):
                if not isinstance(train, Tensor):
                    raise TypeError("Training data should be passed as `Tensor` when passing valid percentage")
                train, valid = pct_split(train, valid_pct=valid)
            # Initialize datasets
            train_ds = TensorDataset(train, train)
            valid_ds = TensorDataset(valid, valid)
            # Initialize data samplers
            train_smp = get_sampler(sampler, train_ds, bs)
            valid_smp = get_sampler(sampler, valid_ds, bs)
            # Create data loaders + wrap samplers into batch samplers
            train_dl = DataLoader(train_ds, sampler=train_smp, batch_size=bs)
            valid_dl = DataLoader(valid_ds, sampler=valid_smp, batch_size=bs)

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

    @classmethod
    def from_tabular_databunch(cls, data: TabularDataBunch, bs: Optional[int] = None, cat_enc: Union[CatEncoderTypeOrString, CatEncoder] = "onehot"):
        """
        Creates a new UnsupervisedDataBunch from a DataFrame.

        Parameters
        ----------
        TODO
        """
        return data.to_unsupervised_databunch(bs=bs, cat_enc=cat_enc)

    @classmethod
    def from_df(cls, df: pd.DataFrame, cat_names: List[str], cont_names: List[str], dep_var: str, bs: int = 128, valid_pct: float = 0.2, cat_enc: Union[CatEncoderTypeOrString, CatEncoder] = "onehot"):
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
        dep_var : str
            The target variable.
        bs: int default=128
            The batch size.
        valid_pct : float default=0.2
            Validation split percentage.
        cat_enc : Union[CatEncoderTypeOrString, CatEncoder] default='onehot'
            Categorical encoder.
        """
        procs = [FillMissing, Categorify, Normalize]
        tabular_data = TabularList.from_df(df, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs) \
            .split_by_rand_pct(valid_pct) \
            .label_from_df(cols=dep_var) \
            .databunch(bs=bs, num_workers=0)
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

        train = self.train_ds.tensors[0]
        norm_train = self.normalizer.normalize(train, save=save_stats)
        self.train_ds.tensors = [norm_train, norm_train]

        if self.valid_ds is not None:
            valid = self.valid_ds.tensors[0]
            norm_valid = self.normalizer.normalize_by(train, valid)
            self.valid_ds.tensors = [norm_valid, norm_valid]

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


def batch_slice(bs: int, maximum: int) -> slice:
    """Generator function. Generates contiguous slices of size `bs`."""
    curr = 0
    while True:
        yield slice(curr, curr+bs)
        # Restart from 0 if max has been reached; advance to next batch otherwise
        curr = 0 if curr+bs > maximum or curr+bs*2 > maximum else curr + bs


def random_batch_slice(bs: int, maximum: int) -> Tensor:
    """Generator function. Generate uniform random long tensors that can be used to index another tensor."""
    base = torch.zeros(bs)
    while True:
        # Fill `base` with uniform data
        yield base.uniform_(0, maximum).long()
