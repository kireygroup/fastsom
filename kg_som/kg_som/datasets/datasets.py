"""
This module mimics Fastai dataset utilities for unsupervised data.

"""
from typing import Union, Optional, List, Callable
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from fastai.basic_data import DataBunch
from fastai.tabular import TabularDataBunch

from .normalizers import Normalizer, get_normalizer
from .samplers import SamplerType, get_sampler, SamplerTypeOrString
from .cat_encoders import CatEncoder

from ..core import ifnone


__all__ = [
    "UnsupervisedDataBunch",
    "pct_split",
]


def pct_split(x: Tensor, valid_pct: float = 0.2):
    "Splits a dataset in `train` and `valid` by using `pct`."
    sep = int(len(x) * (1.0 - valid_pct))
    perm = x[torch.randperm(len(x))]
    return perm[:sep], perm[sep:]


TensorOrDataLoader = Union[torch.Tensor, torch.utils.data.DataLoader]
TensorOrDataLoaderOrSplitPct = Union[TensorOrDataLoader, float]


class UnsupervisedDataBunch(DataBunch):
    "DataBunch without a target."

    def __init__(self, train: TensorOrDataLoader, valid: Optional[TensorOrDataLoaderOrSplitPct] = None,
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
        super().__init__(train_dl, valid_dl, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), dl_tfms=tfms, **kwargs)

    def normalize(self, normalizer: str = 'var') -> None:
        "Normalizes both `train_ds` and `valid_ds`."
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
        "Denormalizes a `Tensor` using own normalizer."
        if self.normalizer is None:
            return data
        return self.normalizer.denormalize(data)



def batch_slice(bs: int, maximum: int) -> slice:
    "Generator function. Generates contiguous slices of size `bs`."
    curr = 0
    while True:
        yield slice(curr, curr+bs)
        # Restart from 0 if max has been reached; advance to next batch otherwise
        curr = 0 if curr+bs > maximum or curr+bs*2 > maximum else curr + bs


def random_batch_slice(bs: int, maximum: int) -> Tensor:
    "Generator function. Generate uniform random long tensors that can be used to index another tensor."
    base = torch.zeros(bs)
    while True:
        # Fill `base` with uniform data
        yield base.uniform_(0, maximum).long()
