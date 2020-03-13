"""
This module mimics Fastai dataset utilities for unsupervised data.

"""
from typing import Union, Optional, List, Callable
from enum import Enum
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler, RandomSampler, SequentialSampler, BatchSampler
from torch import Tensor
from fastai.basic_data import DataBunch

from .normalizers import Normalizer, get_normalizer


__all__ = [
    "UnsupervisedDataset",
    "UnsupervisedDataBunch",
    "pct_split",
    "SamplerType",
]


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


def pct_split(x: Tensor, valid_pct: float = 0.2):
    "Splits a dataset in `train` and `valid` by using `pct`."
    sep = int(len(x) * (1.0 - valid_pct))
    perm = x[torch.randperm(len(x))]
    return perm[:sep], perm[sep:]


class UnsupervisedDataset():
    "Represents a dataset without targets."

    def __init__(
            self, train: Tensor,
            valid: Union[Tensor, float] = 0.2,
            bs: int = 64,
            normalizer: Union[Normalizer, str] = 'minmax',
            random_batching: bool = False, shuffle: bool = False):
        if isinstance(valid, float):
            self.train, self.valid = pct_split(train, valid_pct=valid)
        else:
            self.train, self.valid = train, valid
        self.bs = bs
        self.use_cuda = torch.cuda.is_available()
        # Move train and validation datasets to this dataset's device
        self.train = self._to_device(self.train)
        self.valid = self._to_device(self.valid)
        # Initialize the normalizer
        self.normalizer = self._get_normalizer(normalizer)
        # Save random batching option (accessed by `Learner` to determine batch count)
        self.random_batching = random_batching
        # Setup the slicer to be used for dataset batchwise iteration
        self.slicer = batch_slice(bs, self.train.shape[0]) if not random_batching else random_batch_slice(bs, self.train.shape[0])
        # Optionally shuffle data
        if shuffle:
            self._shuffle()

    @classmethod
    def create(
            cls, train: Tensor, valid: Tensor, bs: int = 64,
            normalizer: Union[Normalizer, str] = 'minmax',
            random_batching: bool = False, shuffle: bool = False):
        """
        Creates a new UnsupervisedDataset.\n
        \n
        Params:\n
        `train`             : training `Tensor`\n
        `valid`             : validation `Tensor`, or percentage w.r.t the train dataset\n
        `bs`                : batch size to use for iteration\n
        `normalizer`        : `Normalizer` instance, or normalizer name\n
        `random_batching`   : enables random batching instead of sequential\n
        `shuffle`           : shuffles the train set on inizialization\n
        """
        return cls(train, valid, bs=bs, normalizer=normalizer, random_batching=random_batching, shuffle=shuffle)

    def _to_device(self, a: Tensor) -> Tensor:
        "Moves a tensor to the correct device."
        return a.cuda() if self.use_cuda else a.cpu()

    def _get_normalizer(self, normalizer: Union[Normalizer, str]) -> Normalizer:
        "Initializes the normalizer correctly."
        if isinstance(normalizer, Normalizer):
            return normalizer
        return get_normalizer(normalizer)

    def _shuffle(self):
        "Shuffles the training set."
        self.train = self.train[torch.randperm(self.train.shape[0])]

    def normalize(self):
        "Normalizes data and validation set."
        if self.normalizer is not None:
            self.valid = self.normalizer.normalize_by(self.train, self.valid)
            self.train = self.normalizer.normalize(self.train)

    def grab_batch(self):
        "Grabs the next batch of training data."
        return self.train[next(self.slicer)]


TensorOrDataLoader = Union[torch.Tensor, torch.utils.data.DataLoader]
TensorOrDataLoaderOrSplitPct = Union[TensorOrDataLoader, float]


class SamplerType(Enum):
    "Enum used to pick PyTorch Samplers."
    RANDOM = 'rand'
    SHUFFLE = 'shuffle'
    SEQUENTIAL = 'seq'


SamplerTypeOrString = Union[str, SamplerType]


def get_sampler(st: SamplerTypeOrString, dataset: Dataset, bs: int) -> torch.utils.data.Sampler:
    "Creates the correct PyTorch sampler for the given `SamplerType`."
    if st == SamplerType.RANDOM or st == SamplerType.RANDOM.value:
        return RandomSampler(dataset, replacement=False, num_samples=bs)
    elif st == SamplerType.SHUFFLE or st == SamplerType.SHUFFLE.value:
        return RandomSampler(dataset, replacement=True, num_samples=bs)
    elif st == SamplerType.SEQUENTIAL or st == SamplerType.SEQUENTIAL.value:
        return SequentialSampler(dataset)
    else:
        print(f'Unknown sampler "{str(st)} requested; falling back to SequentialSampler."')
        return SequentialSampler(dataset)


class UnsupervisedDataBunch(DataBunch):
    "DataBunch without a target."

    def __init__(self, train: TensorOrDataLoader, valid: Optional[TensorOrDataLoaderOrSplitPct] = None,
                 bs: int = 64,
                 sampler: SamplerTypeOrString = SamplerType.SEQUENTIAL,
                 tfms: Optional[List[Callable]] = None,
                 **kwargs):

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

    def normalize(self, normalizer: str = 'minmax') -> None:
        "Normalizes both `train_ds` and `valid_ds`"
        normalizer = get_normalizer(normalizer)

        train = self.train_ds.tensors[0]
        norm_train = normalizer.normalize(train)
        self.train_ds.tensors = [norm_train, norm_train]

        if self.valid_ds is not None:
            valid = self.valid_ds.tensors[0]
            norm_valid = normalizer.normalize_by(train, valid)
            self.valid_ds.tensors = [norm_valid, norm_valid]
