"""
This module contains various PyTorch `Sampler` utilities.
"""

from enum import Enum
from torch.utils.data import Dataset, Sampler, RandomSampler, SequentialSampler
from typing import Union

from ..core import enum_eq

__all__ = [
    'SamplerType',
    'SamplerTypeOrString',
    'get_sampler',
]


class SamplerType(Enum):
    "Enum used to pick PyTorch Samplers."
    RANDOM = 'random'
    SHUFFLE = 'shuffle'
    SEQUENTIAL = 'seq'


SamplerTypeOrString = Union[str, SamplerType]


def get_sampler(st: SamplerTypeOrString, dataset: Dataset, bs: int) -> Sampler:
    "Creates the correct PyTorch sampler for the given `SamplerType`."
    if enum_eq(SamplerType.RANDOM, st):
        return RandomSampler(dataset, replacement=True, num_samples=bs)
    elif enum_eq(SamplerType.SHUFFLE, st):
        return RandomSampler(dataset, replacement=False)
    elif enum_eq(SamplerType.SEQUENTIAL, st):
        return SequentialSampler(dataset)
    else:
        print(f'Unknown sampler "{str(st)}" requested; falling back to SequentialSampler.')
        return SequentialSampler(dataset)
