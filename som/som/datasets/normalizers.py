"""
This file contains various input normalizers
to be used together with the `UnsupervisedDataset` class.

Highly inspired by https://github.com/sevamoo/SOMPY/blob/master/sompy/normalization.py
"""
from typing import Tuple
from torch import Tensor


__all__ = [
    "get_normalizer",
    "Normalizer",
    "VarianceNormalizer",
]


class Normalizer():
    "Base normalizer class."

    def normalize(self, data: Tensor) -> Tensor:
        "Normalizes `data` by using this normalizer's criterion."
        raise NotImplementedError

    def normalize_by(self, source: Tensor, target: Tensor) -> Tensor:
        "Normalizes `target` by using `source` stats and this normalizer's criterion."
        raise NotImplementedError


class VarianceNormalizer(Normalizer):
    "Normalizer that uses mean and standard deviation."

    def _mean_and_std(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        "Calculates mean and std of `data`"
        return data.mean(dim=0), data.std(dim=0)

    def normalize(self, data: Tensor) -> Tensor:
        "Normalizes `data` by using its mean and standard deviation."
        mean, std = self._mean_and_std(data)
        std[std == 0] = 1
        return (data - mean) / std

    def normalize_by(self, source: Tensor, target: Tensor) -> Tensor:
        "Normalizes `target` by using mean and standard deviation of `source`."
        mean, std = self._mean_and_std(source)
        std[std == 0] = 1
        return (target - mean) / std


class MinMaxScaler(Normalizer):
    "Rescales data between a minimum and a maximum"

    def __init__(self, minimum: int = 0, maximum: int = 1) -> None:
        self.minimum, self.maximum = minimum, maximum

    def normalize(self, data: Tensor) -> Tensor:
        "Scales `data` between this scaler's `minimum` and `maximum`."
        return (self.maximum - self.minimum) * (data - data.min()) / (data.max() - data.min()) + self.minimum

    def normalize_by(self, source: Tensor, target: Tensor) -> Tensor:
        "Scales `target` between this scaler's `minimum` and `maximum`, using `source`'s min and max values."
        return (self.maximum - self.minimum) * (target - source.min()) / (source.max() - source.min()) + self.minimum


# Normalizers dict
_NORMALIZERS = {
    'var': VarianceNormalizer(),
    'minmax': MinMaxScaler(),
    'minmax-1': MinMaxScaler(minimum=-1),
}


def get_normalizer(name: str) -> Normalizer:
    "Returns the requested normalizer"
    if name not in _NORMALIZERS:
        raise KeyError(f'Normalizer not found: {name}')
    return _NORMALIZERS[name]
