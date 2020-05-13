"""
This file contains various input normalizers
to be used together with the `UnsupervisedDataset` class.

Highly inspired by https://github.com/sevamoo/SOMPY/blob/master/sompy/normalization.py
"""
import torch
from torch import Tensor
from typing import Tuple, Generator


__all__ = [
    "get_normalizer",
    "Normalizer",
    "VarianceNormalizer",
    "MinMaxScaler",
]


class Normalizer():
    "Base normalizer class."

    def normalize(self, data: Tensor, save: bool = False) -> Tensor:
        "Normalizes `data` by using this normalizer's criterion."
        return torch.stack([self._normalize(feature_tensor, save=save).squeeze(-1) for feature_tensor in self._list_features(data)], dim=-1)

    def normalize_by(self, source: Tensor, target: Tensor, save: bool = False) -> Tensor:
        "Normalizes `target` by using `source` stats and this normalizer's criterion."
        zipped_features = zip(self._list_features(source), self._list_features(target))
        return torch.stack([self._normalize_by(feature_tensor, feature_target, save=save).squeeze(-1) for feature_tensor, feature_target in zipped_features], dim=-1)

    def _normalize(self, data: Tensor, save: bool = False) -> Tensor:
        raise NotImplementedError

    def _normalize_by(self, source: Tensor, target: Tensor, save: bool = False) -> Tensor:
        raise NotImplementedError

    def denormalize(self, data: Tensor) -> Tensor:
        "Denormalizes `data` by using this normalizer's stats."
        return torch.stack([self._denormalize(feature_data, idx).squeeze(-1) for idx, feature_data in enumerate(self._list_features(data))], dim=-1)

    def _denormalize(self, feature_data: Tensor, feature_idx: int) -> Tensor:
        raise NotImplementedError

    def _list_features(self, t: Tensor) -> Generator[Tensor, None, None]:
        "Returns a generator over features in `t`."
        for idx in range(t.shape[-1]):
            yield t.index_select(-1, torch.tensor([idx]))


class VarianceNormalizer(Normalizer):
    "Normalizer that uses mean and standard deviation."

    def __init__(self) -> None:
        self.means, self.stds = [], []

    def _mean_and_std(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        "Calculates mean and std of `data`"
        return data.mean(dim=0), data.std(dim=0)

    def _normalize(self, data: Tensor, save: bool = False) -> Tensor:
        "Normalizes `data` by using its mean and standard deviation."
        mean, std = self._mean_and_std(data)
        if save:
            self.means.append(mean)
            self.stds.append(std)
        std[std == 0] = 1
        return (data - mean) / std

    def _normalize_by(self, source: Tensor, target: Tensor, save: bool = False) -> Tensor:
        "Normalizes `target` by using mean and standard deviation of `source`."
        mean, std = self._mean_and_std(source)
        if save:
            self.means.append(mean)
            self.stds.append(std)
        std[std == 0] = 1
        return (target - mean) / std

    def _denormalize(self, feature_data: Tensor, feature_idx: int) -> Tensor:
        "Denormalizes `feature_data` by using stored mean and standard deviation for each feature."
        if not len(self.means) or not len(self.stds):
            raise RuntimeError("`normalize` must be called with `save=True` before attempting to denormalize data")
        return self.stds[feature_idx] * feature_data + self.means[feature_idx]


class MinMaxScaler(Normalizer):
    "Rescales data between a minimum and a maximum"

    def __init__(self, minimum: int = 0, maximum: int = 1) -> None:
        self.minimum, self.maximum = minimum, maximum
        self.mins, self.maxs = [], []

    def _rescale(self, data: Tensor, minimum: int, maximum: int) -> Tensor:
        return (maximum - minimum) * (data - data.min()) / (data.max() - data.min()) + minimum

    def _normalize(self, data: Tensor, save: bool = False) -> Tensor:
        "Scales `data` between this scaler's `minimum` and `maximum`."
        feature_min = data.min()
        feature_max = data.max()
        if save:
            self.mins.append(feature_min)
            self.maxs.append(feature_max)
        return self._rescale(data, self.minimum, self.maximum)

    def _normalize_by(self, source: Tensor, target: Tensor, save: bool = False) -> Tensor:
        "Scales `target` between this scaler's `minimum` and `maximum`, using `source`'s min and max values."
        return (self.maximum - self.minimum) * (target - source.min()) / (source.max() - source.min()) + self.minimum

    def _denormalize(self, feature_data: Tensor, feature_idx: int) -> Tensor:
        """Denormalizes `data` by rescaling it into its original range."""
        return self._rescale(feature_data, self.mins[feature_idx], self.maxs[feature_idx])


# Normalizers dict
_NORMALIZERS = {
    'var': VarianceNormalizer(),
    'minmax': MinMaxScaler(),
    'minmax-1': MinMaxScaler(minimum=-1),
}


def get_normalizer(name: str) -> Normalizer:
    """
    Returns an instance of the requested normalizer.

    Parameters
    ----------
    name : str
        The normalizer name. Available values are `var`, `minmax` and `minmax-1`.
    """
    if name not in _NORMALIZERS:
        raise KeyError(f'Normalizer not found: {name}')
    return _NORMALIZERS[name]
