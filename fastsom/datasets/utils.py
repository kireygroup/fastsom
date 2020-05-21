import torch

from typing import Union, Tuple, Generator
from fastai.basic_data import DataBunch, DatasetType
from fastai.data_block import ItemList
from fastai.tabular import TabularDataBunch


__all__ = ['get_xy', 'get_xy_batched', 'set_xy']


def get_xy(data: DataBunch, ds_type: DatasetType = DatasetType.Train) -> Tuple[Union[torch.Tensor, ItemList], Union[torch.Tensor, ItemList]]:
    """
    Returns `x` and `y` data from various databunch subclasses.
    """
    dl = data.dl(ds_type=ds_type)

    if isinstance(data, TabularDataBunch):
        if dl.x.codes is None:
            x = dl.x.conts
        else:
            # todo concatenate categoricals
            x = dl.x.conts
        y = None  # todo return dl.y.data if
        return x, y
    else:
        print(f'DataBunch subclass {data.__class__.__name__} not supported directly; defaulting to X and Y')
        return dl.x, dl.y


def get_xy_batched(data: DataBunch, ds_type: DatasetType = DatasetType.Train) -> Generator:
    for xb, yb in data.dl(ds_type=ds_type):
        yield xb, yb


def set_xy(data: DataBunch, x: torch.Tensor, y: torch.Tensor, ds_type: DatasetType = DatasetType.Train):
    raise NotImplementedError
