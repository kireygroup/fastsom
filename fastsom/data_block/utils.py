import torch

from typing import Union, Tuple, Generator
from fastai.basic_data import DataBunch, DatasetType
from fastai.data_block import ItemList
from fastai.tabular import TabularDataBunch


__all__ = ['get_xy', 'get_xy_batched']


def get_xy(data: DataBunch, ds_type: DatasetType = DatasetType.Train) -> Tuple[Union[torch.Tensor, ItemList], Union[torch.Tensor, ItemList]]:
    """
    Returns `x` and `y` data from various databunch subclasses.

    Parameters
    ----------
    data : DataBunch
        The Fastai DataBunch object
    ds_type : DatasetType default=DatasetType.Train
        The type of dataset to use (train/valid/test)
    """
    dl = data.dl(ds_type=ds_type)

    if isinstance(data, TabularDataBunch):
        # For some reason Fast.ai dataloaders have a size of 1 when empty
        if len(dl.x) > 1:
            x = dl.x.conts
        else:
            # In case the dataloader is empty (e.g. when loading from file)
            x = torch.tensor([])
        y = None  # todo return dl.y.data if
        return x, y
    else:
        print(f'DataBunch subclass {data.__class__.__name__} not supported directly; defaulting to X and Y')
        return dl.x, dl.y


def get_xy_batched(data: DataBunch, ds_type: DatasetType = DatasetType.Train) -> Generator:
    """
    Returns an iterator over dataset with type `ds_type` in `data`.

    Parameters
    ----------
    data : DataBunch
        The Fastai DataBunch object
    ds_type : DatasetType default=DatasetType.Train
        The type of dataset to use (train/valid/test)
    """
    for xb, yb in data.dl(ds_type=ds_type):
        yield xb, yb
