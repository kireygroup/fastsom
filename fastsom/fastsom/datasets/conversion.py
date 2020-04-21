"""
This module contains utility methods to convert
Fastai `DataBunch` classes into `UnsupervisedDataBunch`.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Union
from fastai.tabular import TabularDataBunch
from pandas import DataFrame

from .datasets import UnsupervisedDataBunch
from .cat_encoders import CatEncoder, get_cat_encoder, CatEncoderTypeOrString, FastTextCatEncoder

from ..core import ifnone


__all__ = [
    'tabular_ds_to_lists',
    'to_unsupervised_databunch',
    'dataframe_fill_unknown',
]


def tabular_ds_to_lists(ds: Dataset):
    """
    Converts a Dataset from a `TabularDataBunch` into two lists of elements.

    Parameters
    ----------
    ds : Dataset
        The TabularDataBunch Dataset containing categorical and continuous elements.
    """
    x_cat = torch.cat([el[0].data[0].long().unsqueeze(0) for el in ds], dim=0)
    x_cont = torch.cat([el[0].data[1].float().unsqueeze(0) for el in ds], dim=0)
    return x_cat, x_cont


def dataframe_fill_unknown(df: DataFrame, unknown_cat: str = '<unknown>') -> DataFrame:
    """
    For each column in DataFrame `df`, adds a category with value `unknown_cat` and uses it to fill n/a values.

    Parameters
    ----------
    df : pandas.DataFrame
        The source DataFrame.
    `unknown_cat : str default='<unknown>'
        The string to be used when replacing N/A values.
    """
    for column in df.columns:
        if unknown_cat not in df[column].cat.categories:
            df[column] = df[column].cat.add_categories(unknown_cat)
            df[column] = df[column].fillna(unknown_cat)
    return df


def to_unsupervised_databunch(data: TabularDataBunch, bs: Optional[int] = None, cat_enc: Union[CatEncoderTypeOrString, CatEncoder] = 'onehot', **kwargs) -> UnsupervisedDataBunch:
    """
    Transforms a `TabularDataBunch` into an `UnsupervisedDataBunch`.

    Parameters
    ----------
    data : TabularDataBunch
        The source DataBunch.
    bs : int = None
        The output DataBunch batch size. Defaults to source DataBunch batch size.
    cat_enc : Union[CatEncoderTypeOrString, CatEncoder] default='onehot'
        Categorical encoding strategy, used for both cat-to-cont and cont-to-cat conversion.

    """
    train_x_cat, train_x_cont = tabular_ds_to_lists(data.train_ds)
    valid_x_cat, valid_x_cont = tabular_ds_to_lists(data.valid_ds)

    tfm = cat_enc if isinstance(cat_enc, CatEncoder) else get_cat_encoder(cat_enc, data.cat_names, data.cont_names)
    if isinstance(tfm, FastTextCatEncoder):
        # Pass string values to FastTextCatEncoder
        train_x_cat = dataframe_fill_unknown(data.train_ds.inner_df[data.cat_names]).values
        valid_x_cat = dataframe_fill_unknown(data.valid_ds.inner_df[data.cat_names]).values
        tfm.fit(train_x_cat, cat_names=data.cat_names)
        train_x_cat = tfm.make_continuous(train_x_cat)
        valid_x_cat = tfm.make_continuous(valid_x_cat)
    else:
        # Pass categories to other transformers
        tfm.fit(train_x_cat, cat_names=data.cat_names)
        train_x_cat = tfm.make_continuous(train_x_cat)
        valid_x_cat = tfm.make_continuous(valid_x_cat)

    train_ds = torch.cat([train_x_cat.float(), train_x_cont], dim=-1) if len(data.train_ds) > 0 else None
    valid_ds = torch.cat([valid_x_cat.float(), valid_x_cont], dim=-1) if len(data.valid_ds) > 0 else None

    bs = ifnone(bs, data.batch_size)

    return UnsupervisedDataBunch(train_ds, valid=valid_ds, bs=bs, cat_enc=tfm, **kwargs)


TabularDataBunch.to_unsupervised_databunch = to_unsupervised_databunch
