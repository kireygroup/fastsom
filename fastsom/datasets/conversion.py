"""
This module contains utility methods to convert
Fastai `DataBunch` classes into `UnsupervisedDataBunch`.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, Tuple
from fastai.tabular import TabularDataBunch
from pandas import DataFrame

from .datasets import UnsupervisedDataBunch
from .cat_encoders import CatEncoder, get_cat_encoder, CatEncoderTypeOrString, FastTextCatEncoder

from ..core import ifnone


__all__ = [
    'tabular_ds_to_lists',
    'to_unsupervised_databunch',
]


def tabular_ds_to_lists(ds: Dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a Dataset from a `TabularDataBunch` into categorical variables, continuous variables and targets.

    Parameters
    ----------
    ds : Dataset
        The TabularDataBunch Dataset containing categorical and continuous elements.
    """
    if ds is None or len(ds.x) < 2:
        return None, None, None
    x_cat = torch.cat([el[0].data[0].long().unsqueeze(0) for el in ds], dim=0)
    x_cont = torch.cat([el[0].data[1].float().unsqueeze(0) for el in ds], dim=0)
    y = torch.cat([torch.tensor(el[1].data).view(1, -1) for el in ds], dim=0)
    return x_cat, x_cont, y


def dataframe_fill_unknown(df: DataFrame, unknown_cat: str = '<unknown>') -> DataFrame:
    """
    For each column in DataFrame `df`, adds a category with value `unknown_cat` and uses it to fill n/a values.

    Parameters
    ----------
    df : pandas.DataFrame
        The source DataFrame.
    unknown_cat : str default='<unknown>'
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
    train_x_cat, train_x_cont, train_y = tabular_ds_to_lists(data.train_ds)
    valid_x_cat, valid_x_cont, valid_y = tabular_ds_to_lists(data.valid_ds)

    train_x_cat = train_x_cat if len(data.cat_names) > 0 else torch.tensor([])
    valid_x_cat = valid_x_cat if len(data.cat_names) > 0 else torch.tensor([])

    tfm = cat_enc if isinstance(cat_enc, CatEncoder) else get_cat_encoder(cat_enc, data.cat_names, data.cont_names)
    if len(data.cat_names) > 0:
        if isinstance(tfm, FastTextCatEncoder):
            # Pass string values to FastTextCatEncoder
            train_x_cat = dataframe_fill_unknown(data.train_ds.inner_df[data.cat_names]).values
            tfm.fit(train_x_cat.numpy(), cat_names=data.cat_names)
            train_x_cat = tfm.make_continuous(train_x_cat.numpy())
            if valid_x_cat is not None and len(valid_x_cat) > 0:
                valid_x_cat = dataframe_fill_unknown(data.valid_ds.inner_df[data.cat_names]).values
                valid_x_cat = tfm.make_continuous(valid_x_cat.numpy())
        else:
            # Pass categories to other transformers
            tfm.fit(train_x_cat.numpy(), cat_names=data.cat_names)
            train_x_cat = tfm.make_continuous(train_x_cat.numpy())
            if valid_x_cat is not None:
                valid_x_cat = tfm.make_continuous(valid_x_cat.numpy())

    train_x = torch.cat([train_x_cat.float(), train_x_cont], dim=-1) if len(data.train_ds) > 1 else torch.tensor([])
    valid_x = torch.cat([valid_x_cat.float(), valid_x_cont], dim=-1) if valid_x_cat is not None and len(data.valid_ds) > 1 else torch.tensor([])
    train_y = ifnone(train_y, torch.tensor([]))
    valid_y = ifnone(valid_y, torch.tensor([]))

    bs = ifnone(bs, data.batch_size)

    return UnsupervisedDataBunch((train_x, train_y), valid=(valid_x, valid_y), bs=bs, cat_enc=tfm, **kwargs)


TabularDataBunch.to_unsupervised_databunch = to_unsupervised_databunch
