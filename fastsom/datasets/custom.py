import torch
import numpy as np
import pandas as pd

from fastai.basic_data import DatasetType
from fastai.tabular import TabularList, TabularProcessor, TabularProc, TabularDataBunch, OrderedDict

from ..core import ifnone, find


__all__ = [
    'CustomDataBunch',
    'MyTabularProcessor',
    'MyTabularList',
    'OneHotEncode',
    'ToBeContinuousProc',
]


class ToBeContinuousProc(TabularProc):
    """
    Placeholder class for `TabularProc`s that convert cat_names into cont_names.
    Tells `MyTabularProcessor` to ignore cat_names and move own cat_names into cont_names.
    Also defines interface method to backward-encode values.
    """
    _out_cat_names = None
    original_cat_names = None
    original_cont_names = None

    def apply_backwards(self, data: torch.Tensor) -> np.ndarray:
        """Applies the inverse transformation."""
        raise NotImplementedError


class OneHotEncode(ToBeContinuousProc):
    """
    Performs One-Hot encoding of categorical values in `df`.
    """
    n_categories = None

    def apply_train(self, df: pd.DataFrame):
        out_cat_names = []
        self.n_categories = []
        self.original_cat_names = self.cat_names
        self.original_cont_names = self.cont_names
        for cat_col in self.cat_names:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            df[dummies.columns.values] = dummies
            out_cat_names += dummies.columns.values.tolist()
            self.n_categories.append(len(dummies.columns))
        self.cat_names = out_cat_names
        self._out_cat_names = out_cat_names

    def apply_test(self, df: pd.DataFrame):
        for cat_col in self.cat_names:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            df[dummies.columns.values] = dummies
        self.cat_names = self._out_cat_names

    def apply_backwards(self, data: torch.Tensor) -> np.ndarray:
        ret = torch.tensor([]).long()
        idx = 0
        for categories_count in self.n_categories:
            cats = data[:, idx:idx+categories_count].argmax(-1).unsqueeze(1)
            ret = torch.cat([ret, cats], dim=-1).long()
            idx += categories_count
        return ret.float()


class CustomDataBunch(TabularDataBunch):

    _tobecontinuous_proc = None

    def one_batch(self, ds_type=DatasetType.Train, detach=True, denorm=True, cpu=True):
        cats, conts, y = super().one_batch(ds_type=ds_type, detach=detach, denorm=denorm, cpu=cpu)
        if not len(cats):
            return conts, y
        return cats, conts, y

    def one_item(self, item, detach=False, denorm=False, cpu=False):
        cats, conts, y = super().one_item(item, detach=detach, denorm=denorm, cpu=cpu)
        if not len(cats):
            return conts, y
        return cats, conts, y


class MyTabularProcessor(TabularProcessor):

    _stages = None

    def process(self, ds):
        """Processes a dataset, either train_ds or valid_ds."""
        if ds.inner_df is None:
            ds.classes, ds.cat_names, ds.cont_names = self.classes, self.cat_names, self.cont_names
            ds.col_names = self.cat_names + self.cont_names
            ds.preprocessed = True
            return
        self._stages = ifnone(self._stages, {})
        for i, proc in enumerate(self.procs):
            if isinstance(proc, TabularProc):
                # If the process is already an instance of TabularProc,
                # this means we already ran it on the train set!
                proc.cat_names, proc.cont_names = self._stages[proc.__class__.__name__]
                proc(ds.inner_df, test=True)
            else:
                # otherwise, we need to instantiate it first
                # cat and cont names may have been changed by transform (like Fill_NA)
                self._stages[proc.__name__] = ds.cat_names, ds.cont_names
                proc = proc(ds.cat_names, ds.cont_names)
                proc(ds.inner_df)
                ds.cat_names, ds.cont_names = proc.cat_names, proc.cont_names
                self.procs[i] = proc

        # If any of the TabularProcs was a ToBeContinuousProc, we need
        # to move all cat names from that proc to cont names
        last_tobecont_proc = find(self.procs, lambda p: isinstance(p, ToBeContinuousProc), last=True)
        if last_tobecont_proc is not None:
            cat_names = last_tobecont_proc._out_cat_names
            ds.cont_names = cat_names + ds.cont_names
            ds.cat_names = []
        # original Fast.ai code to maintain compatibility
        if len(ds.cat_names) != 0:
            ds.codes = np.stack([c.cat.codes.values for n, c in ds.inner_df[ds.cat_names].items()], 1).astype(np.int64) + 1
            self.classes = ds.classes = OrderedDict({n: np.concatenate([['#na#'], c.cat.categories.values])
                                                    for n, c in ds.inner_df[ds.cat_names].items()})
            cat_cols = list(ds.inner_df[ds.cat_names].columns.values)
        else:
            ds.codes, ds.classes, self.classes, cat_cols = None, None, None, []

        # Build continuous variables
        if len(ds.cont_names) != 0:
            ds.conts = np.stack([c.astype('float32').values for n, c in ds.inner_df[ds.cont_names].items()], 1)
            cont_cols = list(ds.inner_df[ds.cont_names].columns.values)
        else:
            ds.conts, cont_cols = None, []

        ds.col_names = cat_cols + cont_cols
        ds.preprocessed = True

    def process_one(self, item):
        print('process_one')
        return super().process_one(item)


class MyTabularList(TabularList):
    _bunch = CustomDataBunch
    _processor = MyTabularProcessor
