"""
This module contains a customization of some Fastai.tabular classes.
This allows for additional transforms (e.g. OneHotEncode) to be defined,
while maintaining the same data block syntax as the original API.
"""
import numpy as np

from fastai.tabular import TabularLine, TabularList, TabularProcessor, TabularProc, TabularDataBunch, OrderedDict

from fastsom.core import ifnone, find
from .transform import ToBeContinuousProc


__all__ = ['SomTabularList', 'CustomTabularProcessor']


class CustomTabularProcessor(TabularProcessor):
    """
    Custom `TabularProcessor` that stores information about
    categorical/continuous feature names that go into each
    `TabularProc`, allowing for additional procs to be defined
    while maintaining compatibility with existing transforms.
    """
    # cat_names/cont_names for each proc
    _stages = None

    def process(self, ds: TabularList):
        """
        Processes a dataset, either train_ds or valid_ds.

        Parameters
        ----------
        ds: TabularList
            The dataset
        """
        if ds.inner_df is None or ds.inner_df.empty:
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
                print(proc.__class__.__name__, len(proc.cat_names), len(proc.cont_names))
                proc(ds.inner_df, test=True)
            else:
                # otherwise, we need to instantiate it first
                # cat and cont names may have been changed by transform (like Fill_NA)
                self._stages[proc.__name__] = (ds.cat_names.copy(), ds.cont_names.copy())
                print(proc.__name__, len(ds.cat_names), len(ds.cont_names))
                proc = proc(ds.cat_names, ds.cont_names)
                proc(ds.inner_df)
                ds.cat_names, ds.cont_names = proc.cat_names, proc.cont_names
                self.procs[i] = proc

        # If any of the TabularProcs was a ToBeContinuousProc, we need
        # to move all cat names from that proc to cont names
        last_tobecont_proc = find(self.procs, lambda p: isinstance(p, ToBeContinuousProc), last=True)
        if last_tobecont_proc is not None:
            ds.cont_names = last_tobecont_proc.cont_names  # + last_tobecont_proc.transformed_cat_names
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
        self.cat_names, self.cont_names = cat_cols, cont_cols
        ds.preprocessed = True

    def process_one(self, item: TabularLine):
        return super().process_one(item)


class SomTabularList(TabularList):
    """
    `TabularList` with bindings for `CustomTabularProcessor`.
    Can be used the same way as a regular `TabularList`, but also
    supports additional transforms for categorical features.
    """
    _bunch = TabularDataBunch
    _processor = CustomTabularProcessor
