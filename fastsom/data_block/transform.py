import torch
import numpy as np
import pandas as pd

from typing import List, Generator, Iterable
from fastai.tabular import TabularProc

from ..core import slices


__all__ = [
    'ToBeContinuousProc',
    'OneHotEncode',
    'Vectorize',
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
        """Applies the inverse transform on `data`."""
        raise NotImplementedError


class OneHotEncode(ToBeContinuousProc):
    """
    Performs One-Hot encoding of categorical values in `df`.
    """
    n_categories = None

    def apply_train(self, df: pd.DataFrame):
        """
        Applies the transform on the training set, storing
        information about the number of categories.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be transformed
        """
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
        """
        Applies the transform on the training set, using
        information about the training set.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be transformed
        """
        for cat_col in self.cat_names:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            df[dummies.columns.values] = dummies
        self.cat_names = self._out_cat_names

    def apply_backwards(self, data: torch.Tensor) -> np.ndarray:
        """
        Applies the inverse transform on `data`.

        Parameters
        ----------
        data : torch.Tensor
            The transformed data
        """
        ret = torch.tensor([]).long()
        idx = 0
        for categories_count in self.n_categories:
            cats = data[:, idx:idx+categories_count].argmax(-1).unsqueeze(1)
            ret = torch.cat([ret, cats], dim=-1).long()
            idx += categories_count
        return ret.float()


class Vectorize(ToBeContinuousProc):
    """
    Uses FastText to generate unsupervised embeddings from
    variables in the training set.
    """
    # FastText model
    _ft = None

    def apply_train(self, df):
        self._check_module()
        from gensim.models import FastText
        self.original_cat_names = self.cat_names
        self.original_cont_names = self.cont_names
        if self._ft is None:
            ft_size = 3  # TODO: use function
            self._ft = FastText(size=ft_size, batch_words=1_000, min_count=1, sample=0, workers=10)
            self._ft.window = len(self.cat_names)
            self._ft.build_vocab(sentences=self._get_sentences(df))
            print('Training unsupervised embeddings model...')
            self._ft.train(sentences=self._get_sentences(df), total_examples=df.shape[0], epochs=5)
        print('Applying transforms...')
        preds = list(self._get_preds(self._get_sentences(df)))
        ft_sizes = [range(self._ft.vector_size) for _ in range(len(self.cat_names))]
        out_cat_names = np.array([[f'{col}_feature{i+1}' for i in r] for col, r in zip(self.cat_names, ft_sizes)]).flatten().tolist()
        df[out_cat_names] = pd.DataFrame(preds, index=df.index)
        self.cat_names = out_cat_names
        self._out_cat_names = out_cat_names

    def apply_test(self, df):
        preds = list(self._get_preds(self._get_sentences(df)))
        ft_sizes = [range(self._ft.vector_size) for _ in range(len(self.cat_names))]
        out_cat_names = np.array([[f'{col}_feature{i+1}' for i in r] for col, r in zip(self.cat_names, ft_sizes)]).flatten().tolist()
        df[out_cat_names] = pd.DataFrame(preds, index=df.index)
        self.cat_names = self._out_cat_names

    def apply_backwards(self, data: torch.Tensor) -> np.ndarray:
        """Applies the inverse transform on `data`."""
        return np.array(self._inverse_preds(data.cpu().numpy()))

    def _is_mixed(self) -> bool:
        """Checks if this transform encodes all features or if it is mixed."""
        return len(self.original_cont_names) > 0

    def _check_module(self) -> None:
        """Ensures that the optional dependencies for the embedding model are installed."""
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(f'You need to install gensim to use the {self.__class__.name__} \
                transform. Please run `pip install gensim` and try again.')

    def _get_preds(self, sentences: Iterable[List[str]]) -> Generator[List[float], None, None]:
        """Returns FastText vectors for each sentence in `sentences`."""
        for s in sentences:
            yield np.concatenate([self._ft.wv[word] for word in s])

    def _inverse_preds(self, vectors: Iterable[List[float]]) -> List[List[str]]:
        """Returns best matching word for each vector."""
        return [[self._ft.wv.similar_by_vector(vec[i:i+3], topn=1)[0][0].split('__')[-1] for i in range(vec.shape[0] // 3)] for vec in vectors]

    def _get_sentences(self, df: pd.DataFrame) -> Generator[List[str], None, None]:
        """
        Builds sentences using values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be used when building sentences.
        """
        for i in range(df.shape[0]):
            yield list(map(lambda o: f'{o[1]}__{o[0]}', zip(df[self.cat_names].values[i], self.cat_names)))
