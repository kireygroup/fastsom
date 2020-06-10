import torch
import numpy as np
import pandas as pd

from typing import List, Generator, Iterable
from fastai.tabular import TabularProc
from fastsom.core import slices, find


__all__ = [
    'ToBeContinuousProc',
    'OneHotEncode',
    'Vectorize',
    'calc_vector_size',
]


class ToBeContinuousProc(TabularProc):
    """
    Placeholder class for `TabularProc`s that convert cat_names into cont_names.
    Tells `MyTabularProcessor` to ignore cat_names and move own cat_names into cont_names.
    Also defines interface method to backward-encode values.
    """
    transformed_cat_names = None
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
        transformed_cat_names = []
        self.n_categories = []
        self.original_cat_names = self.cat_names.copy()
        self.original_cont_names = self.cont_names.copy()
        for cat_col in self.cat_names:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            df[dummies.columns.values] = dummies
            transformed_cat_names += dummies.columns.values.tolist()
            self.n_categories.append(len(dummies.columns))
        self.cat_names = transformed_cat_names
        self.transformed_cat_names = transformed_cat_names

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
        self.cat_names = self.transformed_cat_names

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


def calc_vector_size(df: pd.DataFrame, cat_names: List[str]) -> int:
    """
    Calculates the appropriate FastText vector size for categoricals in `df`.

    https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    cat_names : List[str]
        The list of categorical column names
    """
    n_categories = sum([len(df[col].unique()) for col in cat_names])
    return int(n_categories ** 0.25)


class Vectorize(ToBeContinuousProc):
    """
    Uses FastText to generate unsupervised embeddings from
    variables in the training set.
    """
    _embedding_model = None
    _category_values = None

    def apply_train(self, df):
        self._check_module()
        self.original_cat_names = self.cat_names.copy()
        self.original_cont_names = self.cont_names.copy()
        self._init_embedding_model(df)
        print('Applying transforms...')
        # store available values for each column
        self._category_values = {col: list(set(df[col].unique().astype(str)) - set(['nan'])) for col in self.cat_names}
        # apply train is the same as apply test + model training
        self.apply_test(df)

    def apply_test(self, df: pd.DataFrame):
        preds = list(self._sentences_to_vecs(self._get_sentences(df)))
        vector_sizes = [range(self._embedding_model.vector_size) for _ in range(len(self.cat_names))]
        transformed_cat_names = np.array([[f'{col}_feature{i+1}' for i in r]
                                          for col, r in zip(self.cat_names, vector_sizes)]).flatten().tolist()
        self.cont_names = transformed_cat_names + self.cont_names
        self.cat_names = []
        self.transformed_cat_names = transformed_cat_names
        df[transformed_cat_names] = pd.DataFrame(preds, index=df.index)

    def apply_backwards(self, data: torch.Tensor) -> np.ndarray:
        """Applies the inverse transform on `data`."""
        return np.array(self._vecs_to_sentences(data.cpu().numpy()))

    def _check_module(self) -> None:
        """Ensures that the optional dependencies for the embedding model are installed."""
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(f'You need to install gensim to use the {self.__class__.name__} \
                transform. Please run `pip install gensim` and try again.')

    def _init_embedding_model(self, df: pd.DataFrame):
        """Creates and trains the unsupervised embedding model."""
        if self._embedding_model is None:
            from gensim.models import FastText
            vector_size = calc_vector_size(df, self.cat_names)
            print('Training unsupervised embeddings model...')
            self._embedding_model = FastText(size=vector_size, batch_words=1_000, min_count=1, sample=0, workers=10)  # , min_n=2, max_n=3)
            self._embedding_model.window = len(self.cat_names)
            self._embedding_model.build_vocab(sentences=self._get_sentences(df))
            self._embedding_model.train(sentences=self._get_sentences(df), total_examples=df.shape[0], epochs=8)

    def _sentences_to_vecs(self, sentences: Iterable[List[str]]) -> Generator[List[float], None, None]:
        """Returns FastText vectors for each sentence in `sentences`."""
        for s in sentences:
            yield np.concatenate([self._embedding_model.wv[word] for word in s])

    def _vecs_to_sentences(self, vectors: np.ndarray) -> List[List[str]]:
        """Returns best matching word for each vector."""
        rows = []
        topn = 1000
        for values, col in zip(slices(vectors.transpose(), self.vector_size), self.original_cat_names):
            row = []
            for vec in np.array(values).transpose():
                words = self._embedding_model.wv.similar_by_vector(vec, topn=topn)
                match = find(words, lambda w: w[0].split('__')[-1] in self._category_values[col])
                if match is None:
                    match = find(words, lambda w: col in w[0])
                row.append(match[0].split('__')[-1] if match is not None else 'None')
            rows.append(row)
        ret = np.array(rows)
        print(ret.shape, ret.transpose().shape)
        return ret.transpose()

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

    @property
    def is_mixed(self) -> bool:
        """Checks if this transform encodes all features or if it is mixed."""
        return len(self.original_cont_names) > 0

    @property
    def vector_size(self) -> int:
        """Returns the configured vector size."""
        return self._embedding_model.vector_size
