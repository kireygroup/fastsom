"""
This module contains classes to perform feature conversion
from categorical to continuous and back.
"""
import torch
import numpy as np
from torch import Tensor
from typing import Callable, Union, Optional, List
from enum import Enum

from gensim.models import FastText

from ..core import enum_eq, ifindict

__all__ = [
    'CatEncoder',
    'CatEncoderType',
    'CatEncoderTypeOrString',
    'FastTextCatEncoder',
    'OneHotCatEncoder',
    'get_cat_encoder',
]


class CatEncoder(Callable):
    """Transforms categorical features into continuous and back."""

    def __init__(self, cat_names: List[str], cont_names: List[str]) -> None:
        self.cat_names, self.cont_names = cat_names, cont_names

    def fit(self, x_cat: np.ndarray, **kwargs) -> None:
        """Uses `x_cat` as a base for this CatEncoder."""
        raise NotImplementedError

    def make_continuous(self, x_cat: np.ndarray) -> Tensor:
        """Transforms `x_cat` using this CatEncoder's state."""
        raise NotImplementedError

    def make_categorical(self, x_cont: Tensor) -> np.ndarray:
        """Transforms `x_cont` back into categorical using this CatEncoder's state."""
        raise NotImplementedError

    def __call__(self, x_cat: np.ndarray) -> Tensor:
        """Transforms `x_cat` using this CatEncoder's state."""
        return self.make_continuous(x_cat)


class FastTextCatEncoder(CatEncoder):
    """
    Uses a small `FastText` model to transform categorical features into continuous ones.

    Parameters
    ----------
    cat_names : List[str]
        The categorical feature names.
    cont_names : List[str]
        The continuous feature names.
    """

    def __init__(self, cat_names: List[str], cont_names: List[str], *args, **kwargs) -> None:
        super().__init__(cat_names, cont_names)
        self._ft = None
        self.cat_names = None
        self.emb_size = -1
        self.rec = args[0]
        self.iter = ifindict(kwargs, 'iter', 5)
        self.bs = ifindict(kwargs, 'bs', 32)
        self.n_categories = []

    def fit(self, x_cat: np.ndarray, **kwargs) -> None:
        """
        Trains a small `FastText` model on `x_cat`, for later use in encoding/decoding.

        Parameters
        ----------
        x_cat : np.ndarray
            The numpy matrix of categorical variables. We use Numpy since Tensors don't support strings.
        """
        self.cat_names = kwargs['cat_names']
        sentences = self._build_sentences(x_cat, self.cat_names)
        self.emb_size = self._get_embed_size(sentences)
        self.n_categories = [self.emb_size for _ in self.cat_names]
        self.rec['sentences'] = sentences
        self.rec['emb_size'] = self.emb_size
        self._ft = FastText(sentences=sentences.tolist(), size=self.emb_size,
                            window=x_cat.shape[-1], batch_words=self.bs, min_count=1, sample=0, sg=1, workers=10, iter=self.iter)

    def make_continuous(self, x_cat: np.ndarray) -> Tensor:
        """
        Performs string-to-vector encoding of `x_cat` for each column using the information stored during `fit`.

        Parameters
        ----------
        x_cat : np.ndarray
            The numpy matrix of categorical variables. We use Numpy since Tensors don't support strings.
        """
        sentences = self._build_sentences(x_cat, self.cat_names)
        self.rec['x_cat'] = sentences
        return torch.tensor([np.concatenate([self._ft.wv[word] for word in s]) for s in sentences])

    def make_categorical(self, x_cont: Tensor) -> np.ndarray:
        """
        Performs vector-to-string conversion of `x_cont` using the information stored during `fit`.

        Parameters
        ----------
        x_cont : Tensor
            The Tensor of embeddings.
        """
        return np.array([[self._ft.wv.similar_by_vector(vec[i*self.emb_size:(i+1)*self.emb_size].numpy())[0][0] for i in range(len(self.cat_names))] for vec in x_cont])

    def _build_sentences(self, x_cat: np.ndarray, cat_names) -> list:
        return np.array([[f'{c}_{v}' for c, v in zip(cat_names, sentence)] for sentence in x_cat.astype('str')])

    def _get_embed_size(self, sentences: np.ndarray) -> int:
        feature_embeds = [np.ceil(len(np.unique(sentences[:, col]))**0.25) for col in range(len(self.cat_names))]
        return int(np.ceil(np.mean(feature_embeds)))


class OneHotCatEncoder(CatEncoder):
    """
    Uses One-Hot encoding to transform categorical features into continous ones.

    Parameters
    ----------
    cat_names : List[str]
        The categorical feature names.
    cont_names : List[str]
        The continuous feature names.
    """

    def __init__(self, cat_names: List[str], cont_names: List[str], *args, **kwargs) -> None:
        super().__init__(cat_names, cont_names)
        self.n_categories = []

    def fit(self, x_cat: np.ndarray, **kwargs) -> None:
        """
        Performs one-hot encoding of `x_cat` for each column, storing
        information about the number of categories.

        Parameters
        ----------
        x_cat : np.ndarray
            The numpy matrix of categorical variables. We use Numpy since Tensors don't support strings.
        """
        t = torch.tensor(x_cat).long()
        ret = torch.tensor([]).long()
        self.n_categories = []
        for idx in range(t.shape[-1]):
            oh = torch.nn.functional.one_hot(t[:, idx])
            self.n_categories.append(oh.shape[-1])
            ret = torch.cat([ret, oh.view(x_cat.shape[0], -1)], dim=-1)

    def make_continuous(self, x_cat: np.ndarray) -> Tensor:
        """
        Performs one-hot encoding of `x_cat` for each column using the information stored during `fit`.

        Parameters
        ----------
        x_cat : np.ndarray
            The numpy matrix of categorical variables. We use Numpy since Tensors don't support strings.
        """
        t = torch.tensor(x_cat).long()
        ret = torch.tensor([]).long()
        for idx in range(t.shape[-1]):
            oh = torch.nn.functional.one_hot(t[:, idx])
            if oh.shape[-1] < self.n_categories[idx]:
                oh = torch.cat([oh, torch.zeros(oh.shape[0], self.n_categories[idx] - oh.shape[-1]).long()], dim=-1)
            ret = torch.cat([ret, oh], dim=-1)
        return ret.float()

    def make_categorical(self, x_cont: Tensor) -> np.ndarray:
        """
        Performs one-hot decoding of `x_cont` using the information stored during `fit`.

        Parameters
        ----------
        x_cont : Tensor
            The Tensor of one-hot encoded values.
        """
        ret = torch.tensor([]).long()
        idx = 0
        for n_oh in self.n_categories:
            cats = x_cont[:, idx:idx+n_oh].argmax(-1).unsqueeze(1)
            ret = torch.cat([ret, cats], dim=-1).long()
            idx += n_oh
        return ret.float()


class CatEncoderType(Enum):
    """Enum used to pick CatEncoders."""
    FAST_TEXT = 'fasttext'
    ONE_HOT = 'onehot'


CatEncoderTypeOrString = Union[CatEncoderType, str]


def get_cat_encoder(tfm: CatEncoderTypeOrString, *tfm_args, **tfm_kwargs) -> Optional[CatEncoder]:
    """Returns the correct CatEncoder."""
    if enum_eq(CatEncoderType.FAST_TEXT, tfm):
        return FastTextCatEncoder(*tfm_args, **tfm_kwargs)
    elif enum_eq(CatEncoderType.ONE_HOT, tfm):
        return OneHotCatEncoder(*tfm_args, **tfm_kwargs)
    else:
        return None
