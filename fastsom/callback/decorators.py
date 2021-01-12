import functools as ft
import warnings
from typing import Callable, Type

from fastai.callback.core import Callback


def training_only(fn: Callable):
    """
    Decorator for callback methods.
    Only calls decorated function if the callback is in training mode.
    """
    if hasattr(fn, '__name__') and fn.__name__ in ['before_epoch', 'after_epoch', 'before_fit', 'after_fit']:
        warnings.warn('Incompatible callback method decorated with @training_only: {fn.__name__}')

    @ft.wraps(fn)
    def _inner(self, *args, **kwargs):
        if not self.training:
            return
        return fn(self, *args, **kwargs)
    return _inner


def no_export(cls: Type[Callback]):
    """
    Decorator for callback classes.
    Marks the decorated callback not to be exported
    with the Learner.
    """
    old_init = cls.__init__

    @ft.wraps(old_init)
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        setattr(self, 'is_exportable', False)
    cls.__init__ = new_init
    return cls
