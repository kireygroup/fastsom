import functools as ft
import logging
from typing import Type

__all__ = [
    'has_logger',
]


def has_logger(cls: Type):
    """
    Decorator. Adds a `self.logger` member to `cls`
    that prefixes messages with the class name.
    """
    orig_init = cls.__init__

    @ft.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] - [%(name)s] - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger(cls.__name__)
        logger.addHandler(handler)
        self.logger = logger
        return

    cls.__init__ = __init__
    return cls
