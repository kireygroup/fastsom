"""
Shamelessly stolen from https://github.com/sevamoo/SOMPY/blob/master/sompy/decorators.py
"""
import logging
from functools import wraps
from time import time


def timeit(log_level=logging.INFO, alternative_title=None):
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            ts = round(time() - t0, 6)

            title = alternative_title or f.__name__
            print(f" {title} took: {ts} seconds")
            # logging.getLogger().log(log_level, f" {title} took: {ts} seconds")

            return result

        return wrapped_f

    return wrap
