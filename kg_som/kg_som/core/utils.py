"""
This file contains various project-wide utilities.

"""

from typing import Collection, Callable
from functools import reduce


__all__ = [
    "ifnone",
    "ifindict",
    "is_iterable",
    "listify",
    "setify",
    "compose",
    "enum_eq",
]

def ifnone(o: any, default: any) -> any:
    "Returns `o` if it is not `None`; returns `default` otherwise."
    return o if o is not None else default


def ifindict(d: dict, k: str, default: any) -> any:
    "Returns the value of `k` inside `d` if it exists; returns `default` otherwise."
    return d[k] if k in d else default


def is_iterable(o: any) -> bool:
    "Checks if `o` is iterable"
    try:
        _ = iter(o)
    except TypeError:
        return False
    return True


def listify(o: any) -> list:
    "Turns `o` into a list."
    if o is None:
        return []
    if not is_iterable(o):
        return o
    return o if isinstance(o, list) else list(o)


def setify(o: any) -> set:
    "Turns `o` into a set."
    if o is None:
        return set()
    if not is_iterable(o):
        return o
    return o if isinstance(o, set) else set(listify(o))


def compose(x: any, fns: Collection[Callable], order_key: str = '_order', **kwargs) -> any:
    """
    Applies each function in `fns` to the output of the previous function.\n
    Function application starts from `x`, and uses `order_key` to sort the `fns` list.

    """
    sorted_fns = sorted(listify(fns), key=lambda o: getattr(o, order_key, 0))
    return reduce(lambda x, fn: fn(x), sorted_fns, x)


def enum_eq(enum, value) -> bool:
    "Checks equality of `enum` and `value`."
    return (enum == value or enum.value == value)


