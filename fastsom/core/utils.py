"""
This file contains various project-wide utilities.

"""

from functools import reduce
from typing import Callable, Collection, Generator, Iterable

__all__ = [
    "ifnone",
    "ifindict",
    "is_iterable",
    "listify",
    "setify",
    "compose",
    "enum_eq",
    "find",
    "any_matches",
    "slices",
]


def ifnone(o: any, default: any) -> any:
    """
    Returns `o` if it is not `None`; returns `default` otherwise.

    Parameters
    ----------
    o : any
        The value to be checked.
    default : any
        The default return value when `o` is `None`.

    Examples
    --------
    >>> ifnone(5, 'default-value')
    5

    >>> ifnone(None, 42)
    42
    """
    return o if o is not None else default


def ifindict(d: dict, k: str, default: any) -> any:
    """
    Returns the value of `k` inside `d` if it exists; returns `default` otherwise.

    Parameters
    ----------
    d : dict
        The source dict.
    k : str
        The lookup key
    default : any
        The default return value when `k` is not in `d`.

    Examples
    --------
    >>> d = dict(a=5, b=6)
    >>> ifindict(d, 'a', 'default-value')
    5
    >>> ifindict(d, 'some_invalid_key', 42)
    42
    """
    return d[k] if k in d else default


def is_iterable(o: any) -> bool:
    """
    Checks if `o` is iterable

    Parameters
    ----------
    o : any
        The value to be checked.

    Examples
    --------
    >>> is_iterable(list(range(5)))
    True

    >>> is_iterable(5)
    False

    >>> is_iterable('hello world')
    True

    >>> is_iterable(None)
    False
    """
    try:
        _ = iter(o)
    except TypeError:
        return False
    return True


def listify(o: any) -> list:
    """
    Turns `o` into a list.

    Parameters
    ----------
    o : any
        The value to be listified.

    Examples
    --------
    >>> listify(None)
    []

    >>> listify('mystring')
    ['m', 'y', 's', 't', 'r', 'i', 'n', 'g']

    >>> listify(5)
    5
    """
    if o is None:
        return []
    if not is_iterable(o):
        return o
    return o if isinstance(o, list) else list(o)


def setify(o: any) -> set:
    """
    Turns `o` into a set.

    Parameters
    ----------
    o : any
        The value to be setified.

    Examples
    --------
    >>> setify(5)
    5

    >>> setify(None)
    set()

    >>> setify('hello world')
    {'h', 'l', 'd', 'e', 'w', 'r', ' ', 'o'}

    >>> setify([1, 2, 2, 3, 3, 4, 5])
    {1, 2, 3, 4, 5}
    """
    if o is None:
        return set()
    if not is_iterable(o):
        return o
    return o if isinstance(o, set) else set(listify(o))


def compose(
    x: any, fns: Collection[Callable], order_key: str = "_order", **kwargs
) -> any:
    """
    Applies each function in `fns` to the output of the previous function.\n
    Function application starts from `x`, and uses `order_key` to sort the `fns` list.

    Parameters
    ----------
    x : any
        The base function parameter.
    fns : Collection[Callable]
        The collection of functions.
    order_key : str default='_order'
        The key to be used to sort the functions.
    """
    sorted_fns = sorted(listify(fns), key=lambda o: getattr(o, order_key, 0))
    return reduce(lambda x, fn: fn(x), sorted_fns, x)


def enum_eq(enum, value) -> bool:
    """
    Checks equality of `enum` and `value`.

    Parameters
    ----------
    enum : Enum
        The enumerator instance
    value : any
        The value or enumerator instance to be compared.
    """
    return enum == value or enum.value == value


def any_matches(iterable: Iterable, cond_fn: Callable = bool) -> bool:
    """
    Returns `True` if `cond_fn` is `True`
    for one or more elements in `iterable`.

    Parameters
    ----------
    iterable : Iterable
        The iterable to be checked
    cond_fn : Callable
        The condition function to be applied
    """
    return any(map(cond_fn, iterable))


def find(iterable: Iterable, cond_fn: Callable, last: bool = False) -> any:
    """
    Finds the first (or last, if `last=True`) element of `iterable`
    for which `cond_fn` is `True`.

    Parameters
    ----------
    iterable : Iterable
        The iterable to be checked
    cond_fn : Callable
        The conditional function, applied on each element
    last : bool default=False
        Whether to returns the last matching element instead of the first
    """
    i = reversed(iterable) if last else iterable
    for item in i:
        if cond_fn(item):
            return item
    return None


def slices(iterable: Iterable, size: int) -> Generator[any, None, None]:
    """
    Returns a generator of slices of size `size` over `iterable`.

    Parameters
    ----------
    iterable: Iterable
        The elements to be sliced
    size: int
        The slice size
    """
    items = []
    for item in iterable:
        items.append(item)
        if len(items) == size:
            yield items
            items = []
    if len(items) > 0:
        yield items
