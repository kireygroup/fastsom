import logging

from .core import ifnone


__all__ = [
    "safeget",
    "getname",
    "get_logger",
]


def safeget(o: any, attr: str) -> any:
    """
    
    Parameters
    ----------
    
    """
    try:
        return getattr(o, attr)
    except Exception:
        return None


def getname(o: any) -> str:
    """
    
    Parameters
    ----------
    
    """
    o = ifnone(safeget(o, "__class__"), o)
    return o.__name__


def get_logger(o: any) -> logging.Logger:
    """

    Parameters
    ----------

    """
    name = o if isinstance(o, str) else getname(o)
    return logging.getLogger(name)
