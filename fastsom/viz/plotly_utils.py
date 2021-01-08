from typing import Union

import numpy as np
import plotly.graph_objects as go
import torch
from fastcore.meta import delegates

__all__ = [
    'scatter',
    'show_figure',
]


@delegates(go.Scatter.__init__)
def scatter(data: Union[torch.Tensor, np.ndarray], **kwargs) -> go.Scatter:
    """Creates an instance of `plotly.graph_objects.Scatter` appropriate to the `data` shape."""
    if data.shape[-1] == 1:
        return go.Scatter(x=data[..., 0], **kwargs)
    elif data.shape[-1] == 2:
        return go.Scatter(x=data[..., 0], y=data[..., 1], **kwargs)
    elif data.shape[-1] == 3:
        return go.Scatter3d(x=data[..., 0], y=data[..., 1], z=data[..., 2], **kwargs)
    else:
        raise ValueError('Unable to plot data with more than 3 dimensions. Reduce dimension with PCA and try again')


def show_figure(f: Union[go.Figure, go.FigureWidget]):
    """
    Tries displaying a Plotly Figure (or FigureWidget) in interactive mode.
    Fallbacks to `f.show()` if not in an IPython environment.
    """
    try:
        from IPython.display import display
        display(f)
    except Exception:
        f.show()
