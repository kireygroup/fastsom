"""
"""
import torch
from torch import Tensor
from typing import Tuple


from .som import Som


class CategoricalSom(Som):
    "SOM subclass that works on mixed continuous/categorical datasets."

    def __init__(self, cat_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cat_size = cat_size

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = self._to_device(x)
        if self.weights.device != self.device:
            self.weights = self._to_device(self.weights)

        # Retrieve categorical and continuous tensors        ""

        w_cat, w_cont = self._split_cat_cont(self.weights)
        cont_f_ratio = (x_cont.t().max(-1)[0] -  x_cont.t().min(-1)[0]).abs()

        # Calculate continuous distance and scale it 
        d_cont = self.expanded_op(x_cont, w_cont.view(-1, x_cont.shape[-1]), lambda a, b: (a - b))
       
        # Calculate 
        d_cat = self.expanded_op(x_cat, w_cat.view(-1, x_cat.shape[-1]), lambda a, b: (a != b).float(), interleave=True, device=self.device)

        # Aggregate back the two partial distances
        distances = torch.cat([d_cat, d_cont.abs() / cont_f_ratio], dim=-1).sum(-1).div(x.shape[-1])

        bmu_indices = self.find_bmus(distances)

        if self.training:
            self._diffs = torch.cat([d_cat, d_cont], dim=-1)
            self._bmus = bmu_indices

        return bmu_indices

    def _split_cat_cont(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        "Splits `x` into continuous and categorical features."
        return x.split([self.cat_size, x.shape[-1] - self.cat_size], dim=-1)
    