import torch
from torch import Tensor


class UnsupervisedDataset():
    "Represents a dataset without targets."

    def __init__(self, train: Tensor, valid: Tensor, bs: int = 64, norm: bool = True, min: int = 0):
        self.train, self.valid = train, valid
        self.bs = bs
        self.use_cuda = torch.cuda.is_available()
        self.train = self._to_device(self.train)
        self.valid = self._to_device(self.valid)
        self.min = min
        if norm:
            self._normalize()
        

    def _to_device(self, a: Tensor) -> Tensor:
        "Moves a tensor to the correct device"
        return a.cuda() if self.use_cuda else a.cpu()

    @classmethod
    def create(cls, train: Tensor, valid: Tensor, bs: int = 64, norm: bool = True, min: int = 0):
        return cls(train, valid, bs=bs, norm=norm, min=min)

    def _normalize(self):
        self.train = (1 - self.min) * (self.train - self.train.min()) / (self.train.max() - self.train.min()) + self.min
        self.valid = (1 - self.min) * (self.valid - self.train.min()) / (self.train.max() - self.train.min()) + self.min

__all__ = [
    "UnsupervisedDataset",
]
