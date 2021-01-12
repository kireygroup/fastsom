import torch
from fastai.tabular.all import Normalize

__all__ = ["norm", "denorm", "norm_with_proc", "denorm_with_proc"]


def norm(x: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return (x - means) * stds


def denorm(x: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return stds * x + means


def norm_with_proc(x: torch.Tensor, proc: Normalize) -> torch.Tensor:
    return norm(x, torch.tensor(list(proc.means.values())), torch.tensor(list(proc.stds.values())))


def denorm_with_proc(x: torch.Tensor, proc: Normalize) -> torch.Tensor:
    return denorm(x, torch.tensor(list(proc.means.values())), torch.tensor(list(proc.stds.values())))
