import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits

__all__ = ["bcewl", "mse", "accuracy", "precision", "recall"]


def bcewl(i: Tensor, t: Tensor):
    i = i.flatten()
    t = t.flatten()
    return binary_cross_entropy_with_logits(i, t)


def mse(i: Tensor, t: Tensor):
    "Mean sqaured error"
    return torch.pow(i - t, 2).mean()


def accuracy(margin):
    def _inner(i: Tensor, t: Tensor):
        i[i < margin] = 0
        i[i >= margin] = 1
        return (i == t).float().mean()

    return _inner


def precision(margin):
    def _inner(i, t):
        i[i < margin] = 0
        i[i >= margin] = 1
        return ((i == 1) == (t == 1)).sum() / (t == 1).numels()

    return _inner


def recall(margin):
    def _inner(i: Tensor, t: Tensor):
        i[i < margin] = 0
        i[i > margin] = 1
        return ((i == 1) == (t == 1)).sum() / (i == 1).numels()
