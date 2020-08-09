from gc import collect
from torch.cuda import empty_cache

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
import numpy as np

__all__ = ["dp", "clear_memory", "T"]


def dp(im):
    """
    Displays an image if the program is running in a notebook
    otherwise fails without crashing the program
    """
    if isinstance(im, np.ndarray):
        im = Tensor(im)
    if isinstance(im, Tensor):
        if len(im.shape) == 4:
            im = im[0]
            print("Showing first image")
        im = to_pil_image(im)

    try:
        display(im)
    except Exception as e:
        print(e)


def clear_memory():
    "Clears the memory of GPU"
    collect()
    empty_cache()
    try:
        1 / 0
    except:
        pass


# Easy access to make a tensor quickly
T = torch.tensor
