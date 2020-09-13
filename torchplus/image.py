from __future__ import annotations

from io import BytesIO
from os import path

import numpy as np
import torch
from cachecontrol import CacheControl
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
from requests.sessions import Session
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor

from .nb import dp

session = CacheControl(Session())

__all__ = ["Image"]


class Image:
    """
    A class for easily working with images of various formats

    Can be created from a variety of different classes but first
    class with tensors

    Provides some useful transformation and conversions to 
    base64, a popular web format
    """

    def __init__(self, source):
        self.source = source
        if isinstance(source, str):
            if "http" in source[:5]:
                image = self._download_image()
            elif path.exists(path.expanduser(source)):
                image = PILImage.open(source)
            else:
                raise Exception("Could not load image")
        elif isinstance(source, np.ndarray):
            image = PILImage.fromarray(source)
            self.source = "numpy"
        elif isinstance(source, Tensor):
            image = to_pil_image(source)
            self.source = "tensor"
        elif isinstance(source, PILImageType):
            image = source
        else:
            raise Exception("Invalid Image")
        self._pil: PILImageType = image
        self._data = source if isinstance(source, Tensor) else to_tensor(image)

    def __repr__(self):
        dp(self.pil)
        return f"Image <{list(self.data.shape)}> <{self.source}>"

    def _download_image(self):
        "Attempts to download the image"
        result = session.get(self.source)
        if result.status_code != 200:
            raise Exception(f"Could not download {self.source}\n {result.content}")
        im = result.content
        im = PILImage.open(BytesIO(im))
        if not isinstance(im, PILImageType):
            raise Exception(f"File downloaded was not an image: {self.source}")
        return im

    @property
    def data(self) -> Tensor:
        "The raw tensor of the Image"
        return self._data

    @data.setter
    def data(self, v: Tensor):
        self._pil = to_pil_image(v)
        self._data = v

    @property
    def pil(self) -> PILImageType:
        return self._pil

    @pil.setter
    def pil(self, v: PILImageType):
        self._pil = v
        self._data = to_tensor(v)

    def resize(self, w, h=None):
        "Resize the image to the give variables"
        if h == None:
            h = w / self.w * self.h
        self.pil = self.pil.resize((int(w), int(h)))

    def scale(self, scale):
        "Scale the image equally in each dimension"
        size = self.w * scale, self.h * scale
        self.resize(*size)

    def add_alpha(self):
        "Adds a 4th alpha channel dimension"
        if self.data.shape[0] == 4:
            return
        t = torch.cat([self.data, torch.ones(1, *self.data.shape[1:])])
        self.data = t

    @property
    def width(self):
        return self.data.shape[2]

    @property
    def w(self):
        return self.width

    @property
    def height(self):
        return self.data.shape[1]

    @property
    def h(self):
        return self.height

    @property
    def size(self):
        "Quick access to the size of the image"
        return torch.tensor([self.w, self.h])

    @property
    def base64(self):
        "Base64 Encoding of the Image"
        bts = BytesIO()
        im.save(bts, format="PNG")
        bts = bytes(bts.getvalue())
        return base64.encodebytes(bts)
