import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image

from torchplus import Image

TEST_IMAGE_SIZE = [1920, 1081]


def test_load_file():
    im = Image("tests/test.jpg")
    assert im.size[0] == TEST_IMAGE_SIZE[0]
    assert im.size[1] == TEST_IMAGE_SIZE[1]


def test_load_file_fail():
    try:
        im = Image("./no-image-here.png")
        raise Exception("Failed when should have found")
    except:
        pass


def test_load_tensor():
    w, h = 25, 25
    im = Image(torch.ones(3, w, h))
    assert im.size[0] == w
    assert im.size[1] == h


def test_load_np():
    w, h = 25, 25
    im = Image(np.zeros((h, w, 3), dtype=np.uint8))
    assert im.size[0] == w
    assert im.size[1] == h


def test_load_pil():
    im = PILImage.open("tests/test.jpg")
    im = Image(im)
    assert im.size[0] == TEST_IMAGE_SIZE[0]
    assert im.size[1] == TEST_IMAGE_SIZE[1]


def test_set_data():
    im = Image(torch.ones(1, 10, 10))
    im.data = torch.ones(1, 20, 20)
    assert im.size[0] == 20
    assert im.size[1] == 20
    assert im.pil.size[0] == 20
    assert im.pil.size[1] == 20


def test_set_pil():
    im = Image(torch.ones(1, 10, 10))
    im.pil = to_pil_image(torch.ones(1, 20, 20))
    assert im.size[0] == 20
    assert im.size[1] == 20
    assert im.data.shape[2] == 20
    assert im.data.shape[1] == 20


def test_resize():
    im = Image(torch.ones(1, 10, 10))
    im.resize(20)
    assert im.size[0] == 20
    assert im.size[1] == 20


def test_scale():
    im = Image(torch.ones(1, 10, 10))
    im.scale(2)
    assert im.size[0] == 20
    assert im.size[1] == 20


def test_alpha():
    im = Image(torch.ones(3, 10, 10))
    im.add_alpha()
    assert im.data.shape[0] == 4


def test_alpha_uncessary():
    im = Image(torch.ones(4, 10, 10))
    im.add_alpha()
    assert im.data.shape[0] == 4


def test_base64():
    im = Image(torch.ones(4, 10, 10))
    im.add_alpha()
    assert im.data.shape[0] == 4
