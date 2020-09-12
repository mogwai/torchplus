from torchplus.data.cache import LazyRAM
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_pil_image, to_tensor

def test_lazyram_can_decorate():
    "Using LazyRAM as a decoration works"

    x = 1

    @LazyRAM()
    class Test:
        def __getitem__(self, i):
            return i * x

    assert Test()[1] == 1
    x = 2
    assert Test()[1] == 1


def test_lazyram_can_instantiate():
    "The cache value doesn't change if we manually change it"

    x = [1]
    arr = LazyRAM(x)

    assert arr[0] == 1
    x[0] = 2
    assert arr[0] == 1


def test_lazyram_dict():
    "Check that we can use a dict"
    arr = LazyRAM({"a": 1})

    assert arr['a'] == 1
    arr["a"] = 2
    assert arr['a'] == 1


def test_lazyram_dataloader():
    "Check that we can use a custom DataLoader"
    fake_images = torch.ones(10, 3, 50, 50)
    
    @LazyRAM
    class CustomDataLoader(VisionDataset):
        def __getitem__(self, k):
            fake_images[k]

        def __len__(self):
            return len(fake_images)
    
def test_lazyram_dataloader_init():
    "We can init lazy ram using the dataloader"
    shape = (3,50,50)
    fake_images = torch.zeros(10, *shape)
    
    class CustomDataLoader(VisionDataset):
        def __getitem__(self, k):
            im = fake_images[k]
            return to_pil_image(im)

        def __len__(self):
            return len(fake_images)
 
    dl = CustomDataLoader('.')
    lazy_dl = LazyRAM(dl)
    image_tensor = fake_images[0]
    im = to_pil_image(image_tensor) 
    # Currently it is all zeros
    assert (image_tensor == torch.zeros(*shape)).all().item()
    # Check that lazy loader has this too

    first_item_lazy_dl = to_tensor(lazy_dl[0])
    assert (first_item_lazy_dl == torch.zeros(*shape)).all().item()
    fake_images[0] = torch.ones(*shape)
    # Make sure that this change has gone into the DataLoader
    assert (to_tensor(dl[0]) == torch.ones(*shape)).all().item()
    
    # The first item is still the same
    first_item_lazy_dl = to_tensor(lazy_dl[0])
    assert (first_item_lazy_dl == torch.zeros(*shape)).all().item()

