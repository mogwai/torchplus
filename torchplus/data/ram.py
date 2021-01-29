from torch.dataset import Dataset
from typing import Dict

class RamCache(Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved into Ram
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.location = location
        self._cache: Dict = {}

    def __getitem__(self, n: int) -> Any:
        if self._cache[n]:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item
        return item

    def __len__(self) -> int:
        return len(self.dataset)
