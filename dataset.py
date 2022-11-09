import os
import torch
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir) -> None:
        super().__init__()
        self.size = len(os.listdir(meta_dir))

    def __len__(self):
        pass

    def __getitem__(self, index):
        return self.size
