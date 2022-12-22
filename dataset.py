import os
import csv
import nrrd
import torch
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir) -> None:
        super().__init__()
        self.meta_dir = meta_dir
        self.csv_names = os.listdir(meta_dir)
        self.size = len(self.csv_names)
        self.metas = self.load(self.meta_dir, self.csv_names)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        meta = self.metas[index]
        echo_data = torch.from_numpy(nrrd.read(meta[0][2])[0]).float()
        echo_data = torch.unsqueeze(echo_data, dim=0)
        truth_data = torch.from_numpy(nrrd.read(meta[0][3])[0]).float()
        structs = [row[1] for row in meta]
        return (echo_data, truth_data, structs)

    def load(self, meta_dir, csv_names):
        metas = []
        for csv_name in csv_names:
            csv_reader = csv.reader(
                open(os.path.join(meta_dir, csv_name), 'r'))
            meta = []
            for row in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                meta.append([int(row[0]), int(row[1]), row[2], row[3]])
            metas.append(meta)
        return metas
