import os
import csv
import nrrd
import torch
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir, norm_echo=True, norm_truth=True) -> None:
        super().__init__()
        self.meta_dir = meta_dir
        self.norm_echo = norm_echo
        self.norm_truth = norm_truth
        self.csv_names = [i for i in sorted(os.listdir(meta_dir)) if os.path.splitext(i)[
            1] == '.csv' or os.path.splitext(i)[1] == '.CSV']
        self.size = len(self.csv_names)
        self.metas = self.load(self.meta_dir, self.csv_names)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        meta = self.metas[index]
        echo_data = torch.from_numpy(nrrd.read(meta[0][2])[0]).float()
        echo_data = torch.unsqueeze(echo_data, dim=0)
        # echo data [-1:1]
        if self.norm_echo:
            max_echo = torch.max(echo_data)
            min_echo = torch.min(echo_data)
            mean_echo = (max_echo+min_echo)/2.0
            scale_echo = (max_echo-min_echo)/2.0
            echo_data = (echo_data-mean_echo)/scale_echo
        truth_data = torch.from_numpy(nrrd.read(meta[0][3])[0]).float()
        # truth data [0:1]
        if self.norm_truth:
            max_truth = torch.max(truth_data)
            truth_data /= max_truth
        structs = torch.IntTensor(sorted([row[1] for row in meta]))
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
