import os
import csv
import nrrd
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir, norm_echo=True, norm_truth=True, augmentation=True) -> None:
        super().__init__()
        self.meta_dir = meta_dir
        self.norm_echo = norm_echo
        self.norm_truth = norm_truth
        self.augmentation = augmentation
        self.csv_names = [i for i in sorted(os.listdir(meta_dir)) if os.path.splitext(i)[
            1] == '.csv' or os.path.splitext(i)[1] == '.CSV']
        self.size = len(self.csv_names)
        self.metas = self.load(self.meta_dir, self.csv_names)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        meta = self.metas[index]
        echo_data = nrrd.read(meta[0][2])[0]
        truth_data = nrrd.read(meta[0][3])[0]

        if self.augmentation:
            zoom = np.random.uniform(0.9, 1.1)
            shift = np.random.randint(-5, 5, size=3)
            angles = (np.random.uniform(-15, 15),
                      np.random.uniform(-5, 5),
                      np.random.uniform(-5, 5))
            echo_data, truth_data = self.augment(
                echo_data, truth_data, zoom, shift, angles)

        if self.norm_echo:
            echo_data = self.normalize_echo(echo_data)

        if self.norm_truth:
            truth_data = self.normalize_truth(truth_data)

        if self.augmentation:
            multiply = np.random.uniform(0.75, 1.25)
            add = np.random.uniform(-0.25, 0.25)
            echo_data = self.adjust_intensity(echo_data, multiply, add)

        echo_data = torch.from_numpy(echo_data).float()
        echo_data = torch.unsqueeze(echo_data, dim=0)
        truth_data = torch.from_numpy(truth_data).float()
        structs = torch.IntTensor(sorted([row[1] for row in meta]))

        return (echo_data, truth_data, structs)

    def augment(self, echo_data, truth_data, zoom, shift, angles):
        size = echo_data.shape

        if zoom < 1.0:
            echo_data = self.scale(echo_data, zoom, size)
        echo_data = self.rotate_and_translate(echo_data, angles, shift, size)
        if zoom > 1.0:
            echo_data = self.scale(echo_data, zoom, size)

        transformed_truth_data = np.zeros_like(truth_data)
        for channel, channel_data in enumerate(truth_data):
            if zoom < 1.0:
                channel_data = self.scale(channel_data, zoom, size)
            channel_data = self.rotate_and_translate(
                channel_data, angles, shift, size)
            if zoom > 1.0:
                channel_data = self.scale(channel_data, zoom, size)
            transformed_truth_data[channel] = channel_data

        return echo_data, transformed_truth_data

    def fit_size(self, data, size):
        data_size = np.array(data.shape)
        size = np.array(size)
        if data_size[0] > size[0]:
            start = (data_size-size)//2
            end = start+size
            fitted_data = data[start[0]:end[0],
                               start[1]:end[1],
                               start[2]:end[2]]
        else:
            start = (size-data_size)//2
            end = start+data_size
            fitted_data = np.zeros(size)
            fitted_data[start[0]:end[0], start[1]
                :end[1], start[2]:end[2]] = data
        return fitted_data

    def rotate_and_translate(self, data, angles, shift, size):
        data = self.rotate(data, angles)
        data = self.translate(data, shift)
        return self.fit_size(data, size)

    def translate(self, data, shift):
        return ndimage.shift(data, shift, cval=0.0)

    def rotate(self, data, angles):
        data = ndimage.rotate(
            data, angles[0], axes=(1, 0), reshape=True, cval=0.0)
        data = ndimage.rotate(
            data, angles[1], axes=(2, 1), reshape=True, cval=0.0)
        data = ndimage.rotate(
            data, angles[2], axes=(0, 2), reshape=True, cval=0.0)
        return data

    def scale(self, data, zoom, size):
        data = ndimage.zoom(data, zoom, cval=0.0)
        return self.fit_size(data, size)

    def adjust_intensity(self, data, multiply, add):
        data *= multiply
        data += add
        return data

    def normalize_echo(self, data):
        max_echo = data.max()
        min_echo = data.min()
        mean_echo = (max_echo+min_echo)/2.0
        scale_echo = (max_echo-min_echo)/2.0
        return (data-mean_echo)/scale_echo

    def normalize_truth(self, data):
        max_truth = data.max()
        return data/max_truth

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
