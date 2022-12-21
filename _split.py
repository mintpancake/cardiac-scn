import os
import numpy as np

meta_dir = 'data/meta/3d_truth'
train_dir = 'data/meta/train'
val_dir = 'data/meta/val'

if __name__ == '__main__':
    structs = os.listdir(meta_dir)
