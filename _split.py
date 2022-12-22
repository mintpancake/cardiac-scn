import os
import numpy as np

CLEANUP_EXIST = False

all_dir = 'data/meta/3d_truth'
views = ['A2C', 'A4C', 'SAXA', 'SAXB', 'ALAX', 'SAXMV', 'SAXM']
train_dir = 'data/meta/train'
val_dir = 'data/meta/val'

if __name__ == '__main__':
    for view in views:
        all_view_dir = os.path.join(all_dir, view)
        train_view_dir = os.path.join(train_dir, view)
        val_view_dir = os.path.join(val_dir, view)
        os.makedirs(train_view_dir, exist_ok=True)
        os.makedirs(val_view_dir, exist_ok=True)
        if CLEANUP_EXIST:
            os.system(f'rm -rf {train_view_dir}')
            os.system(f'rm -rf {val_view_dir}')

        # train : val = 8 : 2
        all_meta = sorted([i for i in os.listdir(all_view_dir) if '.csv' in i])
        np.random.seed(4801)
        shuffled_meta = np.random.permutation(all_meta)
        train_meta = shuffled_meta[:8 * len(shuffled_meta) // 10]
        val_meta = shuffled_meta[8 * len(shuffled_meta) // 10:]

        for m in train_meta:
            os.system(f'cp {os.path.join(all_view_dir, m)} {train_view_dir}')
            print(f'{view} Train {m}')
        for m in val_meta:
            os.system(f'cp {os.path.join(all_view_dir, m)} {val_view_dir}')
            print(f'{view} Val {m}')
