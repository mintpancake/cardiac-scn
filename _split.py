import os
import argparse
import numpy as np

CLEANUP_EXIST = True

all_dir = 'data/meta/3d_truth'
test_meta_path = 'data/meta/test/_TEST.txt'
train_val_dir = 'data/meta/train_val'
train_dir = 'data/meta/train'
val_dir = 'data/meta/val'
test_dir = 'data/meta/test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', type=str,
                        help='A2C,A4C,SAXA,SAXB,ALAX,SAXMV,SAXM')
    args = parser.parse_args()
    views = [v.strip() for v in args.views.split(',')]

    # Read predetermined test data
    test_filenames = set()
    with open(test_meta_path) as file:
        for line in file:
            test_filenames.add(line.strip('\n'))

    for view in views:
        all_view_dir = os.path.join(all_dir, view)
        train_val_view_dir = os.path.join(train_val_dir, view)
        train_view_dir = os.path.join(train_dir, view)
        val_view_dir = os.path.join(val_dir, view)
        test_view_dir = os.path.join(test_dir, view)
        os.makedirs(train_view_dir, exist_ok=True)
        os.makedirs(val_view_dir, exist_ok=True)
        os.makedirs(test_view_dir, exist_ok=True)
        if CLEANUP_EXIST:
            os.system(f'rm -rf {train_view_dir}/*')
            os.system(f'rm -rf {val_view_dir}/*')
            os.system(f'rm -rf {test_view_dir}/*')

        # train : val : test = 64 : 16 : 20
        all_meta = sorted([i for i in os.listdir(all_view_dir) if '.csv' in i])
        test_meta = [i for i in all_meta if os.path.splitext(i)[
            0] in test_filenames]
        train_val_meta = [i for i in all_meta if os.path.splitext(i)[
            0] not in test_filenames]
        np.random.seed(4801)
        shuffled_meta = np.random.permutation(train_val_meta)
        train_meta = shuffled_meta[:8 * len(shuffled_meta) // 10]
        val_meta = shuffled_meta[8 * len(shuffled_meta) // 10:]

        # Record filenames
        with open(os.path.join(train_view_dir, f'_{view}_TRAIN.txt'), 'w') as file:
            for m in train_meta:
                file.write(os.path.splitext(m)[0]+'\n')
        with open(os.path.join(val_view_dir, f'_{view}_VAL.txt'), 'w') as file:
            for m in val_meta:
                file.write(os.path.splitext(m)[0]+'\n')
        with open(os.path.join(test_view_dir, f'_{view}_TEST.txt'), 'w') as file:
            for m in test_meta:
                file.write(os.path.splitext(m)[0]+'\n')

        # Copy meta
        for m in train_meta:
            os.system(f'cp {os.path.join(all_view_dir, m)} {train_view_dir}')
            print(f'{view} Train {m}')
        for m in val_meta:
            os.system(f'cp {os.path.join(all_view_dir, m)} {val_view_dir}')
            print(f'{view} Val {m}')
        for m in test_meta:
            os.system(f'cp {os.path.join(all_view_dir, m)} {test_view_dir}')
            print(f'{view} Test {m}')

        os.system(f'cp -r {train_view_dir}/. {train_val_view_dir}')
        os.system(f'cp -r {val_view_dir}/. {train_val_view_dir}')
