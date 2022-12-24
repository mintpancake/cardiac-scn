import os
import csv
import numpy as np

CLEANUP_EXIST = True

all_dir = 'data/meta/3d_truth'
test_meta_path = 'data/meta/test/TEST.csv'
views = ['A2C', 'A4C', 'SAXA', 'SAXB', 'ALAX', 'SAXMV', 'SAXM']
train_dir = 'data/meta/train'
val_dir = 'data/meta/val'
test_dir = 'data/meta/test'

if __name__ == '__main__':
    # Read predetermined test data
    test_meta_reader = csv.reader(open(test_meta_path, 'r'))
    test_filenames = set()
    for row in test_meta_reader:
        test_filenames.add(row[0])

    for view in views:
        all_view_dir = os.path.join(all_dir, view)
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
        with open(os.path.join(train_view_dir, f'_{view}_TRAIN.csv'), 'w') as file:
            csv_writer = csv.writer(file)
            for m in train_meta:
                csv_writer.writerow([os.path.splitext(m)[0]])
        with open(os.path.join(val_view_dir, f'_{view}_VAL.csv'), 'w') as file:
            csv_writer = csv.writer(file)
            for m in val_meta:
                csv_writer.writerow([os.path.splitext(m)[0]])
        with open(os.path.join(test_view_dir, f'_{view}_TEST.csv'), 'w') as file:
            csv_writer = csv.writer(file)
            for m in test_meta:
                csv_writer.writerow([os.path.splitext(m)[0]])

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
