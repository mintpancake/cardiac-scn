import os
import nrrd
import csv
import argparse
import numpy as np
from scipy import ndimage
import utils

RESIZE_OUT_SIZE = [128, 128, 128]


save_dir = 'data/meta/size'


def pad(data: np.ndarray):
    max_length = max(data.shape)
    out_size = [max_length, max_length, max_length]
    pad_width = np.array([[0, 0], [0, 0], [0, 0]])
    offsets = np.array([0, 0, 0])
    for d in range(3):
        if data.shape[d] > out_size[d]:
            start = (data.shape[d]-out_size[d])//2
            end = start+out_size[d]
            data = data.take(indices=range(start, end), axis=d)
            offsets[d] = -start
        else:
            before = (out_size[d]-data.shape[d])//2
            after = out_size[d]-data.shape[d]-before
            pad_width[d] = [before, after]
            offsets[d] = before
    return np.pad(data, pad_width, 'constant'), offsets, max_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', type=str, required=True,
                        help='A2C,A4C,SAXA,SAXB,ALAX,SAXMV,SAXM')
    args = parser.parse_args()
    csv_dirs = [f'data/meta/4d_ijk/{v.strip()}' for v in args.views.split(',')]

    all_dirs = len(csv_dirs)
    for (processed_dirs, csv_dir) in enumerate(csv_dirs):
        csv_filenames = os.listdir(csv_dir)
        csv_filenames = sorted(csv_filenames)
        view_name = os.path.basename(csv_dir)
        save_path = os.path.join(save_dir, f'{view_name}.csv')
        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'in_size', 'out_size', 'ratio'])

        all_files = len(csv_filenames)
        ratios = []
        for (processed_files, csv_filename) in enumerate(csv_filenames):
            print(
                f'[{processed_dirs+1}/{all_dirs}] [{processed_files+1}/{all_files}] {csv_dir}/{csv_filename}', end=' ')
            filename_wo_ext = os.path.splitext(csv_filename)[0]

            csv_path = os.path.join(csv_dir, csv_filename)
            csv_reader = csv.reader(open(csv_path, 'r'))
            csv_mat = []
            for row in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                csv_mat.append(row)

            nrrd_path = csv_mat[0][0]
            header = nrrd.read_header(nrrd_path)
            original_shape = header['sizes'][1:4]
            space_scales = (header['space directions'][1][0],
                            header['space directions'][2][1],
                            header['space directions'][3][2])
            scaled_shape = np.around(
                np.array(original_shape)*np.array(space_scales)).astype(np.int32)
            in_size = np.max(scaled_shape)
            out_size = RESIZE_OUT_SIZE[0]
            ratio = out_size/in_size
            ratios.append(ratio)
            print(f'{original_shape} {in_size} {out_size} {ratio}')

            # Write meta
            with open(save_path, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([filename_wo_ext, in_size, out_size, ratio])
        ratios = np.array(ratios)
        with open(save_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(['[mean]', '', '', ratios.mean()])
