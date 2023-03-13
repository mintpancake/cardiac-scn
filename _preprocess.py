import os
import nrrd
import csv
import argparse
import numpy as np
from scipy import ndimage
import utils

RESIZE_OUT_SIZE = [128, 128, 128]
SKIP_SAVED_NRRD = False


nrrd_save_dir = 'data/nrrd'
meta_save_dir = 'data/meta/3d_ijk'


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
        utils.ensure_dir(os.path.join(nrrd_save_dir, view_name))
        utils.ensure_dir(os.path.join(meta_save_dir, view_name))
        all_files = len(csv_filenames)
        for (processed_files, csv_filename) in enumerate(csv_filenames):
            print(
                f'[{processed_dirs+1}/{all_dirs}] [{processed_files+1}/{all_files}] {csv_dir}/{csv_filename}', end=' ')
            filename_wo_ext = os.path.splitext(csv_filename)[0]
            nrrd_save_path = os.path.join(
                nrrd_save_dir, view_name, f'{filename_wo_ext}.nrrd')
            if SKIP_SAVED_NRRD and os.path.exists(nrrd_save_path):
                continue
            meta_save_path = os.path.join(
                meta_save_dir, view_name, f'{filename_wo_ext}.csv')
            csv_path = os.path.join(csv_dir, csv_filename)
            csv_reader = csv.reader(open(csv_path, 'r'))
            csv_mat = []
            for row in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                csv_mat.append(row)

            nrrd_path = csv_mat[0][0]
            time_idx = int(csv_mat[0][1])
            data_4d, header = nrrd.read(nrrd_path)
            space_scales = (header['space directions'][1][0],
                            header['space directions'][2][1],
                            header['space directions'][3][2])

            # Scaling and padding
            data_3d = data_4d[time_idx]
            print(data_3d.shape, end=' ')
            data_3d_scaled = ndimage.zoom(data_3d, space_scales)
            print(data_3d_scaled.shape, end=' ')
            data_3d_padded, offsets, max_length = pad(data_3d_scaled)
            print(data_3d_padded.shape, end=' ')
            resize_scale = RESIZE_OUT_SIZE[0]/max_length
            data_3d_resized = ndimage.zoom(data_3d_padded, np.array(
                RESIZE_OUT_SIZE)/np.array([max_length, max_length, max_length]))
            print(data_3d_resized.shape)
            nrrd.write(nrrd_save_path, data_3d_resized)

            # Write meta
            for idx, row in enumerate(csv_mat):
                struct, i, j, k = int(row[2]), float(
                    row[3]), float(row[4]), float(row[5])
                i, j, k = i*space_scales[0], j * \
                    space_scales[1], k*space_scales[2]
                i, j, k = i+offsets[0], j+offsets[1], k+offsets[2]
                i, j, k = i*resize_scale, j*resize_scale, k*resize_scale
                if idx == 0:
                    with open(meta_save_path, 'w') as meta_file:
                        csv_writer = csv.writer(meta_file)
                        csv_head = ['nrrd', 'struct', 'i', 'j', 'k']
                        csv_writer.writerow(csv_head)
                        data_row = [nrrd_save_path, struct, i, j, k]
                        csv_writer.writerow(data_row)
                else:
                    with open(meta_save_path, 'a+') as meta_file:
                        csv_writer = csv.writer(meta_file)
                        data_row = [nrrd_save_path, struct, i, j, k]
                        csv_writer.writerow(data_row)
