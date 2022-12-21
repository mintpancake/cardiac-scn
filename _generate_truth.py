import os
import nrrd
import csv
import math
import numpy as np
from scipy.stats import multivariate_normal
import utils

NRRD_OUT_SIZE = [300, 300, 300]
SKIP_SAVED_NRRD = True

csv_dirs = ['data/meta/3d_ijk/A2C',
            'data/meta/3d_ijk/A4C',
            'data/meta/3d_ijk/ALAX',
            'data/meta/3d_ijk/SAXA',
            'data/meta/3d_ijk/SAXB',
            'data/meta/3d_ijk/SAXM',
            'data/meta/3d_ijk/SAXMV']

truth_save_dir = 'data/truth'
meta_save_dir = 'data/meta/3d_truth'

GAUSS_RANGE = {}


def gaussian_excitation(mean: list,
                        sigma: float = 2.0,
                        size: list = NRRD_OUT_SIZE,
                        eps: float = 1e-5) -> np.ndarray:
    mean = np.array(mean)
    assert sigma > 0.0
    cov = np.diag(np.repeat(sigma**2, 3))
    max_val = multivariate_normal.pdf([[0, 0, 0]], [0, 0, 0], cov)

    # Get hotspot range
    if GAUSS_RANGE.get(sigma):
        g_range = GAUSS_RANGE[sigma]
    else:
        dist = sigma
        for _ in range(20):
            if multivariate_normal.pdf([[dist, 0, 0]], [0, 0, 0], cov)/max_val < eps:
                GAUSS_RANGE[sigma] = math.floor(dist)
                break
            else:
                dist += sigma
        g_range = math.floor(dist)

    center = np.round(mean).astype(int)
    x_range = np.array(
        [max(center[0]-g_range, 0), min(center[0]+g_range+1, size[0])])
    y_range = np.array(
        [max(center[1]-g_range, 0), min(center[1]+g_range+1, size[1])])
    z_range = np.array(
        [max(center[2]-g_range, 0), min(center[2]+g_range+1, size[2])])

    # Compute pdf in hotspot range
    x, y, z = np.mgrid[x_range[0]:x_range[1],
                       y_range[0]:y_range[1],
                       z_range[0]:z_range[1]]
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    patch = multivariate_normal.pdf(xyz, mean, cov).reshape(x.shape)/max_val

    # Put hotspot on heatmap
    heatmap = np.zeros(size)
    heatmap[x_range[0]:x_range[1], y_range[0]
        :y_range[1], z_range[0]:z_range[1]] = patch

    return heatmap


if __name__ == '__main__':
    all_dirs = len(csv_dirs)
    for (processed_dirs, csv_dir) in enumerate(csv_dirs):
        csv_filenames = os.listdir(csv_dir)
        csv_filenames = sorted(csv_filenames)
        view_name = os.path.basename(csv_dir)
        utils.ensure_dir(os.path.join(truth_save_dir, view_name))
        utils.ensure_dir(os.path.join(meta_save_dir, view_name))
        all_files = len(csv_filenames)
        for (processed_files, csv_filename) in enumerate(csv_filenames):
            print(
                f'[{processed_dirs+1}/{all_dirs}] [{processed_files+1}/{all_files}] {csv_dir}/{csv_filename}')
            filename_wo_ext = os.path.splitext(csv_filename)[0]
            truth_save_path = os.path.join(
                truth_save_dir, view_name, f'{filename_wo_ext}.nrrd')
            if SKIP_SAVED_NRRD and os.path.exists(truth_save_path):
                continue
            meta_save_path = os.path.join(
                meta_save_dir, view_name, f'{filename_wo_ext}.csv')
            csv_path = os.path.join(csv_dir, csv_filename)
            csv_reader = csv.reader(open(csv_path, 'r'))
            csv_dict = {}
            nrrd_path = ''
            for row in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                nrrd_path = row[0]
                csv_dict[int(row[1])] = [float(row[2]),
                                         float(row[3]),
                                         float(row[4])]

            # Generate excitation
            heatmaps = []
            for struct_idx in utils.VIEW_STRUCTS[view_name]:
                if not csv_dict.get(struct_idx):
                    heatmap = np.zeros(NRRD_OUT_SIZE)
                else:
                    pos = csv_dict.get(struct_idx)
                    heatmap = gaussian_excitation(pos)
                heatmaps.append(heatmap)
            heatmaps = np.array(heatmaps)
            nrrd.write(truth_save_path, heatmaps)

            # Write meta
            for idx, struct_idx in enumerate(utils.VIEW_STRUCTS[view_name]):
                if not csv_dict.get(struct_idx):
                    truth_path = ''
                else:
                    truth_path = truth_save_path
                if idx == 0:
                    with open(meta_save_path, 'w') as meta_file:
                        csv_write = csv.writer(meta_file)
                        csv_head = ['index', 'struct', 'echo', 'truth']
                        csv_write.writerow(csv_head)
                        data_row = [idx, struct_idx, nrrd_path, truth_path]
                        csv_write.writerow(data_row)
                else:
                    with open(meta_save_path, 'a+') as meta_file:
                        csv_write = csv.writer(meta_file)
                        data_row = [idx, struct_idx, nrrd_path, truth_path]
                        csv_write.writerow(data_row)
