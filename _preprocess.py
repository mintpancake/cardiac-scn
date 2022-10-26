import os
import nrrd
import csv
from scipy import ndimage
import utils

csv_dirs = ['data/meta/4d_ijk/A2C',
            'data/meta/4d_ijk/A4C',
            'data/meta/4d_ijk/ALAX',
            'data/meta/4d_ijk/SAXA',
            'data/meta/4d_ijk/SAXB',
            'data/meta/4d_ijk/SAXM',
            'data/meta/4d_ijk/SAXMV']

nrrd_save_dir = 'data/nrrd'
meta_save_dir = 'data/meta/3d_ijk'

if __name__ == '__main__':
    for csv_dir in csv_dirs:
        csv_filenames = os.listdir(csv_dir)
        csv_filenames = sorted(csv_filenames)
        view_name = os.path.basename(csv_dir)
        utils.ensure_dir(os.path.join(nrrd_save_dir, view_name))
        utils.ensure_dir(os.path.join(meta_save_dir, view_name))
        for csv_filename in csv_filenames:
            print(f'{csv_dir}/{csv_filename}')
            filename_wo_ext = os.path.splitext(csv_filename)[0]
            nrrd_save_path = os.path.join(
                nrrd_save_dir, view_name, f'{filename_wo_ext}.nrrd')
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

            if not os.path.exists(nrrd_save_path):
                data_3d = data_4d[time_idx]
                data_3d_scaled = ndimage.zoom(data_3d, space_scales)
                nrrd.write(nrrd_save_path, data_3d_scaled)

            for idx, row in enumerate(csv_mat):
                struct, i, j, k = int(row[2]), float(
                    row[3]), float(row[4]), float(row[5])
                i *= space_scales[0]
                j *= space_scales[1]
                k *= space_scales[2]
                if idx == 0:
                    with open(meta_save_path, 'w') as meta_file:
                        csv_write = csv.writer(meta_file)
                        csv_head = ['nrrd', 'struct', 'i', 'j', 'k']
                        csv_write.writerow(csv_head)
                        data_row = [nrrd_save_path, struct, i, j, k]
                        csv_write.writerow(data_row)
                else:
                    with open(meta_save_path, 'a+') as meta_file:
                        csv_write = csv.writer(meta_file)
                        data_row = [nrrd_save_path, struct, i, j, k]
                        csv_write.writerow(data_row)
