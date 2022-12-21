import csv
import json
import os
import utils

json_dirs = ['data/raw/json/2021-09-17-annotations', 'data/raw/json/2021-09-20-annotations', 'data/raw/json/2021-09-23-annotations',
             'data/raw/json/2021-09-27-annotations', 'data/raw/json/2021-09-30-annotations', 'data/raw/json/2021-10-04-annotations',
             'data/raw/json/2021-10-05-annotations', 'data/raw/json/2021-10-06-annotations', 'data/raw/json/2021-10-11-annotations',
             'data/raw/json/2021-10-12-annotations', 'data/raw/json/2021-10-21-annotations', 'data/raw/json/2021-10-22-annotations',
             'data/raw/json/2021-10-25-annotations', 'data/raw/json/2021-10-27-annotations', 'data/raw/json/2021-10-28-annotations']
nrrd_dirs = ['data/raw/nrrd/2021-09-17-3d-nrrd', 'data/raw/nrrd/2021-09-20-3d-nrrd', 'data/raw/nrrd/2021-09-23-3d-nrrd',
             'data/raw/nrrd/2021-09-27-3d-nrrd', 'data/raw/nrrd/2021-09-30-3d-nrrd', 'data/raw/nrrd/2021-10-04-3d-nrrd',
             'data/raw/nrrd/2021-10-05-3d-nrrd', 'data/raw/nrrd/2021-10-06-3d-nrrd', 'data/raw/nrrd/2021-10-11-3d-nrrd',
             'data/raw/nrrd/2021-10-12-3d-nrrd', 'data/raw/nrrd/2021-10-21-3d-nrrd', 'data/raw/nrrd/2021-10-22-3d-nrrd',
             'data/raw/nrrd/2021-10-25-3d-nrrd', 'data/raw/nrrd/2021-10-27-3d-nrrd', 'data/raw/nrrd/2021-10-28-3d-nrrd']
meta_dir = "data/meta/4d_ijk"

if __name__ == '__main__':
    for idx in range(len(json_dirs)):
        json_dir = json_dirs[idx]
        nrrd_dir = nrrd_dirs[idx]
        json_filenames = os.listdir(json_dir)
        json_filenames = sorted(json_filenames)
        for json_filename in json_filenames:
            print(json_filename)
            json_path = os.path.join(json_dir, json_filename)
            with open(json_path) as json_file:
                anno = json.load(json_file)
            nrrd_filename = os.path.basename(anno['rawDataFilePath'])
            nrrd_path = os.path.join(nrrd_dir, nrrd_filename)
            filename_wo_ext = os.path.splitext(json_filename)[0]
            for fid_datum in anno['fidData']:
                time_idx = fid_datum['Time Index']
                i, j, k = fid_datum['Position-IJK']
                struct_name = fid_datum['Structure Name']
                struct_idx = utils.get_struct_idx(struct_name)

                view_name = fid_datum['View Name']
                view_idx = utils.get_view_index(view_name, 'name')
                view_abbr = utils.get_view_abbr(view_idx)
                view_dir = os.path.join(meta_dir, view_abbr)
                utils.ensure_dir(view_dir)

                view_filename = f'{filename_wo_ext}.csv'
                view_path = os.path.join(view_dir, view_filename)
                if not os.path.exists(view_path):
                    with open(view_path, 'w') as view_file:
                        csv_write = csv.writer(view_file)
                        csv_head = ['nrrd', 'time', 'struct', 'i', 'j', 'k']
                        csv_write.writerow(csv_head)
                        data_row = [nrrd_path, time_idx, struct_idx, i, j, k]
                        csv_write.writerow(data_row)
                else:
                    with open(view_path, 'a+') as view_file:
                        csv_write = csv.writer(view_file)
                        data_row = [nrrd_path, time_idx, struct_idx, i, j, k]
                        csv_write.writerow(data_row)
