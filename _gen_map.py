import csv
import json
import os
import utils

json_dir = "data/3d-data/json/2021-09-17-annotations"
nrrd_dir = "data/3d-data/nrrd/2021-09-17-3d-nrrd"
meta_dir = "data/meta/map"

if __name__ == '__main__':
    json_filenames = os.listdir(json_dir)
    json_filenames = sorted(json_filenames)
    for json_filename in json_filenames:
        print(json_filename)
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path) as json_file:
            anno = json.load(json_file)
        nrrd_filename = os.path.basename(anno['rawDataFilePath'])
        nrrd_path = os.path.join(nrrd_dir, nrrd_filename)
        for fid_datum in anno['fidData']:
            time_idx = fid_datum['Time Index']
            i, j, k = fid_datum['Position-IJK']
            struct_name = fid_datum['Structure Name']
            struct_idx = utils.get_struct_idx(struct_name)
            view_name = fid_datum['View Name']
            view_idx = utils.get_view_index(view_name, 'name')
            view_abbr = utils.get_view_abbr[view_idx]
            view_filename = f'{view_abbr}.csv'
            view_path = os.path.join(meta_dir, view_filename)
            if not os.path.exists(view_path):
                with open(view_path, 'w') as view_file:
                    csv_write = csv.writer(view_file)
                    csv_head = ['nrrd', 'time', 'struct', 'i', 'j', 'k']
                    csv_write.writerow(csv_head)
            with open(view_path, 'a+') as view_file:
                csv_write = csv.writer(view_file)
                data_row = [nrrd_path, time_idx, struct_idx, i, j, k]
                csv_write.writerow(data_row)
