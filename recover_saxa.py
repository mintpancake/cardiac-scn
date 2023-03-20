import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.scn import SCN
import nrrd
import utils
from visualize import render_cross_section


# use SAXM SAXMV for normal vector
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True, help='SAXA')
    parser.add_argument('--dataset', type=str, default="test", help='test')

    parser.add_argument('--saxa_pth_path', type=str, required=True,
                        help='pths/SAXA/2023-01-01-00-00-00/100.pth')
    parser.add_argument('--saxm_pth_path', type=str, required=True,
                        help='pths/SAXM/2023-01-01-00-00-00/100.pth')
    parser.add_argument('--saxmv_pth_path', type=str, required=True,
                        help='pths/SAXMV/2023-01-01-00-00-00/100.pth')

    parser.add_argument('--saxa_model_key', type=str,
                        default='model_state_dict', help='saxa model key')
    parser.add_argument('--saxm_model_key', type=str,
                        default='model_state_dict', help='saxm model key')
    parser.add_argument('--saxmv_model_key', type=str,
                        default='model_state_dict', help='saxmv model key')
    args = parser.parse_args()
    view = args.view
    if view not in ['SAXA', 'SAXM', 'SAXMV']:
        raise ValueError('only SAXA, SAXM, SAXMV')
    dataset = args.dataset
    saxa_pth_path = args.saxa_pth_path
    saxm_pth_path = args.saxm_pth_path
    saxmv_pth_path = args.saxmv_pth_path
    saxa_model_key = args.saxa_model_key
    saxm_model_key = args.saxm_model_key
    saxmv_model_key = args.saxmv_model_key
    meta_dir = f'data/meta/{dataset}/{view}'
    saxa_meta_dir = f'data/meta/{dataset}/SAXA'
    saxm_meta_dir = f'data/meta/{dataset}/SAXM'
    saxmv_meta_dir = f'data/meta/{dataset}/SAXMV'
    ijk_dir = f'data/meta/3d_ijk/{view}'
    saxa_ijk_dir = f'data/meta/3d_ijk/SAXA'
    saxm_ijk_dir = f'data/meta/3d_ijk/SAXM'
    saxmv_ijk_dir = f'data/meta/3d_ijk/SAXMV'
    structs = utils.VIEW_STRUCTS[view]
    saxa_structs = utils.VIEW_STRUCTS['SAXA']
    saxm_structs = utils.VIEW_STRUCTS['SAXM']
    saxmv_structs = utils.VIEW_STRUCTS['SAXMV']
    save_dir = f'results/{view}'
    ratios = utils.read_ratio(f'data/meta/size/{view}.csv')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'fit.csv')
    error_path = os.path.join(save_dir, f'err.csv')
    image_path = os.path.join(save_dir, 'images')
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        header = ['name', 'p_centroid', 't_centroid', 'p_normal', 't_normal',
                  'normal_angle', 'centroid_dist', 'dist_along_t_normal',
                  'real_centroid_dist', 'real_dist_along_t_normal', 'ratio']
        writer.writerow(header)
    with open(error_path, 'w') as file:
        error_writer = csv.writer(file)
        header = ['name', 'struct', 'distance']
        error_writer.writerow(header)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True,
                       norm_truth=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=4)

    saxa_model = SCN(1, len(saxa_structs), filters=128,
                     factor=4, dropout=0.5).to(device)
    saxm_model = SCN(1, len(saxm_structs), filters=128,
                     factor=4, dropout=0.5).to(device)
    saxmv_model = SCN(1, len(saxmv_structs), filters=128,
                      factor=4, dropout=0.5).to(device)

    if saxa_model_key is None or saxa_model_key == '':
        saxa_model.load_state_dict(torch.load(
            saxa_pth_path, map_location=torch.device(device)))
    else:
        checkpoint = torch.load(
            saxa_pth_path, map_location=torch.device(device))
        saxa_model.load_state_dict(checkpoint[saxa_model_key])
    if saxm_model_key is None or saxm_model_key == '':
        saxm_model.load_state_dict(torch.load(
            saxm_pth_path, map_location=torch.device(device)))
    else:
        checkpoint = torch.load(
            saxm_pth_path, map_location=torch.device(device))
        saxm_model.load_state_dict(checkpoint[saxm_model_key])
    if saxmv_model_key is None or saxmv_model_key == '':
        saxmv_model.load_state_dict(torch.load(
            saxmv_pth_path, map_location=torch.device(device)))
    else:
        checkpoint = torch.load(
            saxmv_pth_path, map_location=torch.device(device))
        saxmv_model.load_state_dict(checkpoint[saxmv_model_key])
    saxa_model.eval()
    saxm_model.eval()
    saxmv_model.eval()

    normal_error, centroid_error, normal_centroid_error, real_centroid_error, real_normal_centroid_error = [], [], [], [], []
    size = len(loader)
    with torch.no_grad():
        for batch, (echo, truth, struct, filename) in enumerate(loader):
            echo = echo.to(device)
            saxa_pred = saxa_model(echo)[0][0]
            saxm_pred = saxm_model(echo)[0][0]
            saxmv_pred = saxmv_model(echo)[0][0]

            saxa_pred_xyz, saxm_pred_xyz, saxmv_pred_xyz = [], [], []
            for i, channel in enumerate(saxa_pred):
                a, b, c = channel.shape
                index = torch.argmax(channel).item()
                x, y, z = index//(b*c), (index % (b*c))//c, (index % (b*c)) % c
                x, y, z = float(x), float(y), float(z)
                saxa_pred_xyz.append([x, y, z])
            for i, channel in enumerate(saxm_pred):
                a, b, c = channel.shape
                index = torch.argmax(channel).item()
                x, y, z = index//(b*c), (index % (b*c))//c, (index % (b*c)) % c
                x, y, z = float(x), float(y), float(z)
                saxm_pred_xyz.append([x, y, z])
            for i, channel in enumerate(saxmv_pred):
                a, b, c = channel.shape
                index = torch.argmax(channel).item()
                x, y, z = index//(b*c), (index % (b*c))//c, (index % (b*c)) % c
                x, y, z = float(x), float(y), float(z)
                saxmv_pred_xyz.append([x, y, z])
            saxa_pred_xyz = np.array(saxa_pred_xyz)
            saxm_pred_xyz = np.array(saxm_pred_xyz)
            saxmv_pred_xyz = np.array(saxmv_pred_xyz)

            pred_xyz = None
            pred_centroid = None
            if view == 'SAXA':
                pred_xyz = saxa_pred_xyz
                pred_centroid = saxa_pred_xyz.mean(axis=0)
            elif view == 'SAXM':
                pred_xyz = saxm_pred_xyz
                pred_centroid = saxm_pred_xyz.mean(axis=0)
            elif view == 'SAXMV':
                pred_xyz = saxmv_pred_xyz
                pred_centroid = saxmv_pred_xyz.mean(axis=0)
            else:
                raise ValueError('Invalid view')

            shifted_saxa_pred_xyz = saxa_pred_xyz - saxa_pred_xyz.mean(axis=0)
            shifted_saxm_pred_xyz = saxm_pred_xyz - saxm_pred_xyz.mean(axis=0)
            shifted_saxmv_pred_xyz = saxmv_pred_xyz - \
                saxmv_pred_xyz.mean(axis=0)
            coplanar_pred_xyz = np.concatenate(
                (shifted_saxa_pred_xyz, shifted_saxm_pred_xyz, shifted_saxmv_pred_xyz), axis=0)
            _, pred_normal = utils.fit_plane(coplanar_pred_xyz)

            truth_nrrd_path = None
            truth_xyz = []
            file_path = os.path.join(ijk_dir, filename[0]+'.csv')
            reader = csv.reader(open(file_path, 'r'))
            for row in reader:
                if reader.line_num == 1:
                    continue
                elif reader.line_num == 2:
                    truth_nrrd_path = row[0]
                truth_xyz.append(
                    [float(row[2]), float(row[3]), float(row[4])])
            truth_xyz = np.array(truth_xyz)
            # groundtruth SAXA needs to be handled differently due to colinearity
            if view == 'SAXA':
                saxm_not_found = False
                saxmv_not_found = False
                saxm_truth_xyz = []
                saxmv_truth_xyz = []
                saxm_file_path = os.path.join(saxm_ijk_dir, filename[0]+'.csv')
                saxmv_file_path = os.path.join(
                    saxmv_ijk_dir, filename[0]+'.csv')
                try:
                    saxm_reader = csv.reader(open(saxm_file_path, 'r'))
                    for row in saxm_reader:
                        if saxm_reader.line_num == 1:
                            continue
                        saxm_truth_xyz.append(
                            [float(row[2]), float(row[3]), float(row[4])])
                    saxm_truth_xyz = np.array(saxm_truth_xyz)
                except FileNotFoundError:
                    saxm_not_found = True
                try:
                    saxmv_reader = csv.reader(open(saxmv_file_path, 'r'))
                    for row in saxmv_reader:
                        if saxmv_reader.line_num == 1:
                            continue
                        saxmv_truth_xyz.append(
                            [float(row[2]), float(row[3]), float(row[4])])
                    saxmv_truth_xyz = np.array(saxmv_truth_xyz)
                except FileNotFoundError:
                    saxmv_not_found = True
                if saxm_not_found and saxmv_not_found:
                    print('SAXM and SAXMV not found, use SAXA')
                    truth_centroid, truth_normal = utils.fit_plane(truth_xyz)
                elif saxm_not_found:
                    print('SAXM not found, use SAXMV')
                    truth_centroid = truth_xyz.mean(axis=0)
                    shifted_truth_xyz = truth_xyz - \
                        truth_xyz.mean(axis=0)
                    shifted_saxmv_truth_xyz = saxmv_truth_xyz - \
                        saxmv_truth_xyz.mean(axis=0)
                    coplanar_truth_xyz = np.concatenate(
                        (shifted_truth_xyz, shifted_saxmv_truth_xyz), axis=0)
                    _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
                elif saxmv_not_found:
                    print('SAXMV not found, use SAXM')
                    truth_centroid = truth_xyz.mean(axis=0)
                    shifted_truth_xyz = truth_xyz - \
                        truth_xyz.mean(axis=0)
                    shifted_saxm_truth_xyz = saxm_truth_xyz - \
                        saxm_truth_xyz.mean(axis=0)
                    coplanar_truth_xyz = np.concatenate(
                        (shifted_truth_xyz, shifted_saxm_truth_xyz), axis=0)
                    _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
                else:
                    truth_centroid = truth_xyz.mean(axis=0)
                    shifted_truth_xyz = truth_xyz - \
                        truth_xyz.mean(axis=0)
                    shifted_saxm_truth_xyz = saxm_truth_xyz - \
                        saxm_truth_xyz.mean(axis=0)
                    shifted_saxmv_truth_xyz = saxmv_truth_xyz - \
                        saxmv_truth_xyz.mean(axis=0)
                    coplanar_truth_xyz = np.concatenate(
                        (shifted_truth_xyz, shifted_saxm_truth_xyz, shifted_saxmv_truth_xyz), axis=0)
                    _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
            else:
                truth_centroid, truth_normal = utils.fit_plane(truth_xyz)

            centroid_distance = np.sqrt(
                np.sum((pred_centroid-truth_centroid)**2))
            normal_angle = utils.angle_between(pred_normal, truth_normal)
            ratio = ratios[filename[0]]
            normal_centroid_distance = utils.distance_along_direction(
                truth_centroid, pred_centroid, truth_normal)
            real_centroid_distance = centroid_distance/ratio
            real_normal_centroid_distance = normal_centroid_distance/ratio

            with open(save_path, 'a+') as file:
                writer = csv.writer(file)
                data_row = [filename[0], pred_centroid, truth_centroid,
                            pred_normal, truth_normal, normal_angle,
                            centroid_distance, normal_centroid_distance,
                            real_centroid_distance, real_normal_centroid_distance, ratio]
                writer.writerow(data_row)

            error_distances = utils.distance_to_plane(
                pred_xyz, pred_centroid, pred_normal)**2
            for i, struct in enumerate(structs):
                with open(error_path, 'a+') as file:
                    error_writer = csv.writer(file)
                    data_row = [filename[0], struct, error_distances[i]]
                    error_writer.writerow(data_row)

            normal_error.append(normal_angle)
            centroid_error.append(centroid_distance)
            normal_centroid_error.append(normal_centroid_distance)
            real_centroid_error.append(real_centroid_distance)
            real_normal_centroid_error.append(real_normal_centroid_distance)

            nrrd_data = nrrd.read(truth_nrrd_path)[0].astype(np.float64)
            pred_image = render_cross_section(
                nrrd_data, pred_centroid, pred_normal)
            truth_image = render_cross_section(
                nrrd_data, truth_centroid, truth_normal)
            utils.draw(pred_image, os.path.join(
                image_path, f'{filename[0]}_pred.png'))
            utils.draw(truth_image, os.path.join(
                image_path, f'{filename[0]}_truth.png'))

            print(f'[{batch:>3d}/{size:>3d}] {centroid_distance} {normal_angle}')
    centroid_error = np.array(centroid_error)
    normal_error = np.array(normal_error)
    normal_centroid_error = np.array(normal_centroid_error)
    real_centroid_error = np.array(real_centroid_error)
    real_normal_centroid_error = np.array(real_normal_centroid_error)
    with open(save_path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(['[median]', '', '', '', '',
                         np.median(normal_error), np.median(centroid_error),
                         np.median(normal_centroid_error), np.median(
                             real_centroid_error),
                         np.median(real_normal_centroid_error), ''])
        writer.writerow(['[mean]', '', '', '', '',
                         normal_error.mean(), centroid_error.mean(),
                         normal_centroid_error.mean(), real_centroid_error.mean(),
                         real_normal_centroid_error.mean(), ''])
        writer.writerow(['[std]', '', '', '', '',
                         normal_error.std(), centroid_error.std(),
                         normal_centroid_error.std(), real_centroid_error.std(),
                         real_normal_centroid_error.std(), ''])
        
    print(f'[median] {np.median(centroid_error)} {np.median(normal_error)}')
    print(f'[mean] {centroid_error.mean()} {normal_error.mean()}')
    print(f'[std] {centroid_error.std()} {normal_error.std()}')
