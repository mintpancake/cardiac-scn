import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.scn import SCN
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True, help='A2C')
    parser.add_argument('--dataset', type=str, default='test', help='test')
    parser.add_argument('--pth_path', type=str, required=True,
                        help='pths/A2C/2023-01-01-00-00-00/100.pth')
    parser.add_argument('--model_key', type=str,
                        default='model_state_dict', help='model key')
    args = parser.parse_args()
    view = args.view
    dataset = args.dataset
    pth_path = args.pth_path
    model_key = args.model_key
    meta_dir = f'data/meta/{dataset}/{view}'
    ijk_dir = f'data/meta/3d_ijk/{view}'
    structs = utils.VIEW_STRUCTS[view]
    save_dir = f'evaluation/{view}'

    time = utils.current_time()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, time+'.csv')
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        header = ['name (landmark)', 'truth_x', 'truth_y', 'truth_z',
                  'pred_x', 'pred_y', 'pred_z', 'euclidean_distance']
        writer.writerow(header)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True,
                       norm_truth=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=4)

    model = SCN(1, len(structs), filters=128, factor=4, dropout=0.5).to(device)
    if model_key is None or model_key == '':
        model.load_state_dict(torch.load(
            pth_path, map_location=torch.device(device)))
    else:
        checkpoint = torch.load(pth_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint[model_key])
    model.eval()

    dists = []
    size = len(loader)
    with torch.no_grad():
        for batch, (echo, truth, struct, filename) in enumerate(loader):
            echo, truth = echo.to(device), truth.to(device)
            pred = model(echo)[0][0]

            pred_xyz = {}
            for i, channel in enumerate(pred):
                a, b, c = channel.shape
                index = torch.argmax(channel).item()
                x, y, z = index//(b*c), (index % (b*c))//c, (index % (b*c)) % c
                pred_xyz[structs[i]] = (float(x), float(y), float(z))

            truth_xyz = {}
            file_path = os.path.join(ijk_dir, filename[0]+'.csv')
            reader = csv.reader(open(file_path, 'r'))
            for row in reader:
                if reader.line_num == 1:
                    continue
                truth_xyz[int(row[1])] = (
                    float(row[2]), float(row[3]), float(row[4]))

            with open(save_path, 'a+') as file:
                writer = csv.writer(file)
                print(f'[{batch:>3d}/{size:>3d}]', end=' ')
                for st in structs:
                    t_xyz = truth_xyz.get(st)
                    p_xyz = pred_xyz[st]
                    row = [f'{filename[0]} ({st})']
                    if t_xyz:
                        row.extend(t_xyz)
                        t_xyz_ndarray = np.array(t_xyz)
                        p_xyz_ndarray = np.array(p_xyz)
                        dist = np.sqrt(
                            np.sum((t_xyz_ndarray-p_xyz_ndarray)**2)).item()
                        dists.append(dist)
                    else:
                        row.extend(['', '', ''])
                        dist = ''
                    row.extend(p_xyz)
                    row.append(dist)
                    writer.writerow(row)
                    print(dist, end=' ')
                print()

    dists = np.array(dists)
    with open(save_path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(['[median]', '', '', '', '', '', '', np.median(dists)])
        writer.writerow(['[mean]', '', '', '', '', '', '', dists.mean()])
        writer.writerow(['[std]', '', '', '', '', '', '', dists.std()])
    print(f'[median] {np.median(dists)}')
    print(f'[mean] {dists.mean()}')
    print(f'[std] {dists.std()}')
