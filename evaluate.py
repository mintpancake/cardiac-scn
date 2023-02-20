import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.scn import SCN
import utils


def save_txt(text, file):
    f = open(file, 'a+')
    f.write(str(text)+'\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, help='A2C')
    parser.add_argument('--dataset', type=str, help='test')
    parser.add_argument('--pth_path', type=str,
                        help='pths/A2C/2023-01-01-00-00-00/100.pth')
    args = parser.parse_args()
    view = args.view
    dataset = args.dataset
    pth_path = args.pth_path
    meta_dir = f'data/meta/{dataset}/{view}'
    ijk_dir = f'data/meta/3d_ijk/{view}'
    structs = utils.VIEW_STRUCTS[view]
    save_dir = f'evaluation/{view}'

    time = utils.current_time()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, time+'.txt')
    open(save_path, 'w').close()
    save_txt(f'pth_path: {pth_path}', save_path)
    save_txt(f'meta_dir: {meta_dir}', save_path)
    save_txt(f'ijk_dir: {ijk_dir}', save_path)
    save_txt('', save_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True,
                       norm_truth=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=4)

    model = SCN(1, len(structs), filters=128, factor=4, dropout=0.5).to(device)
    model.load_state_dict(torch.load(
        pth_path, map_location=torch.device(device)))
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

            dist = 0.0
            num = 0
            for st in structs:
                if truth_xyz.get(st):
                    num += 1
                    t_xyz = np.array(truth_xyz[st])
                    p_xyz = np.array(pred_xyz[st])
                    dist += np.sqrt(np.sum((t_xyz-p_xyz)**2))
            dist /= num
            dists.append(dist)
            save_txt(
                f'[{batch:>3d}/{size:>3d}] {filename[0]} {dist}', save_path)
            save_txt(truth_xyz, save_path)
            save_txt(pred_xyz, save_path)
            save_txt('', save_path)

            print(f'[{batch:>3d}/{size:>3d}] {dist}')
    dists = np.array(dists)
    save_txt(f'[median] {np.median(dists)}', save_path)
    print(f'[median] {np.median(dists)}')
    save_txt(f'[mean] {dists.mean()}', save_path)
    print(f'[mean] {dists.mean()}')
    save_txt(f'[std] {dists.std()}', save_path)
    print(f'[std] {dists.std()}')
