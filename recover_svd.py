import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.scn import SCN
import utils

pth_path = 'pths/tune/2023-02-06-15-53-33/100-latest.pth'
meta_dir = 'data/meta/test/A2C'
ijk_dir = 'data/meta/3d_ijk/A2C'
structs = [0, 5, 25]

save_dir = 'res/svd/A2C'


if __name__ == '__main__':
    time = utils.current_time()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, time+'.csv')
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        header = ['name', 'p_centroid', 't_centroid', 'centroid_dist',
                  'p_normal', 't_normal', 'normal_angle']
        writer.writerow(header)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True,
                       norm_truth=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=4)

    model = SCN(1, len(structs), filters=128, factor=4, dropout=0.5).to(device)
    model.load_state_dict(torch.load(
        pth_path, map_location=torch.device(device)))
    model.eval()

    centroid_error, normal_error = [], []
    size = len(loader)
    with torch.no_grad():
        for batch, (echo, truth, struct, filename) in enumerate(loader):
            echo, truth = echo.to(device), truth.to(device)
            pred = model(echo)[0][0]

            pred_xyz = []
            for i, channel in enumerate(pred):
                a, b, c = channel.shape
                index = torch.argmax(channel).item()
                x, y, z = index//(b*c), (index % (b*c))//c, (index % (b*c)) % c
                x, y, z = float(x), float(y), float(z)
                pred_xyz.append([x, y, z])
            pred_xyz = np.array(pred_xyz)
            pred_centroid, pred_normal = utils.fit_plane(pred_xyz)

            truth_xyz = []
            file_path = os.path.join(ijk_dir, filename[0]+'.csv')
            reader = csv.reader(open(file_path, 'r'))
            for row in reader:
                if reader.line_num == 1:
                    continue
                truth_xyz.append([float(row[2]), float(row[3]), float(row[4])])
            truth_xyz = np.array(truth_xyz)
            truth_centroid, truth_normal = utils.fit_plane(truth_xyz)

            centroid_distance = np.sqrt(
                np.sum((pred_centroid-truth_centroid)**2))
            normal_angle = utils.angle_between(pred_normal, truth_normal)

            with open(save_path, 'a+') as file:
                writer = csv.writer(file)
                data_row = [filename[0], pred_centroid, truth_centroid,
                            centroid_distance, pred_normal, truth_normal, normal_angle]
                writer.writerow(data_row)

            centroid_error.append(centroid_distance)
            normal_error.append(normal_angle)
            print(f'[{batch:>3d}/{size:>3d}] {centroid_distance} {normal_angle}')
    centroid_error = np.array(centroid_error)
    normal_error = np.array(normal_error)
    with open(save_path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(
            [f'Centroid Distance: [median]{np.median(centroid_error)} [mean]{centroid_error.mean()} [std]{centroid_error.std()}, Normal Angle: [median]{np.median(normal_error)} [mean]{normal_error.mean()} [std]{normal_error.std()}'])
