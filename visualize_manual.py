import os
import torch
from torch import nn
from utils import draw
from models.scn import SCN
from dataset import EchoData

pth_path = 'pths/tune/2023-01-08-00-03-49/100.pth'
save_dir = 'images'

train_meta_path = 'data/meta/train/A2C'
val_meta_path = 'data/meta/val/A2C'
num_structs = 3

# data/nrrd/A2C/PWHOR191326532P_1Nov2021_E1W04Z16_3DQ.nrrd,0,59.6267809031786,59.20592574032862,102.81748813845164
# data/nrrd/A2C/PWHOR191326532P_1Nov2021_E1W04Z16_3DQ.nrrd,5,67.51562838702642,67.85268018021235,57.949426460171246
# data/nrrd/A2C/PWHOR191326532P_1Nov2021_E1W04Z16_3DQ.nrrd,25,58.96937694619128,58.48536287033831,56.97403381499123
use_val = True
data_index = 1
channel, x = 0, 60
supervised_channels = [0, 1, 2]


def criterion(pred, truth, channels):
    channels = torch.tensor(channels)
    loss_fn = nn.MSELoss(reduction='sum').to('cpu')
    loss = loss_fn(torch.index_select(pred, 0, channels),
                   torch.index_select(truth, 0, channels))
    return loss/len(channels)


if __name__ == '__main__':
    train_dataset = EchoData(train_meta_path, norm_echo=True,
                             norm_truth=True, augmentation=False)
    val_dataset = EchoData(val_meta_path, norm_echo=True,
                           norm_truth=True, augmentation=False)
    if use_val:
        echo_data, truth_data, _s, _f = val_dataset[data_index]
    else:
        echo_data, truth_data, _s, _f = train_dataset[data_index]

    draw(echo_data.cpu().numpy()[0][x], os.path.join(
        save_dir, 'echo.png'), mode='raw')

    draw(truth_data.cpu().numpy()[channel][x],
         os.path.join(save_dir, 'truth.png'))

    echo_data.unsqueeze_(dim=0)
    model = SCN(1, num_structs, filters=128, factor=4, dropout=0.5).to('cpu')
    model.load_state_dict(torch.load(
        pth_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred, pred_HLA, pred_HSC = model(echo_data)
    print('loss', criterion(pred[0], truth_data, supervised_channels))
    print('pred', pred[0][channel][x].max(), pred[0][channel][x].min())
    print('pred_HLA', pred_HLA[0][channel]
          [x].max(), pred_HLA[0][channel][x].min())
    print('pred_HSC', pred_HSC[0][channel]
          [x].max(), pred_HSC[0][channel][x].min())

    draw(pred.cpu().numpy()[0][channel][x], os.path.join(save_dir, 'pred.png'))

    if torch.abs(pred_HLA.max()) < torch.abs(pred_HLA.min()) and torch.abs(pred_HSC.max()) < torch.abs(pred_HSC.min()):
        sign = -1
        print('Invert HLA HSC')
    else:
        sign = 1
    pred_HLA *= sign
    pred_HSC *= sign

    draw(pred_HLA.cpu().numpy()[0][channel][x],
         os.path.join(save_dir, 'pred_HLA.png'))

    draw(pred_HSC.cpu().numpy()[0][channel][x],
         os.path.join(save_dir, 'pred_HSC.png'))
