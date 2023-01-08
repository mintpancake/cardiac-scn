import os
import torch
from utils import draw
from models.scn import SCN
from dataset import EchoData

pth_path = 'pths/tune/2023-01-08-00-03-49/60.pth'

train_meta_path = 'data/meta/train/A2C'
val_meta_path = 'data/meta/val/A2C'
num_structs = 3

data_index = 0
channel, x = 0, 66

use_val = True
save_dir = 'images'

if __name__ == '__main__':
    train_dataset = EchoData(train_meta_path, norm_echo=True,
                          norm_truth=True, augmentation=False)
    val_dataset = EchoData(val_meta_path, norm_echo=True,
                        norm_truth=True, augmentation=False)
    if use_val:
        echo_data, truth_data, _ = val_dataset[data_index]
    else:
        echo_data, truth_data, _ = train_dataset[data_index]

    draw(echo_data.cpu().numpy()[0][x], os.path.join(save_dir, 'echo.png'))

    draw(truth_data.cpu().numpy()[channel][x], os.path.join(save_dir, 'truth.png'))

    echo_data.unsqueeze_(dim=0)
    model = SCN(1, num_structs, filters=128, factor=4, dropout=0.5).to('cpu')
    model.load_state_dict(torch.load(
        pth_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred, pred_HLA, pred_HSC = model(echo_data)
    print('pred', pred[0][channel][x].max(), pred[0][channel][x].min())
    print('pred_HLA', pred_HLA[0][channel]
          [x].max(), pred_HLA[0][channel][x].min())
    print('pred_HSC', pred_HSC[0][channel]
          [x].max(), pred_HSC[0][channel][x].min())

    draw(pred.cpu().numpy()[0][channel][x], os.path.join(save_dir, 'pred.png'))

    if torch.abs(pred_HLA.max()) < torch.abs(pred_HLA.min()):
        sign = -1
        print('Invert HLA')
    else:
        sign = 1
    pred_HLA *= sign
    draw(pred_HLA.cpu().numpy()[0][channel][x],
         os.path.join(save_dir, 'pred_HLA.png'))

    if torch.abs(pred_HSC.max()) < torch.abs(pred_HSC.min()):
        sign = -1
        print('Invert HSC')
    else:
        sign = 1
    pred_HSC *= sign
    draw(pred_HSC.cpu().numpy()[0][channel][x],
         os.path.join(save_dir, 'pred_HSC.png'))
