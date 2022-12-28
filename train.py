import os
import time
import argparse
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EchoData
from models.scn import SCN
import utils


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


class Trainer:
    def __init__(self, config):
        self.init_time = utils.current_time()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.view = config['view']
        self.structs = sorted(utils.VIEW_STRUCTS[self.view])
        self.print_interval = config['print_interval']
        self.save_interval = config['save_interval']
        self.log_interval = config['log_interval']

        self.pth_path = os.path.join(config['pth_path'], self.init_time)
        self.log_path = os.path.join(config['log_path'], self.init_time)
        os.makedirs(self.pth_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = SummaryWriter(self.log_path)

        self.train_data = EchoData(config['train_meta_path'])
        self.val_data = EchoData(config['val_meta_path'])

        self.train_loader = DataLoader(
            self.train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=0)
        self.val_loader = DataLoader(
            self.val_data, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=0)

        self.epochs = config['epochs']
        self.model = SCN(1, len(self.structs)).to(self.device)
        self.loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        self.optimizer = SGD(self.model.parameters(
        ), lr=config['lr'], momentum=0.99, weight_decay=config['weight_decay'])

        self.total_train_step = 0
        self.total_val_step = 0
        self.start_time = 0.0
        self.end_time = 0.0

    def train(self):
        self.model.train()
        size = len(self.train_loader.dataset)

        for batch, (echo, truth, structs) in enumerate(self.train_loader):
            echo, truth = echo.to(self.device), truth.to(self.device)
            pred = self.model(echo)[0]
            loss = self.criterion(pred, truth, structs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % self.print_interval == 0:
                loss_val, curr = loss.item(), batch*len(echo)
                print(f'loss: {loss_val:>7f}  [{curr:>5d}/{size:>5d}]')

            self.total_train_step += 1
            if self.total_train_step % self.log_interval == 0:
                self.logger.add_scalar(
                    'training loss', loss.item(), self.total_train_step)

    def eval(self):
        self.model.eval()
        size = len(self.val_loader.dataset)
        val_loss = 0.0

        with torch.no_grad():
            for echo, truth, structs in self.val_loader:
                echo, truth = echo.to(self.device), truth.to(self.device)
                pred = self.model(echo)[0]
                loss = self.criterion(pred, truth, structs)
                val_loss += len(echo)*loss.item()

        val_loss /= size
        self.end_time = time.time()
        print(f'Test error: \n'
              f'  Avg loss: {val_loss:>8f} \n'
              f'      Time: {(self.end_time - self.start_time):>8f} \n')
        self.total_val_step += 1

        if self.total_val_step % self.log_interval == 0:
            self.logger.add_scalar(
                'validation loss', val_loss, self.total_val_step)

    def criterion(self, pred, truth, structs):
        loss = 0
        for i in range(len(pred)):
            if len(structs[i]) == len(self.structs):
                loss += self.loss_fn(pred[i], truth[i])
            else:
                diff = list(set(self.structs).difference(set(structs[i])))
                diff_idx = [
                    utils.VIEW_STRUCTS[self.view].index(_) for _ in diff]
                pred_i, truth_i = pred[i], truth[i]
                for j in diff_idx:
                    pred_i = del_tensor_ele(pred_i, j)
                    truth_i = del_tensor_ele(truth_i, j)
                loss += self.loss_fn(pred_i, truth_i)
        return loss

    def start(self):
        print(f"Training on {self.device}...")
        self.start_time = time.time()

        for t in range(self.epochs):
            print(
                f'Epoch {t+1} ({utils.current_time()})\n-----------------------------')

            self.train()
            self.eval()

            if (t+1) % self.save_interval == 0:
                pth_file_path = os.path.join(self.pth_path, f'{str(t+1)}.pth')
                torch.save(self.model.state_dict(), pth_file_path)

        pth_file_path = os.path.join(self.pth_path, 'latest.pth')
        torch.save(self.model.state_dict(), pth_file_path)

        self.logger.close()
        print(
            f'Completed {self.epochs} epochs; saved in "{self.pth_path}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cardiac SCN Trainer')
    parser.add_argument(
        '--config', type=str, default='configs/default.json', help='configuration path')
    args = parser.parse_args()
    trainer = Trainer(utils.load_config(args.config))
    trainer.start()
