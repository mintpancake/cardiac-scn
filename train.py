import os
import time
import argparse
import json
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EchoData
from models.scn import SCN
import utils


def weight_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.trunc_normal_(m.weight, mean=0, std=1e-2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Trainer(object):
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
        with open(os.path.join(self.pth_path, '_CONFIG.json'), 'w') as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(self.log_path, '_CONFIG.json'), 'w') as f:
            json.dump(config, f, indent=4)
        self.logger = SummaryWriter(self.log_path)
        self.console = open(os.path.join(
            self.log_path, 'console_history.txt'), 'w')

        self.train_data = EchoData(config['train_meta_path'])
        self.val_data = EchoData(config['val_meta_path'])

        self.train_loader = DataLoader(
            self.train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=4)
        self.val_loader = DataLoader(
            self.val_data, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=4)

        self.epochs = config['epochs']
        self.model = SCN(1, len(self.structs)).to(self.device)
        self.model.apply(weight_init)
        self.loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        self.optimizer = SGD(self.model.parameters(
        ), lr=config['lr'], momentum=0.99, weight_decay=config['weight_decay'])

        self.total_train_step = 0
        self.total_val_step = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.last_val_loss = float('inf')
        self.best_epoch = 0

    def __del__(self):
        self.logger.close()
        self.console.close()

    def train(self):
        self.model.train()
        size = len(self.train_loader.dataset)
        num_batches = len(self.train_loader)
        train_loss = 0

        for batch, (echo, truth, structs) in enumerate(self.train_loader):
            echo, truth = echo.to(self.device), truth.to(self.device)
            pred = self.model(echo)[0]
            loss = self.criterion(pred, truth, structs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            if batch % self.print_interval == 0:
                loss_val, curr = loss.item(), batch*len(echo)
                self.print(f'loss: {loss_val:>9f} [{curr:>5d}/{size:>5d}]')

            self.total_train_step += 1
            if self.total_train_step % self.log_interval == 0:
                self.logger.add_scalar(
                    'training loss', loss.item(), self.total_train_step)

        train_loss /= num_batches
        self.print(f'loss: {train_loss:>9f} [  average  ]')
        return train_loss

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
        text = f'Test error: \n'\
               f'  Avg loss: {val_loss:>8f} \n'\
               f'      Time: {(self.end_time - self.start_time):>8f} \n'
        self.print(text)
        self.total_val_step += 1

        if self.total_val_step % self.log_interval == 0:
            self.logger.add_scalar(
                'validation loss', val_loss, self.total_val_step)

        return val_loss

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
                    pred_i = utils.del_tensor_ele(pred_i, j)
                    truth_i = utils.del_tensor_ele(truth_i, j)
                loss += self.loss_fn(pred_i, truth_i)
        return loss

    def start(self):
        self.print(f'Training on {self.device}...')
        self.start_time = time.time()

        for t in range(self.epochs):
            self.print(
                f'Epoch {t+1} ({utils.current_time()})\n------------------------------')

            self.train()
            val_loss = self.eval()

            if (t+1) % self.save_interval == 0:
                pth_file_path = os.path.join(self.pth_path, f'{str(t+1)}.pth')
                torch.save(self.model.state_dict(), pth_file_path)

            if val_loss <= self.last_val_loss:
                pth_file_path = os.path.join(self.pth_path, f'{self.best_epoch}-best.pth')
                if os.path.exists(pth_file_path):
                    os.remove(pth_file_path)
                self.best_epoch = t+1
                pth_file_path = os.path.join(self.pth_path, f'{self.best_epoch}-best.pth')
                torch.save(self.model.state_dict(), pth_file_path)

        pth_file_path = os.path.join(
            self.pth_path, f'{self.epochs}-latest.pth')
        torch.save(self.model.state_dict(), pth_file_path)

        self.print(
            f'Completed {self.epochs} epochs; saved in "{self.pth_path}"')

    def print(self, text):
        print(text)
        self.console.write(text+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cardiac SCN Trainer')
    parser.add_argument(
        '--config', type=str, default='configs/default.json', help='configuration path')
    args = parser.parse_args()
    trainer = Trainer(utils.load_config(args.config))
    trainer.start()
