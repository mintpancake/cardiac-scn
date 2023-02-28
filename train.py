import os
import time
import argparse
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EchoData
from models.scn import SCN
from criterion.adaptive_wing_loss import AdaptiveWingLoss
from criterion.weighted_loss import WeightedAdaptiveWingLoss
import utils


class Trainer(object):
    def __init__(self, config, ckpt_path=None):
        # continue training
        self.resume = False
        if ckpt_path is not None and ckpt_path != '':
            self.resume = True
            try:
                self.ckpt = torch.load(ckpt_path)
            except Exception:
                print('Error: cannot load checkpoint!')
                exit()

        self.init_time = utils.current_time()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.view = config['view']
        self.structs = torch.IntTensor(sorted(utils.VIEW_STRUCTS[self.view]))
        self.print_interval = config['print_interval']
        self.save_interval = config['save_interval']
        self.log_interval = config['log_interval']

        self.pth_path = os.path.join(config['pth_path'], self.init_time)
        self.log_path = os.path.join(config['log_path'], self.init_time)
        os.makedirs(self.pth_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.pth_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(self.log_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        self.logger = SummaryWriter(self.log_path)
        self.console_path = os.path.join(self.log_path, 'console_history.txt')
        console_file = open(self.console_path, 'w')
        console_file.close()

        self.train_data = EchoData(config['train_meta_path'], norm_echo=True,
                                   norm_truth=True, augmentation=True)
        self.val_data = EchoData(config['val_meta_path'], norm_echo=True,
                                 norm_truth=True, augmentation=False)

        self.train_loader = DataLoader(self.train_data, batch_size=config['batch_size'],
                                       shuffle=True, drop_last=False, num_workers=8)
        self.val_loader = DataLoader(self.val_data, batch_size=config['batch_size'],
                                     shuffle=False, drop_last=False, num_workers=8)

        self.epochs = config['epochs']
        self.model = SCN(1, len(self.structs), filters=128,
                         factor=4, dropout=0.5).to(self.device)
        if self.resume:
            self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.loss_fn = WeightedAdaptiveWingLoss(
            reduction='sum').to(self.device)
        # self.loss_fn = AdaptiveWingLoss(reduction='sum').to(self.device)
        # self.loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        # self.loss_fn = nn.L1Loss(reduction='sum').to(self.device)
        # self.loss_fn = nn.SmoothL1Loss(reduction='sum', beta=1.0).to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(
        # ), lr=config['lr'], momentum=0.99, nesterov=True, weight_decay=config['weight_decay'])
        self.optimizer = optim.AdamW(self.model.parameters(
        ), lr=config['lr'], weight_decay=config['weight_decay'])
        if self.resume:
            self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
        # self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 30, 0.1)
        if self.resume:
            self.lr_scheduler.load_state_dict(
                self.ckpt['lr_scheduler_state_dict'])

        self.start_time = 0.0
        self.end_time = 0.0
        self.total_train_step = 0
        self.total_val_step = 0
        self.last_val_loss = float('inf')
        self.best_epoch = 0
        if self.resume:
            self.total_train_step = self.ckpt['total_train_step']
            self.total_val_step = self.ckpt['total_val_step']
            self.last_val_loss = self.ckpt['last_val_loss']
            self.best_epoch = self.ckpt['best_epoch']

    def train(self):
        self.print('Train loss:')
        self.model.train()
        size = len(self.train_loader.dataset)
        train_loss = 0

        for batch, (echo, truth, structs, _) in enumerate(self.train_loader):
            echo, truth = echo.to(self.device), truth.to(self.device)
            pred = self.model(echo)[0]
            loss = self.criterion(pred, truth, structs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += len(echo)*loss.item()
            if batch % self.print_interval == 0:
                loss_val, curr = loss.item(), batch*len(echo)
                self.print(f'train: {loss_val:.9e} [{curr:>3d}/{size:>3d}]')

            self.total_train_step += 1
            if self.total_train_step % self.log_interval == 0:
                self.logger.add_scalar(
                    'training loss', loss.item(), self.total_train_step)

        train_loss /= size
        self.print(f'train: {train_loss:.9e} [average]')
        return train_loss

    def eval(self):
        self.print('Val loss:')
        self.model.eval()
        size = len(self.val_loader.dataset)
        val_loss = 0.0

        with torch.no_grad():
            for batch, (echo, truth, structs, _) in enumerate(self.val_loader):
                echo, truth = echo.to(self.device), truth.to(self.device)
                pred = self.model(echo)[0]
                loss = self.criterion(pred, truth, structs)
                val_loss += len(echo)*loss.item()

                if batch % self.print_interval == 0:
                    loss_val, curr = loss.item(), batch*len(echo)
                    self.print(
                        f'valid: {loss_val:.9e} [{curr:>3d}/{size:>3d}]')

        val_loss /= size
        self.end_time = time.time()
        self.print(f'valid: {val_loss:.9e} [average]')
        self.print(f'Time: {(self.end_time - self.start_time):>8f}\n')
        self.total_val_step += 1

        if self.total_val_step % self.log_interval == 0:
            self.logger.add_scalar(
                'validation loss', val_loss, self.total_val_step)

        return val_loss

    def criterion(self, pred, truth, structs):
        loss = 0.0
        for i in range(len(pred)):
            if len(structs[i]) == len(self.structs):
                loss += self.loss_fn(pred[i], truth[i])/len(structs[i])
            else:
                structs_idx = sorted(
                    [utils.VIEW_STRUCTS[self.view].index(_) for _ in structs[i].tolist()])
                structs_idx = torch.IntTensor(structs_idx).to(self.device)

                pred_i = torch.index_select(pred[i], dim=0, index=structs_idx)
                truth_i = torch.index_select(
                    truth[i], dim=0, index=structs_idx)

                loss += self.loss_fn(pred_i, truth_i)/len(structs[i])
        return loss/len(pred)

    def start(self):
        self.print(f'Training on {self.device}...')
        self.start_time = time.time()
        start_epoch = 0
        if self.resume:
            start_epoch = self.ckpt['epoch']
            self.print(
                f'Continue training from epoch {start_epoch+1} with lr={self.lr_scheduler.get_last_lr()}...')

        for t in range(start_epoch, self.epochs):
            self.print(
                f'Epoch {t+1} ({utils.current_time()})\n------------------------------')

            self.train()
            val_loss = self.eval()
            self.lr_scheduler.step()
            self.print(f'lr decreased to {self.lr_scheduler.get_last_lr()}')

            if (t+1) % self.save_interval == 0:
                pth_file_path = os.path.join(self.pth_path, f'{str(t+1)}.pth')
                ckpt_dict = {
                    'epoch': t,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'total_train_step': self.total_train_step,
                    'total_val_step': self.total_val_step,
                    'last_val_loss': self.last_val_loss,
                    'best_epoch': self.best_epoch,
                }
                torch.save(ckpt_dict, pth_file_path)
                self.print(f'Checkpoint saved to {pth_file_path}...')

            if val_loss <= self.last_val_loss:
                pth_file_path = os.path.join(
                    self.pth_path, f'{self.best_epoch}-best.pth')
                if os.path.exists(pth_file_path):
                    os.remove(pth_file_path)
                self.best_epoch = t+1
                pth_file_path = os.path.join(
                    self.pth_path, f'{self.best_epoch}-best.pth')
                ckpt_dict = {
                    'epoch': t,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'total_train_step': self.total_train_step,
                    'total_val_step': self.total_val_step,
                    'last_val_loss': self.last_val_loss,
                    'best_epoch': self.best_epoch,
                }
                torch.save(ckpt_dict, pth_file_path)
                self.print(f'Best checkpoint saved to {pth_file_path}...')
                self.last_val_loss = val_loss

        pth_file_path = os.path.join(
            self.pth_path, f'{self.epochs}-latest.pth')
        ckpt_dict = {
            'epoch': t,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'total_train_step': self.total_train_step,
            'total_val_step': self.total_val_step,
            'last_val_loss': self.last_val_loss,
            'best_epoch': self.best_epoch,
        }
        torch.save(ckpt_dict, pth_file_path)

        self.print(
            f'Completed {self.epochs} epochs\nLatest checkpoint saved to {pth_file_path}')
        self.logger.close()

    def print(self, text):
        print(text)
        console_file = open(self.console_path, 'a+')
        console_file.write(text+'\n')
        console_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cardiac SCN Trainer')
    parser.add_argument('--config', type=str,
                        default='configs/default.json',
                        help='configuration path')
    parser.add_argument('--ckpt', type=str,
                        default=None, help='continue checkpoint path')
    args = parser.parse_args()
    trainer = Trainer(utils.load_config(args.config), ckpt_path=args.ckpt)
    trainer.start()
