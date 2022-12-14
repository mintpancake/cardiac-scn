import numpy as np
import torch
from torch import nn
from scipy import ndimage


class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y))) * \
            (self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega * \
            torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs(
            (y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind] * \
            torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        return lossMat


class WeightedAdaptiveWingLoss(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5, reduction='mean'):
        super().__init__()
        self.W = float(W)
        self.AWing = AdaptiveWingLoss(alpha, omega, epsilon, theta)
        self.reduction = reduction

    def generate_weight_map(self, y):
        '''
        y: [c x l x w x h]
        '''
        heatmap = torch.clone(y).cpu()
        heatmap_numpy = heatmap.numpy()
        weight_map = torch.zeros_like(heatmap)
        dilate = np.zeros_like(heatmap_numpy)
        for i, h in enumerate(heatmap_numpy):
            dilate[i] = ndimage.grey_dilation(h, size=(3, 3, 3))
        weight_map[np.where(dilate > 0.2)] = 1
        return weight_map.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, y_pred, y):
        M = self.generate_weight_map(y)
        loss = self.AWing(y_pred, y)
        weighted = loss * (self.W * M + 1.)
        if self.reduction == 'sum':
            return weighted.sum()
        else:
            return weighted.mean()
