import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.linear1 = nn.Linear(60, 60)
        self.linear2 = nn.Linear(60, 30)
        self.batchnorm1 = nn.BatchNorm1d(60)
        self.batchnorm2 = nn.BatchNorm1d(30)
        self.drop1 = nn.Dropout(p = 0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(30, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)# flatten the input to (batch_size,)
        x = self.relu(self.linear1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.linear2(x))
        x = self.batchnorm2(x)
        x = self.drop1(x)
        output = self.sigmoid(self.out(x))
        output = output.view(-1)
        return output


def binary_acc(y_predict, y_test, threshold):
    """"""""""
    compute accuracy
    """""""""
    target_y = y_test.data.numpy().squeeze()
    pred_y = y_predict.data.numpy().squeeze()
    pred_y = np.where(pred_y <= threshold, 0, 1)
    #print('pred_y:', pred_y)
    #print('target_y:', target_y)
    acc = sum(pred_y == target_y)/target_y.shape[0]
    return acc
