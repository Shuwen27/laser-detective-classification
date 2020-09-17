import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
import DNN_model
from DNN_model import *
#import tikzplotlib

def train_DNN(X_train,Y_train, EPOCHES, BATCH_SIZE, LEARNING_RATE):
    """"
    training the DNN model
    return the trained DNN
    """""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train_torch = torch.from_numpy(X_train)
    Y_train_torch = torch.from_numpy(Y_train)

    dataset = TensorDataset(X_train_torch, Y_train_torch)
    dataloaderr = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=2, drop_last=True)
    DNN = FFNet()
    episode = 0
    optimizer = Adam(DNN.parameters(),lr=LEARNING_RATE)
    DNN.to(device)
    print('start')
    for episode_i in range(episode,EPOCHES):
        #print('episode:', episode_i)
        counter = 0
        DNN.train()
        for batchh in dataloaderr:
            counter += 1
            input_data =batchh[0].to(device).float()
            gt_data = batchh[1].to(device).float()
            # zero old gradients
            optimizer.zero_grad()
            # compute loss
            loss = nn.BCELoss()
            # predict output with DNN
            output = DNN(input_data)
            batch_error = loss(output, gt_data)
            # backpropagate loss
            batch_error.backward()
            # clip gradients
            #gradient_clipping(DNN)
            #learn
            optimizer.step()
    return DNN
