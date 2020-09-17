import numpy as np
import scipy.io
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import DNN_model
from DNN_model import *
from DNN_train import *
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# hyperparameter
EPOCHES = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
THRESHOLD = 0.5 # for classifying the two classes

#load the data
mat = scipy.io.loadmat('laser.mat')
X_all = mat.get('X')
Y_all = mat.get('Y').ravel()
# code output class: convert label (-1,1) to (0, 1)
Y_all = np.where(Y_all==-1, 0, 1)
# set kfold for cross validation, k=5
kfold = KFold(n_splits=5, shuffle=False)
# initilize accuracy dictionary to store the results
acc_dict = {'model_LR':[], 'model_DNN':[]}

for train_index,test_index in kfold.split(X_all):
    # split data, one folder for test, the other four folders for training
    #print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    Y_train, Y_test = Y_all[train_index], Y_all[test_index]

    # Baseline LogisticRegression method
    model_LR = LogisticRegression().fit(X_train, Y_train)
    #test
    acc_LR = model_LR.score(X_test, Y_test)
    print('acc_LR:', acc_LR)
    acc_dict['model_LR'].append(acc_LR)

    # DNN_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DNN = train_DNN(X_train,Y_train, EPOCHES, BATCH_SIZE, LEARNING_RATE)
    #test
    X_test_torch = torch.from_numpy(X_test).to(device).float()
    Y_test_torch = torch.from_numpy(Y_test)
    DNN.eval()
    with torch.no_grad():
        THRESHOLD = 0.5
        output_test = DNN(X_test_torch)
        acc_DNN = binary_acc(output_test, Y_test_torch, THRESHOLD)
        print('acc_DNN:',acc_DNN)
        acc_dict['model_DNN'].append(acc_DNN)

print("acc_dict:", acc_dict)
