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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import pandas as pd
import os
import pickle
from sklearn import metrics
from pytorch_lightning.metrics.classification import ROC, AUROC


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
result_LR_dict = {}
result_DNN_dict = {}
num_folder = 0
for train_index,test_index in kfold.split(X_all):
    num_folder += 1
    # split data, one folder for test, the other four folders for training
    #print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    Y_train, Y_test = Y_all[train_index], Y_all[test_index]

    # Baseline LogisticRegression method
    model_LR = LogisticRegression().fit(X_train, Y_train)
    y_score = model_LR.predict_proba(X_test)

    #test
    acc_LR = model_LR.score(X_test, Y_test)
    print('acc_LR:', acc_LR)
    acc_dict['model_LR'].append(acc_LR)

    #save the result for each folder
    df = pd.DataFrame([[Y_test, y_score[:,1]]], columns=["y_test", "y_score"])
    result_LR_dict['Folder_{}_model_LR'.format(num_folder)] = df

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

    #save the result for each folder
    df = pd.DataFrame([[Y_test, output_test.data.numpy().squeeze()]], columns=["y_test", "y_score"])
    result_DNN_dict['Folder_{}_model_DNN'.format(num_folder)] = df

print("acc_dict:", acc_dict)
print('result_LR_dict:', result_LR_dict)
print('result_DNN_dict:', result_DNN_dict)

# save result for two models
data_path = os.getcwd()
with open('{}/{}.pickle'.format(data_path,'result_LR_dict'), 'wb') as handle:
    pickle.dump(result_LR_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('{}/{}.pickle'.format(data_path,'result_DNN_dict'), 'wb') as handle:
    pickle.dump(result_DNN_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
