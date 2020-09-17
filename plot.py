import numpy as np
import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn import metrics
import tikzplotlib

model_name = 'DNN'
# load data
data_path = os.getcwd()
with open(os.path.join(data_path, 'result_{}_dict.pickle'.format(model_name)), 'rb') as handle:
    result_dict = pickle.load(handle)

i = 0
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for key, value in result_dict.items():
    i += 1
    pred = value['y_score'].squeeze()
    target = value['y_test'].squeeze()
    # ROC curve
    fpr, tpr, thresholds= roc_curve(target, pred)
    # Area under Curve
    roc_auc = metrics.auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)

    plt.title('DNN_Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'Fold{}: AUC = {:.2f}'.format(i,roc_auc), alpha=0.4, lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

#Chance curve
plt.plot([0, 1], [0, 1],linestyle='-.',color='r', lw=2,label = 'Chance',alpha=.5)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.legend(loc='lower right')
#plt.show()
plt.savefig("ROC_{}.pdf".format(model_name))
#tikzplotlib.save("ROC_{}.tex".format(model_name))
