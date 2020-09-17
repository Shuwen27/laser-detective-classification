# laser-detective-classification

The document inluding two method for classification, a baseline using Logistic Regression, and a DNN based method

DNN_model is included in the file 'DNN_model.py'

DNN_training is included in the file 'DNN_train.py'

The file 'data_test' include the data generation and performance evaluation (cross validation).

'plot.py' for drawing ROV curves

To see the result, just run the file 'data_test.py', return a dictionary including two keys 'acc_LR' and 'acc_DNN' with their accuracy for each folders (cross validation process)

To see the ROV curves, run the file 'plot.py'. Two figure examples are shown as 'ROC_LC.pdf' and 'ROC_DNN.pdf'
