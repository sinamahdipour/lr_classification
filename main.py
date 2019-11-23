# Sina Mp. Saravani
# Oct 2019
# Advanced Big Data Analytics
import numpy as np
import sys
import matplotlib.pyplot as plt
from texttable import Texttable

# read call arguments to a variable
args = sys.argv

# read .csv file of training and labels
dataset = np.loadtxt(open(args[1], "r"), delimiter=",")
labels = np.loadtxt(open(args[2], "r"))
# reshape the labels to be vertical
labels = np.reshape(labels, [dataset.shape[0], 1])


# function for normalizing data, find the min and max values and divides the distance of each value from min by
# the difference of max and min
def normalize_data(ds):
    max = np.max(ds)
    min = np.min(ds)
    normalized_ds = (ds - min) / (max - min)
    return normalized_ds


# function to add 1's to the data, creates a column of one's to be added to the feature set for having the bias in
# the coefficients. Concatenates this column with input matrix
def add_ones(ds):
    matones = [1 for i in range(ds.shape[0])]
    matones = np.reshape(matones, [ds.shape[0], 1])
    dso = np.concatenate((matones, ds), axis=1)
    return dso


# function to train ridge regression model on data. Receives the input matrix (samples * features), sample labels and
# the lambda value as inputs
def train_model(x_mat, trainlabels, lam):
    # first calculate the X'
    transx_mat = np.transpose(x_mat)
    # multiply X' by X to achieve X'X
    x_transx_mat = np.matmul(transx_mat, x_mat)
    # initialize an identity matrix with appropriate dimention
    i_mat = np.identity(x_mat.shape[1])
    # multiply the identity matrix by lambda value
    lambda_i_mat = lam * i_mat
    # sum X'X with lambda*I
    s_mat = x_transx_mat + lambda_i_mat
    # try to inverse the sum matrix
    try:
        sinv_mat = np.linalg.inv(s_mat)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        print("The X'X + lambda*I matrix is not invertible.")
        pass
    else:
        # Invertible, multiply the inverse with X' and name it tmp
        tmp = np.matmul(sinv_mat, transx_mat)
        # multiply tmp with Y to achieve final b matrix
        learned_coefs = np.matmul(tmp, trainlabels)
        # transpose b
        learned_transcoefs = np.reshape(learned_coefs, [x_mat.shape[1], 1])
        return learned_transcoefs


# the function to do the 10-fold cross validation, receives the featureset and labels concatenated together and
# the lambda as inputs
def cross_validation(concat_dataset, lamb):
    # first randomly shuffle the samples
    np.random.shuffle(concat_dataset)
    # Since we have 335 sample here, choose 33 for each fold, the final test fold will be 38 samples
    sec = 33
    # define three arrays for recording the measurements of accuracy, TPR, and FPR for 10 experiments
    exp_acc = np.zeros(10)
    exp_tpr = np.zeros(10)
    exp_fpr = np.zeros(10)
    # the experiments loop
    for k in range(0, 10):
        # if it is the last experiment, instead of 33 samples, add 38 samples to test set and the rest to the train set
        if k == 9:
            test = concat_dataset[k * sec:]
            deletables = np.array(range(k * sec, concat_dataset.shape[0]))
        else:
            test = concat_dataset[k*sec:k*sec+sec]
            deletables = np.array(range(k * sec, k * sec + sec))
        # delete the set set from the whole dataset to make the train set
        train = np.delete(concat_dataset, deletables, 0)
        # normalize and add one's to the train set and test set
        ntrain = normalize_data(train[:, :-1])
        ntrain = add_ones(ntrain)
        ntest = normalize_data(test[:, :-1])
        ntest = add_ones(ntest)
        # train the model
        cf = train_model(ntrain, train[:, -1], lamb)
        # predict the regression values for the test set
        res = np.matmul(ntest, cf)
        # use the 0.5 threshold to label the outputs as 5 or 6
        for j in range(0, res.shape[0]):
            if res[j] < 5.5:
                res[j] = 5
            else:
                res[j] = 6
        # we could also use the rounding function like below for more than 2 classes
        # res = np.rint(res)

        # compute accuracy and other measurements for this experiment
        # tp: true positive, fp: false positive, cp: condition positive, cn: condition negative, falses: total number
        # of wrongly labeled samples
        falses = 0
        tp, fp, cp, cn = 0, 0, 0, 0
        for i in range(0, test.shape[0]):
            if res[i] != test[i, -1]:
                # print(str(test[i, -1]) + " " + str(np.squeeze(res[i])))
                falses += 1
            if test[i, -1] == 6:
                cp += 1
                if res[i] == 6:
                    tp += 1
            if test[i, -1] == 5:
                cn += 1
                if res[i] == 6:
                    fp += 1
        # print("acc for this exp: " + str(((test.shape[0] - falses)/test.shape[0])*100))
        # enter the accuracy, tpr and fpr to the defined arrays
        exp_tpr[k] = tp/cp
        exp_fpr[k] = fp/cn
        exp_acc[k] = ((test.shape[0] - falses)/test.shape[0])*100
    # calculate the average accuracy for 10 experiments
    accuracy = np.sum(exp_acc)/10
    # print("Avg Accuracy for 10 Exp: " + str(acc))
    return accuracy, exp_acc, exp_tpr, exp_fpr


# concatenate features of samples with their labels
con_dataset = np.concatenate((dataset, labels), axis=1)
# repeat the experiment for many times to figure out what is the best lambda
max_acc = 0
best_lam = 0.1
for s in np.arange(0.01, 1, 0.01):
    # for each lambda candidate call the cross validation function and compare the the accuracy with the best accuracy
    # up to now
    acc, exp_acc, exp_tpr, exp_fpr = cross_validation(con_dataset, s)
    if acc > max_acc:
        max_acc = acc
        best_lam = s
# print("maximum acc: " + str(max_acc))
print("best lamda: " + str(best_lam))

# run the model with the best lambda found
acc, exp_acc, exp_tpr, exp_fpr = cross_validation(con_dataset, best_lam)
print("Avg Accuracy for 10 Exp: " + str(acc))

# print the performance measures table:
t = Texttable()
t.set_max_width(0)
headers = ['Exp_' + str(i) for i in range(1, 11)]
headers = ['   '] + headers
ar1, ar2, ar3 = list(), list(), list()
ar1.append('ACC')
ar2.append('TPR')
ar3.append('FPR')
for i in range(0, 10):
    ar1.append(exp_acc[i])
    ar2.append(exp_tpr[i])
    ar3.append(exp_fpr[i])
t.add_rows([headers, ar1, ar2, ar3])
print(t.draw())
