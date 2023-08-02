import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import model_get
import pickle as pkl


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def loadGlove(inputpath, outputpath=""):
    data_list = []
    wordEmb = {}
    with open(inputpath) as f:
        for line in f:
            ll = line.strip().split(',')
            ll[0] = str(int(float(ll[0])))
            data_list.append(ll)

            ll_new = [float(i) for i in ll]
            emb = np.array(ll_new[1:], dtype="float32")
            wordEmb[str(int(ll_new[0]))] = emb

    if outputpath != "":
        with open(outputpath) as f:
            for data in data_list:
                f.writelines(' '.join(data))
    return wordEmb


def plotPrecisionRecallCurve(estimators, labels, xtests, ytests, flnm, icol=1):
    indx = 0
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for estimator in estimators:
        if len(ytests[indx].shape) == 2:
            pre, rec, _ = precision_recall_curve(
                ytests[indx][:, icol],
                estimator.predict(xtests[indx])[:, icol],
                pos_label=icol)
        else:
            pre, rec, _ = precision_recall_curve(
                ytests[indx],
                estimator.predict_proba(xtests[indx])[:, icol],
                pos_label=icol)
        #
        plt.plot(
            rec, pre,
            label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
                np.round(auc(rec, pre), 3))
        )
        indx += 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.savefig(flnm)


def plotRocCurve(
        estimators, labels,
        xtests, ytests,
        flnm, icol=1):
    indx = 0
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for estimator in estimators:
        if len(ytests[indx].shape) == 2:
            fprs, tprs, _ = roc_curve(
                ytests[indx][:, icol],
                estimator.predict(xtests[indx])[:, icol]
            )
        else:
            fprs, tprs, _ = roc_curve(
                ytests[indx],
                estimator.predict_proba(xtests[indx])[:, icol]
            )
        # print(estimator)
        plt.plot(
            fprs, tprs,
            label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
                np.round(auc(fprs, tprs), 3))
        )
        indx += 1
    #
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.savefig(flnm)


import random
import os
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    patience=10, verbose=0, mode='auto')
callbacks = [early_stopping]

dataset = 'hek293t'
num_classes = 2
epochs = 5
batch_size = 128#64
flpath = 'data/'


retrain = False

# CrisprNet-----------------------------------------------------------------------------------------------
print('CRISPR_Net')
open_name = 'encoded6x23' + dataset + '.pkl'
encoder_shape = (23, 6)
seg_len, coding_dim = encoder_shape
print('load data!')
print(open_name)
loaddata = pkl.load(
    open(flpath + open_name, 'rb'),
    encoding='latin1'
)

x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    loaddata.target,
    stratify=pd.Series(loaddata.target),
    test_size=0.2,
    shuffle=True,
    random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    stratify=pd.Series(y_train),
    test_size=0.2,
    shuffle=True,
    random_state=42)

neg = 0
for i in y_train:
    if i == 0:
        neg += 1
print(neg)

xtrain, xtest4, ytrain, ytest4, xval, yval, inputshape = model_get.CRISPR_Net_transformIO(
    x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

pos_indices = y_train == 1
pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
print(len(pos_y))
print(len(neg_y))

pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
resampled_ds = resampled_ds.batch(batch_size).prefetch(2)

resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
print(resampled_steps_per_epoch)

test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
test_ds = test_ds.batch(batch_size)

print('Training!!')

CRISPR_Net_model = model_get.CRISPR_Net_model(test_ds, resampled_steps_per_epoch, resampled_ds,
                                               xtrain, ytrain,
                                               xtest4,
                                               ytest4,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)
yscore = CRISPR_Net_model.predict(xtest4)
ypred = np.argmax(yscore, axis=1)
yscore = yscore[:, 1]
ytest = np.argmax(ytest4, axis=1)
eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]
for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
# cnnDNT-----------------------------------------------------------------------------------------------
print('new_model')
open_name = 'encodedmismatchtype14x23' + dataset + '.pkl'
encoder_shape = (23, 14)
seg_len, coding_dim = encoder_shape

print('load data!')
print(open_name)

loaddata = pkl.load(
    open(flpath + open_name, 'rb'),
    encoding='latin1'
)

x_train, x_test, y_train, y_test = train_test_split(
    loaddata.images,
    loaddata.target,
    stratify=pd.Series(loaddata.target),
    test_size=0.2,
    shuffle=True,
    random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    stratify=pd.Series(y_train),
    test_size=0.2,
    shuffle=True,
    random_state=42)

neg = 0
for i in y_train:
    if i == 0:
        neg += 1
print(neg)

xtrain, xtest3, ytrain, ytest3, xval, yval, inputshape = model_get.transformIO(
    x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

pos_indices = y_train == 1
pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
print(len(pos_y))
print(len(neg_y))
pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
print(resampled_steps_per_epoch)

test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
test_ds = test_ds.batch(batch_size)

print('Training!!')

new_model = model_get.crisprDNT(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                     xtest3,
                                     ytest3,
                                     inputshape, num_classes, batch_size, epochs, callbacks,
                                     open_name, retrain)

yscore = new_model.predict(xtest3)
ypred = np.argmax(yscore, axis=1)
yscore = yscore[:, 1]
ytest = np.argmax(ytest3, axis=1)
eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]
for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
# cnnCrispr-----------------------------------------------------------------------------------------------
print('cnn_crispr model')
print("GloVe model loaded")
VOCAB_SIZE = 16  # 4**3
EMBED_SIZE = 100
glove_inputpath = "data/keras_GloVeVec_" + dataset + "_5_100_10000.csv"
# load GloVe model
model_glove = loadGlove(glove_inputpath)
embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))
for i in range(VOCAB_SIZE):
    embedding_weights[i, :] = model_glove[str(i)]

open_name = 'encoded_CnnCrispr_' + dataset + '.pkl'


print('load data!')
print('load data!')
print(open_name)

loaddata = pkl.load(
    open(flpath + open_name, 'rb'),
    encoding='latin1'
)

x_train, x_test, y_train, y_test = train_test_split(
    np.array(loaddata.images),
    loaddata.target,
    stratify=pd.Series(loaddata.target),
    test_size=0.2,
    shuffle=True,
    random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    stratify=pd.Series(y_train),
    test_size=0.2,
    shuffle=True,
    random_state=42)

neg = 0
for i in y_train:
    if i == 0:
        neg += 1
print(neg)

xtrain, xtest5, ytrain, ytest5, xval, yval = model_get.offt_transformIO(x_train, x_test, y_train, y_test,
                                                                         x_val, y_val, num_classes)

pos_indices = y_train == 1
pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
print(len(pos_y))
print(len(neg_y))

pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
print(resampled_steps_per_epoch)

test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
test_ds = test_ds.batch(batch_size)

print('Training!!')

CnnCrispr_model = model_get.CnnCrispr(embedding_weights, test_ds, resampled_steps_per_epoch, resampled_ds,
                                       xtrain, ytrain,
                                       xtest5,
                                       ytest5, num_classes, batch_size, epochs, callbacks,
                                       open_name, retrain)

yscore = CnnCrispr_model.predict(xtest5)
ypred = np.argmax(yscore, axis=1)
yscore = yscore[:, 1]
ytest = np.argmax(ytest5, axis=1)
eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]
for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
#Crispr_IP-----------------------------------------------------------------------------------------------
print('crispr_ip_model')
encoder_shape = (23, 9)
seg_len, coding_dim = encoder_shape

open_name = 'encodedposition9x23' + dataset + '.pkl'


print('load data!')
print(open_name)

loaddata = pkl.load(
    open(flpath + open_name, 'rb'),
    encoding='latin1'
)

x_train, x_test, y_train, y_test = train_test_split(
    np.array(loaddata.images),
    loaddata.target,  # loaddata.target,
    stratify=pd.Series(loaddata.target),
    test_size=0.2,
    shuffle=True,
    random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,  # loaddata.target,
    stratify=pd.Series(y_train),
    test_size=0.2,
    shuffle=True,
    random_state=42)

neg = 0
for i in y_train:
    if i == 0:
        neg += 1
print(neg)



xtrain, xtest1, ytrain, ytest1, xval, yval, inputshape = model_get.transformIO(
    x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

pos_indices = y_train == 1
pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
print(len(pos_y))
print(len(neg_y))

pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
print(resampled_steps_per_epoch)

test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
test_ds = test_ds.batch(batch_size)


print('Training!!')

crispr_ip_model = model_get.crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                       xtest1,
                                       ytest1,
                                       inputshape, num_classes, batch_size, epochs, callbacks,
                                       open_name, retrain)

yscore = crispr_ip_model.predict(xtest1)
ypred = np.argmax(yscore, axis=1)
yscore = yscore[:, 1]
ytest = np.argmax(ytest1, axis=1)
eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]
for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
# cnn_std-----------------------------------------------------------------------------------------------
print('cnn_std')
encoder_shape = (23, 4)
seg_len, coding_dim = encoder_shape
open_name = 'encoded4x23' + dataset + '.pkl'

print('load data!')
print(open_name)

loaddata = pkl.load(
    open(flpath + open_name, 'rb'),
    encoding='latin1'
)

x_train, x_test, y_train, y_test = train_test_split(
    np.array(loaddata.images),
    loaddata.target,
    stratify=pd.Series(loaddata.target),
    test_size=0.2,
    shuffle=True,
    random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    stratify=pd.Series(y_train),
    test_size=0.2,
    shuffle=True,
    random_state=42)

neg = 0
for i in y_train:
    if i == 0:
        neg += 1
print(neg)

xtrain, xtest2, ytrain, ytest2, xval, yval, input_shape = model_get.cnn_std_transformIO(x_train, x_test,
                                                                                         y_train,
                                                                                         y_test, x_val, y_val,
                                                                                         seg_len, coding_dim,
                                                                                         num_classes)
pos_indices = y_train == 1
pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
print(len(pos_y))
print(len(neg_y))

pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
print(resampled_steps_per_epoch)

test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
test_ds = test_ds.batch(batch_size)

print('Training!!')

cnn_std_model = model_get.cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                   xtest2,
                                   ytest2, num_classes, batch_size, epochs, callbacks,
                                   open_name, retrain)

yscore = cnn_std_model.predict(xtest2)
ypred = np.argmax(yscore, axis=1)
yscore = yscore[:, 1]
ytest = np.argmax(ytest2, axis=1)
eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]
for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

models = [new_model, crispr_ip_model, cnn_std_model, CRISPR_Net_model, CnnCrispr_model]

labels = ['CRISPR_IP', 'CNN_std', 'CrisprDNT', 'CRISPR_Net', 'CnnCrispr']

xtests = [xtest1, xtest2, xtest3, xtest4, xtest5]

ytests = [ytest1, ytest2, ytest3, ytest4, ytest5]

roc_name = 'roccurve_compare_' + dataset + '.pdf'
pr_name = 'precisionrecallcurve_compare_' + dataset + '.pdf'

plotRocCurve(models, labels, xtests, ytests, roc_name)

plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)
