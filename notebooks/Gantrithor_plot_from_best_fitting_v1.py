#!/usr/bin/env python
# coding: utf-8

# # Gantrithor: A machine learning model that detects fraud from transaction information.
# This is the initial version for our Gantrithor model,which is based on gradient boost and neural network. Furhter informtation will be added later.

# Updates: Try DART and do grid search

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd
import time
import tensorflow as tf
import keras
import seaborn as sns
from xgboost import XGBClassifier
from xgboost import plot_importance
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import copy
from scipy.stats import pearsonr
# get_ipython().run_line_magic('matplotlib', 'inline')

import os

# root_path = "../input/"
root_path = "Data/"

# Any results you write to the current directory are saved as output.


# # Read and visualize data

# In[3]:


print(os.listdir(root_path))
train_identity = pd.read_csv(root_path + 'train_identity.csv')
train_transaction = pd.read_csv(root_path + "train_transaction.csv")

test_identity = pd.read_csv(root_path + 'test_identity.csv')
test_transaction = pd.read_csv(root_path + "test_transaction.csv")
print("finish loading data")

# less memory:

# reduce memory usage:

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train_identity = reduce_mem_usage(train_identity)
train_transaction = reduce_mem_usage(train_transaction)

test_identity = reduce_mem_usage(test_identity)
test_transaction = reduce_mem_usage(test_transaction)

def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    # select missing data
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            # if it's an object, fill that with the *most common* object in that column
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)


# prepare data for training

def impute_cats(df):
    '''This function converts categorical and non-numeric
       columns into numeric columns to feed into a ML algorithm'''
    # Find the columns of object type along with their column index
    # only select columns with obejcts
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    # return the index for columns with object
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    # Encode the categorical columns with numbers
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    # It's still an object but this time with index from 0 to num_features-1
    ##!! I will modify this part later since I want to rank order these catagorical features by their fraud rate.
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:, i] = label_enc.fit_transform(df.iloc[:, i])


result = pd.concat([train_transaction.set_index('TransactionID'), train_identity.set_index('TransactionID')], axis=1,
                   sort=False)
# calculate pearsons correlation between C... V.. and fraud
y_nan = result.isnull().sum(axis=1)

# add y_nan
result["y_nan"] = y_nan

n = result.shape[1]
# index_label_name_part = np.r_[11:15,17:n]
index_label_name_part = np.r_[1:n]
label_names_part = result.iloc[:, index_label_name_part].columns

temp = result[label_names_part]

# Use mean strategy: Need to think about this
# temp.fillna(temp.mean())
fill_missing_values(temp)
# impute_cats(temp)


# In[5]:


y_fraud = result["isFraud"]

y_pearsons_array_part = []
# If a categorical column include too much categories, ignore it.
mask_too_long = []

for i in range(temp.shape[1]):
    if i % 50 == 0:
        print("Doing %d of %d for pearsons correlation" % (i, temp.shape[1]))
    try:
        corr, _ = pearsonr(temp[label_names_part[i]], y_fraud)
        y_pearsons_array_part.append(corr)
        mask_too_long.append(True)
    except:
        print("%s is a character" % label_names_part[i])
        label_enc = LabelEncoder()
        label_enc.fit(temp[label_names_part[i]])
        corr, _ = pearsonr(label_enc.transform(temp[label_names_part[i]]), y_fraud)
        y_pearsons_array_part.append(corr)
        if len(label_enc.classes_) > 100:
            print("Too long", len(label_enc.classes_), corr)
            mask_too_long.append(False)
        else:
            mask_too_long.append(True)

y_pearsons_array_part = np.array(y_pearsons_array_part)
mask_too_long = np.array(mask_too_long)

# In[6]:


# Visualize the correlation between each column and fraud
plt.hist(y_pearsons_array_part)

# In[ ]:


# In[ ]:


# In[7]:


mask_too_long.shape

# # One thing I want to add is: y_nan, which is proved to be "highly" correlated to fraud
# From the histogram for pearsons correlation, we can see -0.128 is not a small value :)

# In[8]:


# absolute value for 1 sigma. Let's select these features
percentile = 10
mask = (abs(y_pearsons_array_part) > np.percentile(abs(y_pearsons_array_part), percentile)) & (mask_too_long)
np.percentile(abs(y_pearsons_array_part), percentile)

# In[ ]:


# # I will prepare data for our model (Finally)

# In[9]:


N_train = result.shape[0]
# deal with imbalance:
print(len(train_transaction.loc[train_transaction["isFraud"] == 0]), "no-fraud")
print(len(train_transaction.loc[train_transaction["isFraud"] == 1]), "fraud")

# In[10]:


temp[label_names_part[mask]].shape

# In[ ]:


# In[11]:


result_submission = pd.concat([test_transaction.set_index('TransactionID'), test_identity.set_index('TransactionID')],
                              axis=1, sort=False)
# calculate pearsons correlation between C... V.. and fraud
y_nan_submission = result_submission.isnull().sum(axis=1)

# add y_nan
result_submission["y_nan"] = y_nan_submission

# concatenate:
X = pd.concat([temp[label_names_part[mask]], result_submission[label_names_part[mask]]])
print(X.shape)

# In[12]:


print(result_submission.shape, temp.shape)

# In[13]:


X.shape

# In[14]:


# Here we will use another method which is a little different than one hot encoder:
# rank the rellation between categorical data and fraud and encode them:


# In[15]:


# one hot encoder:
one_hot_X = pd.get_dummies(X)

# In[16]:


print("Finish data preparation")

# In[17]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
y = y_fraud

# In[22]:


# fill nan:
impute_cats(one_hot_X)

# In[23]:


one_hot_X.shape

# # Use XGBoost to have a try

# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(one_hot_X[:N_train], y, test_size=0.2, shuffle=True)

print("Shape of training set", X_train.shape, y_train.shape)

params_all = {"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
              "max_depth": [3, 5, 7, 9],
              " booster": ["gbtree", "dart", "gblinear"],
              "subsample": [1, 0.8, 0.65, 0.5]}

# remember this is a very simple toy model which doesn't use any train_identify file or grid search
time_start = time.time()
params = {}
params['booster'] = "gbtree"
params['learning_rate'] = 0.01
params['max_depth'] = 12
params['gpu_id'] = 0
params['max_bin'] = 512
params['tree_method'] = 'gpu_hist'
params['reg_alpha'] = 0.15
params['reg_lamdba'] = 0.85


model = XGBClassifier(n_estimators=3000, verbose=2, n_jobs=-1, **params)
model.fit(X_train, y_train)

# AUROC:

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib
from matplotlib.pylab import rc


def confusion_matrix(y_pred, y_true):
    TP = len(y_pred[(y_pred == 1) & (y_true == 1)])
    TN = len(y_pred[(y_pred == 1) & (y_true == 0)])
    # type1 error : false alarm
    FP = len(y_pred[(y_pred == 1) & (y_true == 0)])
    # type 2 error. Fail to make alarm
    FN = len(y_pred[(y_pred == 0) & (y_true == 1)])

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / len(y_pred)

    f1_score = 2 / (1 / precision + 1 / recall)
    return TP, TN, FP, FN, recall, precision, accuracy, f1_score


testy = y_test
prob = model.predict_proba(X_test)
probs = prob[:, 1]

auc = roc_auc_score(testy, probs)
print('AUROC: %.4f' % auc)

## draw ROC:
fpr, tpr, thresholds = roc_curve(testy, probs)

plt.close()

font = {'family': 'normal', 'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)
rc('axes', linewidth=3)

color_array = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

plt.plot(fpr, tpr, color='r', label='AUROC=%.4f' % auc)
plt.plot([0, 1], [0, 1], color='k', linewidth=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend()

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(14, 12)


plot_path = "/Users/caojunzhi/Downloads/upload_201909_Jeremy/"


save_path = plot_path+"AUROC_IEEE" + ".png"

fig.savefig(save_path, dpi=300)

plt.close()

# submission:
submission_csv = pd.read_csv(root_path + "sample_submission.csv")

X_submission = one_hot_X[N_train:]
fill_missing_values(X_submission)
# for submission:

df = X_submission
# try k fold:
n_k_fold=5
acc_i = []
auroc_i = []
accuracy_array = []
AUROC_array = []
probs_all = []
for i in range(n_k_fold):
    print("Doing %d of %d fold"%(i,n_k_fold))
    X_train, X_test, y_train, y_test = train_test_split(one_hot_X[:N_train], y, test_size=0.2, shuffle=True)

    params = {}
    params['booster'] = "gbtree"
    params['learning_rate'] = 0.008
    params['max_depth'] = 12
    params['gpu_id'] = 0
    params['max_bin'] = 512
    params['tree_method'] = 'gpu_hist'
    params['reg_alpha'] = 0.15
    params['reg_lamdba'] = 0.85

    model = XGBClassifier(n_estimators=5000, verbose=2, n_jobs=-1, **params)
    model.fit(X_train, y_train)

    # predict:

    # test
    Y_predict_test = model.predict(X_test)

    mask_good = abs(Y_predict_test - y_test) < 0.01

    # print("Good=%d Bad=%d" % (len(Y_predict_test[mask_good]), len(Y_predict_test) - len(Y_predict_test[mask_good])))

    acc_i.append(len(Y_predict_test[mask_good]) / len(Y_predict_test))

    prob = model.predict_proba(X_test)
    probs = prob[:, 1]

    auroc_i.append(roc_auc_score(y_test, probs))

    Y_predict_test_prob = model.predict_proba(df)

    probs_all.append(Y_predict_test_prob)

print("Accuracy=%.4f for testing set AUROC=%.4f" % (np.nanmean(acc_i), np.nanmean(auroc_i)))
accuracy_array.append(np.nanmean(acc_i))
AUROC_array.append(np.nanmean(auroc_i))

probs_all = np.array(probs_all)

# Y_predict_test_prob = model.predict_proba(df)
submission_csv["isFraud"] = np.nanmean(probs_all,axis=0)
submission_csv.to_csv("submissions/Gantrithor_v3_prob_best_fitting.csv", index=False)
