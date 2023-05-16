import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
import time

train_data = pd.read_csv("burglary_train.csv")
val_data = pd.read_csv("burglary_validation.csv")
test_data = pd.read_csv("burglary_test.csv")

# The LSOA name will be the y, so create new data for it
Y_train = train_data["LSOA name"]
Y_val = val_data["LSOA name"]
Y_test = test_data["LSOA name"]

# Delete the y from the x data and unwanted columns
drop = ["Crime type", "Last outcome category", "Reported by", "Crime ID", "Location", "LSOA name"]
X_train = train_data.drop(labels=drop, axis=1, inplace=False)
X_val = val_data.drop(labels=drop, axis=1, inplace=False)
X_test = test_data.drop(labels=drop, axis=1, inplace=False)

dtrain_reg = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, Y_test, enable_categorical=True)

# Define hyperparameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss', # Log loss as the evaluation metric
    'tree_method': 'gpu_hist'     
}         

n = 10
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)


