import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
from xgboost import DMatrix

import matplotlib.pyplot as plt
import seaborn as sns
import time

train_data = pd.read_csv("burglary_train.csv")
val_data = pd.read_csv("burglary_validation.csv")
test_data = pd.read_csv("burglary_test.csv")

# The LSOA name will be the y, so create new data for it
y_train = train_data["LSOA name"]
y_val = val_data["LSOA name"]
y_test = test_data["LSOA name"]

# Change the y data to numerical data for model by replacing letter at the end and removing the word Barnet
mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
print('The LSOA name will be changed to numerical data by replacing letter at the end: ' + str(mapping))
print('For example, Barnet 023A will be 230.')

num_train = np.array([int(value.split()[1][:3]) for value in y_train])
let_train = np.array([mapping[value.split()[1][-1]] for value in y_train])
y_train = num_train * 10 + let_train


num_val = np.array([int(value.split()[1][:3]) for value in y_val])
let_val = np.array([mapping[value.split()[1][-1]] for value in y_val])
y_val = num_val * 10 + let_val
y_val = y_val.reshape(-1, 1)

num_test = np.array([int(value.split()[1][:3]) for value in y_test])
let_test = np.array([mapping[value.split()[1][-1]] for value in y_test])
y_test = num_test * 10 + let_test
y_test = y_test.reshape(-1, 1)
print("The outcome data is now numerical.")

# Delete the y from the x data and unwanted columns
drop = ["Crime type", "Last outcome category", "Reported by", "Crime ID", "Location", "LSOA name"]
x_train = train_data.drop(labels=drop, axis=1, inplace=False)
x_val = val_data.drop(labels=drop, axis=1, inplace=False)
x_test = test_data.drop(labels=drop, axis=1, inplace=False)

# Change month data to numerical data
x_train["Month"] = x_train["Month"].str.replace("-", "").astype(int)
x_val["Month"] = x_val["Month"].str.replace("-", "").astype(int)
x_test["Month"] = x_test["Month"].str.replace("-", "").astype(int)

# Extract numerical data of LSOA code
x_train["LSOA code"] = x_train["LSOA code"].str[1:].astype(int)
x_val["LSOA code"] = x_val["LSOA code"].str[1:].astype(int)
x_test["LSOA code"] = x_test["LSOA code"].str[1:].astype(int)

# Encoding the data
data_list = [x_train, x_val, x_test]

def data_encoder(lst: list):
    for data in lst:
        if 'LSOA code' in data.columns:
            encoder = LabelEncoder()
            try:
                encoded_labels = encoder.fit_transform(data['LSOA code'])
                data['LSOA_Encoded'] = np.where(encoded_labels <= 210, encoded_labels, -1)
            except Exception as e:
                print(f"Error occurred while encoding column: {e}")
                continue
        else:
            print("Column 'LSOA code' not found in dataframe.")

data_encoder(data_list)

# Create DMatrix to input in the model
dtrain_reg_1 = xgb.DMatrix(x_train, y_train, enable_categorical = True)
dtest_reg_1 = xgb.DMatrix(x_test, y_test, enable_categorical = True)

# Define hyperparameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss', # Log loss as the evaluation metric
    'tree_method': 'gpu_hist'     
}         

# Train model
n = 10
model = xgb.train(
   params=params,
   dtrain=dtrain_reg_1,
   num_boost_round=n,
)


