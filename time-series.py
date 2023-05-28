import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "seaborn"
plt.style.use('seaborn-v0_8-darkgrid')
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

train_data = pd.read_csv("burglary_train.csv")
val_data = pd.read_csv("burglary_validation.csv")
test_data = pd.read_csv("burglary_test.csv")

# The LSOA name will be the y, so create new data for it
y_train = train_data["LSOA name"]
y_val = val_data["LSOA name"]
y_test = test_data["LSOA name"]

# Encode labels with integer values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Delete the y from the x data and unwanted columns
drop = ["Crime type", "Last outcome category", "Reported by", "Crime ID", "LSOA name"]
x_train = train_data.drop(labels=drop, axis=1, inplace=False)
x_val = val_data.drop(labels=drop, axis=1, inplace=False)
x_test = test_data.drop(labels=drop, axis=1, inplace=False)

# Create DMatrix to input in the model
dtrain_reg_1 = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
dval_reg_1 = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
dtest_reg_1 = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

# Create forecaster
forecaster = ForecasterAutoregMultiOutput(regressor=xgb.XGBRFRegressor(random_state=42), lags=12)

# Define hyperparameters
params = {
    'n_estimators': 'multi:softmax',  # Multi-class classification
    'eval_metric': 'merror',  # Classification error as the evaluation metric
    'num_class': 211,  # Number of classes
    'tree_method': 'gpu_hist'
}

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Lags used as predictors
lags_grid = [4, 12, 24, 48, 72]

# Obtaining best model
x = pd.concat([x_train, x_val])
y = pd.concat([y_train, y_val])

result_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=y,
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=36,
    refit=False,
    metric='mean_squared_error',
    initial_train_size=len(x_train),
    fixed_train_size=False,
    return_best=True,
    verbose=False
)

# Backtesting test data
x2 = pd.concat([x, x_test])
y = pd.concat([y, y_test])

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    initial_train_size=len(x),
    fixed_train_size=False,
    steps=36,
    refit=False,
    metric='mean_squared_error',
    verbose=False
)

