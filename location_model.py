import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
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
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
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

# Set format of Location and LSOA code to category
x_train['Location'] = x_train['Location'].astype('category')
x_train['LSOA code'] = x_train['LSOA code'].astype('category')

x_val['Location'] = x_val['Location'].astype('category')
x_val['LSOA code'] = x_val['LSOA code'].astype('category')

x_test['Location'] = x_test['Location'].astype('category')
x_test['LSOA code'] = x_test['LSOA code'].astype('category')

# Change format of the Month column and set it to index
x_train['Time'] = pd.to_datetime(x_train['Month'], format='%Y-%m')
x_val['Time'] = pd.to_datetime(x_val['Month'], format='%Y-%m')
x_test['Time'] = pd.to_datetime(x_test['Month'], format='%Y-%m')

# Set the time column as the index
# x_train.set_index('Time', inplace=True)
# x_val.set_index('Time', inplace=True)
# x_test.set_index('Time', inplace=True)

exog = ['Location', 'LSOA code']

# Transformer: Ordinal encoding + cast to category type
pipeline_categorical = make_pipeline(
                            OrdinalEncoder(
                                dtype=int,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1
                            ),
                            FunctionTransformer(
                                func=lambda x: x.astype('category'),
                                feature_names_out= 'one-to-one'
                            )
                       )
transformer_exog = make_column_transformer(
                        (
                            pipeline_categorical,
                            make_column_selector(dtype_exclude=np.number)
                        ),
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                   ).set_output(transform="pandas")

# Create forecaster with automatic categorical detection
forecaster = ForecasterAutoregDirect(
    regressor=xgb.XGBRFRegressor(tree_method='hist', random_state=123, enable_categorical='auto'),
    lags=12,
    steps=12,
    transformer_exog=transformer_exog
)

# Define hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Lags used as predictors
lags_grid = [4, 12, 24, 48, 72]

# Obtaining best model
x = pd.concat([x_train, x_val])
y = np.append(y_train, y_val)
y = pd.Series(y)

result_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=y,
    exog=x[exog],
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=12,
    refit=False,
    metric='mean_absolute_error',
    initial_train_size=len(x_train),
    fixed_train_size=False,
    return_best=True,
    verbose=False
)

# Backtesting with the test data the best model
x2 = pd.concat([x, x_test])
y = np.append(y, y_test)
y = pd.Series(y)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    exog=x2[exog],
    initial_train_size=len(x),
    fixed_train_size=False,
    steps=12,
    refit=False,
    metric='mean_absolute_error',
    verbose=False
)
print(predictions)

# Predictions for the next year
future_exog = x_test[exog]  # Exogenous features for the future period
predictions = forecaster.predict(steps=12, exog=future_exog)

# Convert the predicted labels back to the original LSOA names
label_encoder = result_grid.best_forecaster_._label_encoder
predicted_lsoa_names = label_encoder.inverse_transform(predictions)

# Add the predicted LSOA names to the test data
print('\n' + str(predicted_lsoa_names))

# Save the test data with predictions to a CSV file
predicted_lsoa_names.to_csv('burglary_test_predictions1.csv', index=False)
predictions.to_csv('burglary_test_predictions2.csv', index=False)