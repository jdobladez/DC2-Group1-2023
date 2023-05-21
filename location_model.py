import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

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
drop = ["Crime type", "Last outcome category", "Reported by", "Crime ID", "Location", "LSOA name"]
x_train = train_data.drop(labels=drop, axis=1, inplace=False)
x_val = val_data.drop(labels=drop, axis=1, inplace=False)
x_test = test_data.drop(labels=drop, axis=1, inplace=False)

# Convert "Month" and "LSOA code" columns to numeric
x_train["Month"] = pd.to_numeric(x_train["Month"].str.replace("-", ""))
x_train["LSOA code"] = pd.to_numeric(x_train["LSOA code"].str[1:])

x_val["Month"] = pd.to_numeric(x_val["Month"].str.replace("-", ""))
x_val["LSOA code"] = pd.to_numeric(x_val["LSOA code"].str[1:])

x_test["Month"] = pd.to_numeric(x_test["Month"].str.replace("-", ""))
x_test["LSOA code"] = pd.to_numeric(x_test["LSOA code"].str[1:])

# Create DMatrix to input in the model
dtrain_reg_1 = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
dval_reg_1 = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
dtest_reg_1 = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

# Define hyperparameters
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'eval_metric': 'merror',  # Classification error as the evaluation metric
    'num_class': 211,  # Number of classes
    'tree_method': 'gpu_hist'
}

# Train model
n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg_1,
   num_boost_round=n,
)

# Make predictions on the test set
predictions = model.predict(dtest_reg_1)
predictions = predictions.astype(int)  # Convert predictions to integers

# Decode label predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Evaluate the model
accuracy = np.mean(predicted_labels == test_data["LSOA name"])
print(f"Accuracy: {accuracy}")
