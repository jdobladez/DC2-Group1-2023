import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
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

print("The input data has been preprocessed and is ready to be used in the model")

# Specify the arguments
learning_rates = [1.0, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]
reports = []

for lr in learning_rates:
    start_time = time.time()
    # Define the model
    print('\n'+f"Model is being defined for learning rate {lr}")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr)

    # Train the model
    print("\n"+"Model is being trained")
    gb.fit(x_train, y_train)
    print("Model is trained")

    # Evaluate model on validation dataset
    print("\n"+"Model is predicting the validation dataset")
    y_val_pred = gb.predict(x_val)
    val_report = classification_report(y_val, y_val_pred, zero_division=0)
    print("Model was evaluated on validation dataset")

    # Evaluate model on testing dataset
    print("\n"+"Model is predicting the test dataset")
    y_pred = gb.predict(x_test)
    test_report = classification_report(y_test, y_pred, zero_division=0)
    print("Model was evaluated on testing dataset")

    # Store the reports for each learning rate
    reports.append((lr, val_report, test_report))

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion matrix for learning rate {lr}')
    plt.show()

    # Save the plots
    plt.savefig(f'conf_mat_lr_{lr}.png')

    # Print running time taken per learning rate
    end_time = time.time()
    print('\n' + f"Running time for learning rate {lr}: {(end_time - start_time)/60:.2f} minutes\n")

# Print the results
for lr, v, t in reports:
    print('\n' + f"Learning Rate: {lr}")
    print("Validation set performance:")
    print(v)
    print("Testing set performance:")
    print(t)


# Extract validation reports, and testing reports from the reports
validation_reports = [v for _, v, _ in reports]
testing_reports = [t for _, _, t in reports]

# Define the metrics to plot
metrics = ['precision', 'recall', 'f1-score']

# Iterate over the metrics
for metric in metrics:
    # Get the scores for validation and testing datasets
    validation_scores = [report[metric] for report in validation_reports]
    testing_scores = [report[metric] for report in testing_reports]

    # Set the positions of the bars on the x-axis
    x = range(len(validation_scores))

    # Plot the bars
    plt.figure()
    plt.bar(x, validation_scores, color='olivedrab', label='Validation')
    plt.bar(x, testing_scores, color='cadetblue', label='Testing', alpha=0.5)

    # Add x-axis ticks and labels
    plt.xticks(x, learning_rates)

    # Add labels and title
    plt.xlabel('Learning Rate')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} for Different Learning Rates')
    plt.legend()

    # Show the plot
    plt.show()



