import math
import pandas as pd

# TODO only needs to be run once!!

# Constants
train_percentage = 0.8  # Percentage of train data as a decimal number, test data is 1 - train_percentage


def split_train_test_data(train_percentagex):
    # Import df from csv - already sorted on timestamp so no need to sort
    df = pd.read_csv("Final_dataset.csv")

    # Convert month to datetime and sort ascending
    # df['Month'] = pd.to_datetime(df['Month'], format="%Y-%m")
    df.sort_values(by='Month', inplace=True)

    # Create index number to split on
    split_border = len(df) * train_percentagex

    # Split df into train and test data set
    df_training = df[:math.floor(split_border)]
    df_test = df[math.ceil(split_border):]

    # Remove duplicated data
    # Should not be the case here but have to double check

    # Convert df training to csv
    df_training.to_csv('burglary_train.csv')

    # Convert df test to csv
    df_test.to_csv('burglary_test.csv')

    print("Train len: ", len(df_training))
    print("Test len: ", len(df_test))
    

split_train_test_data(train_percentage)
