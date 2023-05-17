import math
import pandas as pd

# TODO only needs to be run once!!

# Constants - the sum of all should be 1
train = 0.7
validation = 0.1
test = 0.2  # Test percentage is 1 - train_percentage - validation_percentage


def split_train_test_data(train_percentage, validation_percentage):
    # Import df from csv - already sorted on timestamp so no need to sort
    df = pd.read_csv("burglary_data.csv")

    # Create index number to split on
    split_border_train_valid = len(df) * train_percentage
    split_border_valid_test = len(df) * validation_percentage

    # Split df into train and test data set
    df_training = df[:math.floor(split_border_train_valid)]
    df_test_and_validation = df[math.ceil(split_border_train_valid):]

    # Split validation and test df into separate df
    df_validation = df_test_and_validation[:math.floor(split_border_valid_test)]
    df_test = df_test_and_validation[math.ceil(split_border_valid_test):]

    # Remove duplicated data
    # Should not be the case here but have to double check

    # Convert df to csv
    df_training.to_csv('burglary_train.csv')
    df_validation.to_csv("burglary_validation.csv")
    df_test.to_csv('burglary_test.csv')

    print("Df len: ", len(df))
    print("Train len: ", len(df_training))
    print("Validation len: ", len(df_validation))
    print("Test len: ", len(df_test))


split_train_test_data(train, validation)