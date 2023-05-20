import numpy as np
import pandas as pd


def merge_files():
    # Load csv files
    df_median_prices = pd.read_csv("house_median_prices.csv", sep=",")
    df_ward_profiles = pd.read_csv("ward_profiles.csv", sep=",")
    df_train = pd.read_csv("burglary_train.csv")

    # Drop weird column
    df_median_prices = df_median_prices.drop(columns="Unnamed: 0")

    df_min_max = df_median_prices.agg(['min', 'max'])
    # df_min_max = df_min_max.drop(columns=["Local authority code", "Local authority name"])

    # Calculate the mean for every row in columns that contain numbers
    numeric_columns = df_median_prices.select_dtypes(include=[float, int]).columns
    df_median_prices['Mean'] = df_median_prices[numeric_columns].mean(axis=1)

    # Calculate percentiles
    n33th = np.percentile(df_median_prices[numeric_columns], 33)
    n66th = np.percentile(df_median_prices[numeric_columns], 66)

    # Check if mean value falls in percentile ranges
    df_median_prices['Below 33%'] = n33th > df_median_prices["Mean"]
    df_median_prices['Above 66%'] = n66th < df_median_prices["Mean"]

    # Add new columns to train df
    # Low range means that the area is cheap, high range means that the area is expensive
    df_train["House price in low range"] = df_median_prices['Below 33%']
    df_train['House price in high range'] = df_median_prices['Above 66%']

    print(df_min_max.head())
    print(df_median_prices.head())

    # print(df_median_prices.columns.values.tolist())
    # df_median_prices.to_csv('house_median_prices.csv')


merge_files()
