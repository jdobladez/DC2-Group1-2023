import pandas as pd


def merge_files():
    df_median_prices = pd.read_csv("house_median_prices.csv", sep=",")
    df_ward_profiles = pd.read_csv("ward_profiles.csv", sep=",")

    df_median_prices = df_median_prices.drop(columns="Unnamed: 0")

    # df_median_prices.to_csv('house_median_prices.csv')

    df_min_max = df_median_prices.agg(['min', 'max'])
    # df_min_max = df_min_max.drop(columns=["Local authority code", "Local authority name"])

    print(df_min_max.head())
    print(df_median_prices.head())

    # print(df_median_prices.columns.values.tolist())


merge_files()
