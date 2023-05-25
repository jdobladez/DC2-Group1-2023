import numpy as np
import pandas as pd
import requests


def merge_median_house_prices():
    # Load csv files
    df_median_prices = pd.read_csv("house_median_prices.csv", sep=",", index_col=0)
    df_train = pd.read_csv("burglary_train.csv", index_col=0)

    # df_min_max = df_median_prices.agg(['min', 'max'])
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

    # print(df_median_prices.head())
    # print(df_train.head())

    median_prices_subset = df_median_prices["Ward name", "Below 33%", "Above 66%"]

    # print(df_median_prices.columns.values.tolist())
    df_median_prices.to_csv('house_median_prices.csv', index=False)
    df_train.to_csv("burglary_train_merged.csv")


def merge_ward_profiles():
    df_ward_profiles = pd.read_csv("ward_profiles_cleaned.csv", sep=",", index_col=0)
    df_train = pd.read_csv("burglary_train.csv", index_col=0)

    print(df_ward_profiles.head())


def merge_ward_median():
    # Read CSV
    df_median_prices = pd.read_csv("house_median_prices.csv", sep=",", index_col=0)
    df_ward_profiles = pd.read_csv("ward_profiles_cleaned.csv", sep=",", index_col=0)

    df_median_prices.reset_index(inplace=True)
    df_ward_profiles.reset_index(inplace=True)

    # Create subset of relevant columns that need to be joined
    median_prices_subset = df_median_prices[["Ward name", "Below 33%", "Above 66%"]].copy()
    median_prices_subset.sort_values("Ward name", inplace=True)

    # print(median_prices_subset.head())

    # Copy the relevant columns to ward profiles df since ward names match
    df_ward_profiles["Below 33%"] = median_prices_subset["Below 33%"]
    df_ward_profiles["Above 66%"] = median_prices_subset["Above 66%"]

    # print(df_ward_profiles.head(10))

    df_ward_profiles.to_csv("ward_median_house_merged.csv", index=False)


def lsoa_to_ward():
    df_data = pd.read_csv("burglary_data.csv", sep=",", index_col=0)

    df_data["Ward"] = get_ward_from_lsoa(df_data["LSOA code"])

    print(df_data.head())


def get_ward_from_lsoa(lsoa_code):

    api_url = 'https://statistics.data.gov.uk/area?areaType=LSOA&areaCode=' + str(lsoa_code)
    response = requests.get(api_url)
    data = response.json()

    if 'result' in data:
        ward_name = data['result']['primaryTopic']['ward']['label']
        return ward_name
    else:
        return None

