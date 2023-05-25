import pandas as pd


def clean_file():
    df_ward_profiles = pd.read_csv("ward_median_house_merged.csv", sep=",", index_col=0)

    df_ward_profiles.drop(columns=["% English is First Language of no one in household - 2011"])

    df_ward_profiles.to_csv("ward_median_house_merged_cleaned.csv", index=False)


def merge_additional_with_main():
    burglary_data = pd.read_csv("burglary_data.csv", index_col=0)
    df_ward_profiles = pd.read_csv("ward_median_house_merged.csv", sep=",", index_col=0)

    print(burglary_data.head())


def clean_LSOA_to_Ward():
    df_LSOA_Ward = pd.read_csv("Lower_Layer_Super_Output_Area_(2001)_to_Ward_(2011)_Lookup_in_England_and_Wales.csv")
    df_wards_merged = pd.read_csv("ward_median_house_merged_cleaned.csv", sep=",", index_col=0)
    df_wards_merged.reset_index()
    df_wards_list = df_wards_merged.index

    df_LSOA_Ward["Boolean"] = df_LSOA_Ward["WD11NM"].isin(df_wards_list)
    df_LSOA_Ward = df_LSOA_Ward[df_LSOA_Ward["Boolean"]]
    df_LSOA_Ward = df_LSOA_Ward.drop(columns=["Boolean", "FID", "LSOA01CD", "WD11CD", "WD11CDO"])

    print(df_LSOA_Ward.head())

    df_LSOA_Ward.to_csv("LSOA_name_and_Ward_codes.csv", index=False)


