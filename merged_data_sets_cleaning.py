import pandas as pd


def clean_merged_file():
    df_ward_profiles = pd.read_csv("ward_median_house_merged.csv", sep=",", index_col=0)

    df_ward_profiles = df_ward_profiles[["Ward name", "New code", "% All Working-age (16-64) - 2015", "% All Older people aged 65+ - 2015",
                                        "Population density (persons per sq km) - 2013", "% Not Born in UK - 2011", "Below 33%", "Above 66%"]]

    df_ward_profiles.to_csv("ward_median_house_merged_cleaned.csv", index=False)


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


def clean_final_merged_set():
    df = pd.read_csv("LSOA_name_ward_median_house.csv")
    df["% All Working-age (16-64) - 2015"] = df["% All Working-age (16-64) - 2015"]/10
    df["% All Older people aged 65+ - 2015"] = df["% All Older people aged 65+ - 2015"] / 10
    df["% Not Born in UK - 2011"] = df["% Not Born in UK - 2011"] / 10

    df = df.drop(columns=["New code","Crime ID", "Reported by", "Crime type"])

    print(df.head())
    df.to_csv("Final_dataset.csv", index=False)
