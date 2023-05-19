import pandas as pd


def add_files_to_git():
    df_median_prices = pd.read_csv(
        "D:\\OneDrive\\Universiteit\\Year 2\\Q4\\JBG050 (Data Challenge 2)\\Median house prices by ward\\Barnet - Median price paid by ward.csv", encoding= 'unicode_escape')
    df_ward_profiles = pd.read_csv(
        "D:\\OneDrive\\Universiteit\\Year 2\\Q4\\JBG050 (Data Challenge 2)\\ward-profiles-excel-version.csv", encoding= 'unicode_escape')

    df_median_prices.to_csv('house_median_prices.csv')
    df_ward_profiles.to_csv('ward_profiles.csv')



