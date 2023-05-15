import os
import pandas as pd

# Define function to remove all files that are unnecessary
def rem_files(path):
    # Define the word to filter files by
    filter_word = 'metropolitan'
    # Loop through all the files in the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file name does not contain the filter word
            if filter_word not in file:
                print(file + "has been removed")
                # Remove the file
                os.remove(os.path.join(root, file))
    print("\n" + "Every unwanted file has been removed")

def merge_csv_files(path, output_file_name):
    # Initialize an empty list to store the contents of all CSV files
    # Loop through all files in the directory and its subdirectories
    df_csv_concat = pd.DataFrame()
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            df_csv = pd.read_csv(file_path)
            df_csv_concat = pd.concat([df_csv_concat, df_csv], ignore_index=True)
            print("\n" + file + " has been merged")
    df_csv_concat = df_csv_concat.loc[df_csv_concat["Crime type"] == "Burglary"]
    df_csv_concat = df_csv_concat.loc[(df_csv_concat["LSOA name"] == "NaN") | df_csv_concat["LSOA name"].str.contains("Barnet", case=False)]
    df_csv_concat.to_csv(output_file_name, index=False)
    print("The merge data has been saved as " + output_file_name + " in " + str(os.getcwd()))
    return df_csv_concat


# Define the path where the files are
path = 'Files/'
# Define the output file name for the merged CSV files
merged_csv_file_name = 'init-merge.csv'
# Call the functions defined above
rem_files(path)
df = merge_csv_files(path, merged_csv_file_name)

df = df.drop(['Context', 'Type', 'Date', 'Part of a policing operation', "Policing operation", "Gender", "Age range",
              "Self-defined ethnicity", "Officer-defined ethnicity", "Legislation", "Outcome", "Object of search",
              "Outcome linked to object of search", "Removal of more than just outer clothing", "Outcome type",
              "Falls within"], axis=1)


df1 = df.dropna(subset=['Latitude', 'Longitude', 'LSOA code', 'LSOA name'])

df1 = df1[df1["LSOA name"].str.contains("Barnet", case=False)]

df1.to_csv("burglary_data.csv", index=False)

print("\n" + "File for burglary data in Barnet without NAN values and removed columns saved as: burglary_data.csv")
