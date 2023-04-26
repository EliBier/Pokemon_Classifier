# Pokemon 7000 data preparation

#%% Import os & shutil module
import os
import shutil
import pandas as pd


# %%
def enumerate_files(path):
    """Enumerates the files in all subdirectories of the given path, starting from 1 in each subdirectory."""
    count = 1
    for root, dirs, files in os.walk(path):
        for file in files:
            file_ext = os.path.splitext(file)[1]  # get the file extension
            new_file_name = str(count) + file_ext  # create the new file name
            os.rename(os.path.join(root, file), os.path.join(root, new_file_name))  # rename the file
            count += 1

# Get the current working directory (CWD) 
cwd = os.getcwd()

enumerate_files(cwd + '/pokemon_7000')


# %%
def create_pokemon_df(path, df):
    # Create an empty dataframe with the desired columns
    poke_new_df = pd.DataFrame(columns=['No', 'Name', 'Type 1', 'Type 2'])

    # Iterate through the subdirectories in the given path
    for dir_name in os.listdir(path):
        # Skip any directories that start with '.' or are not alphabetical
        if dir_name.startswith('.') or not dir_name.isalpha():
            continue
        dir_path = os.path.join(path, dir_name)

        # Iterate through the files in each subdirectory
        for file_name in os.listdir(dir_path):
            # Get the file name without the file extension
            file_base_name = os.path.splitext(file_name)[0]

            # Get the pokemon's name, type 1, and type 2 from the input dataframe
            poke_row = df.loc[df['Name'].str.lower() == dir_name.lower()]
            if not poke_row.empty:
                name = poke_row['Name'].values[0]
                type_1 = poke_row['Type 1'].values[0]
                type_2 = poke_row['Type 2'].values[0]
            else:
                name, type_1, type_2 = '', '', ''

            # Add a new row to the dataframe
            poke_new_df.loc[file_base_name] = [file_base_name, name, type_1, type_2]

    # Save the dataframe as a CSV file
    poke_new_df.to_csv('pokemon_more.csv')

    # Return the new dataframe
    return poke_new_df


# %%
poke_df = pd.read_csv("pokemon.csv")

create_pokemon_df(cwd+'/pokemon_7000', poke_df)

# %%
