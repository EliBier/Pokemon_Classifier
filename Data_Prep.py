# Pokemon 7000 data preparation

# %% Import os & shutil module
import os
import shutil
import random


# %%
def enumerate_files(path):
    """Enumerates the files in all subdirectories of the given path, starting from 1 in each subdirectory."""
    count = 1
    for root, dirs, files in os.walk(path):
        for file in files:
            file_ext = os.path.splitext(file)[1]  # get the file extension
            new_file_name = str(count) + file_ext  # create the new file name
            os.rename(
                os.path.join(root, file), os.path.join(root, new_file_name)
            )  # rename the file
            count += 1


def create_train_test_datasets(path, split_ratio):
    # Create the train and test directories
    train_dir = os.path.join(path, "train_gen1")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(path, "test_gen1")
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through the subfolders and move the images to the appropriate directory
    for subdir, dirs, files in os.walk(path):
        # Determine the number of files to use for the training set
        num_files = len(files)
        num_train = int(num_files * split_ratio)
        num_test = num_files - num_train

        # Shuffle the files randomly
        random.shuffle(files)

        # Move the files to the appropriate directory
        for i, filename in enumerate(files):
            src_path = os.path.join(subdir, filename)
            if i < num_train:
                dst_path = os.path.join(train_dir, os.path.relpath(src_path, path))
            else:
                dst_path = os.path.join(test_dir, os.path.relpath(src_path, path))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)


# Get the current working directory (CWD)
cwd = "C:/Users/elias/OneDrive/Neural-Final"

enumerate_files(cwd + "/pokemon_7000")
create_train_test_datasets(cwd + "/pokemon_7000")


# %%


# %%
