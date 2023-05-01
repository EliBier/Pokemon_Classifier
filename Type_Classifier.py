# %%
"""
Import statements
"""
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import random
from keras import regularizers
from tensorflow.keras.preprocessing import image

from keras.preprocessing import image
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
from keras import layers
from keras.callbacks import ModelCheckpoint

from utils import *
import glob
from pathlib import Path

# %%
"""
Setup Pandas Dataframe
"""
poke_df = pd.read_csv("pokemon.csv")
poke_df = poke_df.rename(columns={"#": "No"})

poke_df = poke_df.rename(columns={"Type 1": "Type1"})
poke_df = poke_df.rename(columns={"Type 2": "Type2"})
poke_df.loc[poke_df["Type1"].isin(["Dark", "Steel", "Fairy"]), "Type1"] = ""
poke_df.loc[poke_df["Type2"].isin(["Dark", "Steel", "Fairy"]), "Type2"] = ""

poke_df = poke_df.drop_duplicates(subset=["No"])

poke_df["Type2"] = poke_df.Type2.fillna("")
conditions = [poke_df["Type2"] == "", poke_df["Type2"] != ""]

values = [poke_df["Type1"], poke_df["Type1"] + "|" + poke_df["Type2"]]

poke_df["Types"] = poke_df["Type1"] + "," + poke_df["Type2"]
poke_df["Types"] = poke_df["Types"].apply(
    lambda s: [l for l in str(s).split(",") if l not in [""]]
)

poke_df = poke_df[poke_df["Generation"] == 1]
poke_df = poke_df.loc[:, ["No", "Name", "Type1", "Type2", "Types"]]

poke_df
# %%
"""
Get images from files
"""


def getImages(path):
    png_files = glob.glob(path + "/**/*/*.png", recursive=True)

    png_filter = [i for i in png_files if Path(i).stem.isdigit()]

    gen_poke = pd.DataFrame(png_filter)
    indices = []
    for path in png_filter:
        name = path.split("\\")[-2]
        match = poke_df[poke_df["Name"] == name]
        if len(match) > 0:
            indices.append(match.index[0])
        else:
            print(f"No match found for pokemon name {name}")
    gen_poke["Types"] = [poke_df["Types"][int(i)] for i in indices]

    gen_poke = gen_poke.rename(columns={0: "sprites"})

    X_train, X_val, y_train, y_val = train_test_split(
        gen_poke["sprites"], gen_poke["Types"], test_size=0.2, random_state=44
    )

    X_train.index = list(range(len(X_train)))
    y_train.index = list(range(len(y_train)))

    X_val.index = list(range(len(X_val)))
    y_val.index = list(range(len(y_val)))

    return X_train, X_val, y_train, y_val


X_train, X_val, y_train, y_val = getImages(
    "C:/Users/elias/OneDrive/Neural-Final/pokemon_7000"
)
X_train = pd.concat(
    [X_train],
    ignore_index=True,
)
X_val = pd.concat([X_val], ignore_index=True)
y_train = pd.concat(
    [y_train],
    ignore_index=True,
)
y_val = pd.concat([y_val], ignore_index=True)
# %%
"""
Shuffle data
"""
p = np.random.permutation(len(y_train))

X_train = X_train[p]
y_train = y_train[p]

X_train.index = list(range(len(X_train)))
y_train.index = list(range(len(y_train)))

q = np.random.permutation(len(y_val))

X_val = X_val[q]
y_val = y_val[q]

X_val.index = list(range(len(X_val)))
y_val.index = list(range(len(y_val)))
# %%
"""
Binarize using one-hot encoding
"""
# Fit the multi-label binarizer on the training set
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for i, label in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print example of Pokemon png files and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])
# %%
"""
Resize and normalize images so that they can be used with transfer learning
"""


def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


# %%
"""
Define constants
"""
IMG_SIZE = (
    224  # Specify height and width of image to match the input format of the model
)
CHANNELS = 3  # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 256
AUTOTUNE = (
    tf.data.experimental.AUTOTUNE
)  # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024  # Shuffle the training data by a chunck of 1024 observations

# %%
"""
Create training and testing dataset
"""


def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

# %%
"""
Get pre-trained feature extractor layers and make them not trainable (Google's training is going to be better than anything we can do here).
"""
feature_extractor_url = (
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
)
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)
)

# this prevents the parameters of the pre-trained model being adjusted during the training
feature_extractor_layer.trainable = False
# %%
"""
Define our model. It has the feature extactor as the first layer. It then has a dense layer consisting of 1024 neurons.
The size of the output layer is then equal to N_LABELS which is the number of types.
"""

model = tf.keras.Sequential(
    [
        feature_extractor_layer,
        layers.Dense(1024, activation="relu"),
        #        layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l1(0.001)),
        #        layers.Dropout(0.5),
        layers.Dense(N_LABELS, activation="sigmoid", name="output"),
    ]
)
# %%
"""
Define learning rate and epochs and then compile and train the model
"""
LR = 1e-4
EPOCHS = 80

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss = "binary_crossentropy"
metrics = ["accuracy"]

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=create_dataset(X_val, y_val_bin),
    verbose=1,
    callbacks=[early_stop],
)

# %%
"""
Save weights and biases after training
"""
model.save_weights("model_weights_no_dropout_no_regularization.h5")


# %%
# %%
def show_prediction(name, model):
    # Get each Pokemon's name and type from the generation 6 games
    pokeId = poke_df.loc[poke_df["Name"] == name]["No"]
    types = poke_df.loc[poke_df["Name"] == name]["Types"]
    # Set the path to the folder that contains images
    folder_path = os.path.join(
        "C:/Users/elias/OneDrive/Neural-Final/pokemon_7000", name
    )

    # Get a list of all the image files in the folder
    image_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".png")
    ]

    # Select a random image file from the list
    img_path = random.choice(image_files)
    print(img_path)
    # Read and prepare image
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(IMG_SIZE, IMG_SIZE, CHANNELS)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)

    # Generate prediction
    prediction = (model.predict(img) > 0.5).astype("int")
    prediction = pd.Series(prediction[0])
    prediction.index = mlb.classes_
    prediction = prediction[prediction == 1].index.values

    # Predict each Pokemon's type
    print("\n\n{}\nType\n{}\n\nPrediction\n{}\n".format(name, types, list(prediction)))


# %%
pokemon = ["Horsea", "Poliwrath", "Machamp"]

for poke in pokemon:
    print(poke)
    show_prediction(poke, model)
# %%
# %%
