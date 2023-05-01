# %%
import tensorflow as tf
import os
from keras import backend as K
from tensorflow.python.client import device_lib
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from skimage.io import imread
from skimage.transform import resize
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import random
import pandas as pd
from time import time

# %%
# Define classes that will be used to train model
poke_df = pd.read_csv("pokemon.csv")
# %%
poke_df = poke_df.rename(columns={"#": "No"})
poke_df = poke_df.drop_duplicates(subset=["No"])
poke_df = poke_df[poke_df["Generation"] == 1]
poke_df = poke_df.loc[:, ["No", "Name"]]
# %%
classes = poke_df["Name"].to_list()
# %%
print(classes)
# %%
# Set paths
current_dir = os.getcwd()
# %%
print(current_dir)
# %%
path_to_train_set = os.path.join(current_dir, "train_gen1")
path_to_test_set = os.path.join(current_dir, "test_gen1")
path_to_sample = os.path.join(current_dir, "train_gen1", "pikachu", "4492.png")

# %%
# Helper Functions


def plot_training_set_images(class_to_plot, path_to_train_set):
    """
    This function plots all the JPG images in the specified path
    """
    images = []
    for img_path in glob.glob(f"{path_to_train_set}/{class_to_plot}/*.jpg"):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)


def plot_image_with_pixel_values(img, ax):
    """
    This function displays an image and the grayscale for each pixel
    """
    ax.imshow(img, cmap="gray")
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(round(img[x][y], 2)),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if img[x][y] < thresh else "black",
            )


def plot_image(path_to_image):
    """
    This function displays an image resized to the height and width we will use for the model
    And also displays the pixel grayscale in a separate image
    """
    # Load a color image in grayscale
    image = imread(path_to_image, as_gray=True)
    image = resize(image, (28, 28), mode="reflect")
    print("This image is: ", type(image), "with dimensions:", image.shape)
    plt.imshow(image, cmap="gray")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plot_image_with_pixel_values(image, ax)


def plot_model_learning(history):
    """
    This function generates the plots to view the learning of the model through the epochs
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


def predict_new_image(classifier, classes):
    """
    This function takes a random image from the test set and predicts its class using our trained classifier
    """

    # Randomly choose from one of our classes
    random_class = random.choice(classes)

    # Select a random image from the class selected above
    random_test_image = random.choice(
        os.listdir(f"{path_to_test_set}/{random_class}/")
    )  # change dir name to whatever

    # Build the path for the random image
    img_path = f"{path_to_test_set}/{random_class}/{random_test_image}"

    # Load the image
    img = image.load_img(img_path, target_size=(28, 28))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    # Plot the image
    plt.imshow(img_tensor[0])
    plt.show()

    # Predict the class of the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    # Convert the predicted class from an integer into the named class:
    if np.argmax(classifier.predict(images), axis=-1)[0] == 0:
        prediction = classes[0]
    else:
        if np.argmax(classifier.predict(images), axis=-1)[0] == 1:
            prediction = classes[1]
        else:
            prediction = classes[2]

    print(f"Predicted class is {prediction}\nTrue class is {random_class}")

    print(f"Probability vector: {classifier.predict_proba(images)}")


# %%
# initialize the CNN
classifier = Sequential()
# %%
# Adding a first convolutional layer
classifier.add(
    Conv2D(32, (3, 3), padding="valid", input_shape=(28, 28, 3), activation="relu")
)
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units=128, activation="relu"))


classifier.add(Dropout(0.5))
classifier.add(Dense(units=len(classes), activation="softmax", name="preds"))
# %%
# Compiling the CNN
classifier.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
# %%
# Transform the dataset into the required shape
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    path_to_train_set, target_size=(28, 28), batch_size=16, class_mode="categorical"
)
test_set = test_datagen.flow_from_directory(
    path_to_test_set, target_size=(28, 28), batch_size=16, class_mode="categorical"
)
# %%
checkpointer = ModelCheckpoint(
    filepath="best_weights.hdf5", monitor="val_accuracy", verbose=1, save_best_only=True
)
# %%
history = classifier.fit(
    training_set,
    steps_per_epoch=10,
    epochs=100,
    callbacks=[checkpointer],
    verbose=1,
    validation_data=test_set,
)

# %%
model = tf.keras.models.load_model("best_weights.hdf5")

# %%
# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_set)

# Print the results
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
# %%
