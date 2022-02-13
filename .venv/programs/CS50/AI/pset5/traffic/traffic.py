import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    # Initialize return lists
    images = []
    labels = []

    print('Loading Data...')
    for folder_index in range(0, NUM_CATEGORIES):
        
        # Go to folder and get files
        folder_path = os.path.join(data_dir, str(folder_index))
        files = os.listdir(folder_path)
        
        # Get individual file and convert to np array
        for file in files:
            img = np.array(cv2.imread(os.path.join(folder_path, file)))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Append to return lists
            images.append(img)
            labels.append(folder_index)

        # Print status
        print(f'Loaded set {folder_index}')

    # Return
    print('Loading complete')
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    model = tf.keras.models.Sequential()
    
    # First set of Conv/pooling
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))

    # Dropout
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Second set of Conv/pooling
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))

    # Dropout
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Third set of Conv/pooling
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))

    # Dropout
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Hidden layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    # Dropout
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Output layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Additional model settings
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    main()
