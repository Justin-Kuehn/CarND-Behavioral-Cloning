import csv
import os
import json
import argparse
import cv2
import numpy as np
from scipy.ndimage import imread
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.convolutional import Convolution2D

# Hyperparams
EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
LEFT_STEERING_OFFSET = 0.15
RIGHT_STEERING_OFFSET = -0.15


def parse_driving_log(data_dir):
    """Parse the driving log and return pairs of images and steering angles"""
    data = []
    with open(os.path.join(data_dir, "driving_log.csv"), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip header row
        for row in reader:
            center_path = os.path.join(data_dir, row[0].strip())
            left_path = os.path.join(data_dir, row[1].strip())
            right_path = os.path.join(data_dir, row[2].strip())
            angle = float(row[3])
            data.append([center_path, angle])
            data.append([left_path, angle + LEFT_STEERING_OFFSET])
            data.append([right_path, angle + RIGHT_STEERING_OFFSET])
    return data


def load_images(image_paths):
    """load images from a list"""
    return np.asarray([imread(p) for p in image_paths])


def crop_images(images):
    """Crop 60 pixels from top and 20 from bottom"""
    return images[:,60:140]


def adjust_gamma(image, gamma=1.0):
    """Adjust image gamma"""
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def fix_contrast(images):
    """Boost contrast to help with shadows"""
    equalized = np.zeros_like(images)
    for i, img in enumerate(images):
        equalized[i] = adjust_gamma(img, 1.5)
    return equalized


def resize_images(images, shape=(64, 64)):
    """Resize a list of images to the given shape"""
    height, width = shape
    resized = np.zeros((len(images), height, width, 3))
    for i, img in enumerate(images):
        resized[i] = cv2.resize(img, (height, width))
    return resized


def preprocess_images(img):
    """Preprocess an image for use in the steering model"""
    img = fix_contrast(img)
    img = crop_images(img)
    img = resize_images(img)
    img = img / 127.5 - 1
    return img


def random_shadow(image, alpha=0.75):
    """Simulate shadows by shading out a region of the image"""
    height, width, _ = image.shape
    x1, x2 = np.random.randint(25, width - 25, 2)
    y1, y2 = np.random.randint(25, height - 25, 2)
    choice = np.random.choice(range(4))
    if choice == 0:
        pts = np.array([[x1, 0], [width, 0], [width, height], [x2, height]])
    elif choice == 1:
        pts = np.array([[x1, 0], [0, y1], [0, 0]])
    elif choice == 2:
        pts = np.array([[0, y1], [width, y2], [width, height], [0, height]])
    else:
        pts = np.array([[x1, height], [width, y1], [width, height]])
    mask = image.copy()
    cv2.fillPoly(mask, [pts], (0, 0, 0))
    cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)
    return image


def randomize(img, angle):
    """Randomly flip images and angles horizontally"""
    if np.random.random() > 0.1:
        img = random_shadow(img)
    if np.random.random() > 0.5:
        img, angle = np.fliplr(img), -1 * angle

    return img, angle


def gen(images, angles):
    """
    create a generator that randomly samples from the images, angles
    :param images:
    :param angles:
    :return:
    """
    images = np.asarray(images)
    angles = np.asarray(angles)
    while 1:
        indices = np.random.choice(len(images), BATCH_SIZE)
        batch_images = load_images(images[indices])
        batch_angles = angles[indices]
        for i in range(len(batch_images)):
            batch_images[i], batch_angles[i] = randomize(batch_images[i], batch_angles[i])
        yield preprocess_images(batch_images), batch_angles


def get_steering_model():
    """Return a CNN used to predict steering angle"""
    ch, row, col = 3, 64, 64
    model = Sequential([
        Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)),
        ELU(),
        Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"),
        ELU(),
        Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"),
        Flatten(),
        Dropout(.2),
        ELU(),
        Dense(512),
        Dropout(.5),
        ELU(),
        Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss="mse")
    return model


def main():
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--directory', type=str, required=True, help='Data directory')
    args = parser.parse_args()
    data_dir = os.path.abspath(args.directory)
    print("Loading training data...")
    data = parse_driving_log(data_dir)
    np.random.shuffle(data)

    split_idx = int(0.9 * len(data))
    train, test = data[:split_idx], data[split_idx:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Train the model
    model = get_steering_model()
    model.fit_generator(
        gen(X_train, y_train),
        samples_per_epoch=len(X_train),
        nb_epoch=EPOCHS,
        validation_data=gen(X_test, y_test),
        nb_val_samples=len(X_test)
    )

    if not os.path.exists("./steering_model"):
        os.makedirs("./steering_model")

    model.save_weights("./steering_model/steering_model.h5", True)
    with open('./steering_model/steering_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    print("saved Models")

if __name__ == "__main__":
    main()
