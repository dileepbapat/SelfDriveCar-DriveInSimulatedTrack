import csv
import cv2
import keras
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D

import numpy as np
import os

lines = []
# 1. large set, mouse
# 2. bit smaller set using mouse
# 3  large, keyboard
# 4. even large with mouse
names = ["data", "data-recovery"]
basepath  = "../carnd/" # path to training data organized in sub folders for drive data, recovery etc.

for name in names:
    with open(basepath + name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

images = []
measurements = []

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def file_path(source_path):
    filename = "/".join(source_path.split("/")[-3:])
    path = basepath + filename
    return path

if os.path.exists("X_train.npy"):
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
else:
    correction = .15

    for line in lines:
        if line[0] == "center":
            continue
        image = cv2.imread(file_path(line[0]))
        images.append(image)
        steering_center = float(line[3])
        measurements.append(steering_center)

        img_left = cv2.imread(file_path(line[1]))
        images.append(img_left)
        measurements.append(steering_center + correction) #left

        images.append(np.fliplr(img_left))
        measurements.append(-(steering_center + correction)) #left

        img_right = cv2.imread(file_path(line[2]))
        images.append(img_right)
        measurements.append(steering_center - correction) #right

        images.append(np.fliplr(img_right))
        measurements.append( - (steering_center - correction))  # right

        images.append(np.fliplr(image))
        measurements.append(-steering_center)

    X_train = np.array(images)
    y_train = np.array(measurements)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)


model = Sequential([
     Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)),
     Cropping2D(((70, 25), (0,0))),
     Conv2D(8, (5,5)),
     MaxPooling2D((2,4)),
     Conv2D(16,  (3,3)),
     MaxPooling2D((2,4)),
     Conv2D(24, (3,3)),
     Flatten(),
     Dropout(.4),
     Dense(108, activation='elu'),
     Dense(21, activation='elu'),
     Dense(7, activation='elu'),
     Dense(1)
 ])

# Using adam optimizer so that learning rate will be controlled by it, applying a small decay.

adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
model.compile(loss='mse', optimizer=adam)

model.fit(X_train, y_train, epochs=5, validation_split=.2, shuffle=True)

model.save("model_small.h5")