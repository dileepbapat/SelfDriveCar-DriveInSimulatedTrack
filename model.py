import csv
import cv2
import keras
import sklearn
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D

import numpy as np
import os
from sklearn.utils import shuffle

lines = []
# 1. large set, mouse
# 2. bit smaller set using mouse
# 3  large, keyboard
# 4. even large mouse
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

def generator(samples, batch_size=32):
    correction = .9
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(file_path(batch_sample[0]))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                img_left = cv2.imread(file_path(batch_sample[1]))
                images.append(img_left)
                angles.append(center_angle + correction) #left

                img_right = cv2.imread(file_path(batch_sample[2]))
                images.append(img_right)
                angles.append(center_angle - correction)  # right

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


batch_size = 100
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


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

# fit model using generator, dataset may not fit in memory otherwise.
model.fit_generator(train_generator, len(train_samples)/batch_size,
                    validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
                    epochs=22) #, initial_epoch=10)

model.save("model.h5")