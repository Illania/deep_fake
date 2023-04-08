import numpy as np
import cv2
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


def create_dataset(path):
    images = []
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                print(filename)
                image = cv2.imread(os.path.join(directory, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype('float32')
                image /= 255.0
                images.append(image)
            except Exception as e:
                print(f"There was an error parsing the file {filename}:{str(e)}")
    images = np.array(images)
    return images


def unzip_file(path_to_zip_file):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall("./")


# unzip_file("./results_dst.zip")
faces_2 = create_dataset('./results_dst/')
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(faces_2, faces_2, test_size=0.20, random_state=0)

input_img = layers.Input(shape=(120, 120, 3))
x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(1024, kernel_size=5, strides=2, padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(9216)(x)
encoded = layers.Reshape((3, 3, 1024))(x)
encoder = keras.Model(input_img, encoded, name="encoder")

decoder_input = layers.Input(shape=(3, 3, 1024))
x = layers.Conv2D(1024, kernel_size=5, strides=2, padding='same', activation='relu')(decoder_input)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(np.prod((120, 120, 3)))(x)
decoded = layers.Reshape((120, 120, 3))(x)
decoder = keras.Model(decoder_input, decoded, name="decoder")

auto_input = layers.Input(shape=(120, 120, 3))
encoded = encoder(auto_input)
decoded = decoder(encoded)
autoencoder = keras.Model(auto_input, decoded, name="autoencoder")
autoencoder.compile(optimizer=keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mae')
autoencoder.summary()

checkpoint = ModelCheckpoint("./autoencoder.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto',
                             save_freq=1)
history = autoencoder.fit(X_train_b, X_train_b, epochs=5, batch_size=512, shuffle=True,
                          validation_data=(X_test_b, X_test_b), callbacks=[checkpoint])
