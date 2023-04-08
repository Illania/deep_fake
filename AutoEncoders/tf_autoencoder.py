# Install TensorFlow 2.0 by using the following command
# For CPU installation
# pip install -q tensorflow == 2.0
# For GPU installation (CUDA and CuDNN must be available)
# pip install -q tensorflow-gpu == 2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from AutoEncoders.autoencoder import AutoEncoder


class TfAutoencoder(tf.keras.Model, AutoEncoder):
    """Autoencoder for creating Deep fake videos."""

    def __init__(self,
                 n_dims=[200, 392, 784],
                 name='autoencoder',
                 **kwargs):
        super(TfAutoencoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.encoder = Encoder(n_dims[0])
        self.decoder = Decoder([n_dims[1], n_dims[2]])

    @tf.function
    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

    def compile(self, optimizer, loss):
        pass


class Encoder(tf.keras.layers.Layer):
    """Encodes an input image"""

    def __init__(self,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        input_image = layers.Input(shape=(120, 120, 3))
        self.encode_layers = self.get_layers(self, input_image)

    def get_layers(self, input_image):
        x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(input_image)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(1024, kernel_size=5, strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(9216)(x)
        return layers.Reshape((3, 3, 1024))(x)

    @tf.function
    def call(self, inputs):
        return self.encode_layers(inputs)


class Decoder(tf.keras.layers.Layer):
    """Decodes an image"""

    def __init__(self,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        decoder_input = layers.Input(shape=(3, 3, 1024))
        self.decode_layers = layers.get_layers(self, decoder_input)

    def get_layers(self, decoder_input):
        x = layers.Conv2D(1024, kernel_size=5, strides=2, padding='same', activation='relu')(decoder_input)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(np.prod((120, 120, 3)))(x)
        return layers.Reshape((120, 120, 3))(x)

    @tf.function
    def call(self, inputs):
        return self.decode_layers(inputs)
