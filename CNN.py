# -*- coding: utf-8 -*-
"""
Author: Ari Bormanis
Purpose: Creating and training the CNN.
"""

import tensorflow as tf
import keras.layers as layers
import numpy as np
import keras

from decoderHead import derivLayer
    

# first we define the PCNN
x_dim = 322
y_dim = 162

input_img = keras.Input(shape=(y_dim, x_dim, 2))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# now we decode

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(2, (3, 3), activation='linear', padding='valid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='MSE')
autoencoder.summary()

# now we train the model
train_data_in = np.load('train_data_in.npy')
train_data_out = np.load('train_data_out.npy')

autoencoder.fit(train_data_in, train_data_out, epochs=20, batch_size=8)
autoencoder.save('CNN_batch_8_20epochs_learn1e5')
