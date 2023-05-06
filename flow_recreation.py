# -*- coding: utf-8 -*-
"""
Author: Ari Bormanis
Purpose: Using the various models to recreate the flow in the test dataset.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio


# loading models
test_size = 1025

test_data_in = np.load('test_data_in.npy')
test_data_out = np.load('test_data_out.npy')


pcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_20epochs_learn1e5')
longPcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_100epochs_learn1e5')
cnn = tf.keras.models.load_model('final_CNN_test_batch_8_20epochs_learn1e5')

x = np.linspace(0,2,322)
y = np.linspace(0,1,162)
X, Y = np.meshgrid(x,y)

# using models to make predictions and saving the predictions as pngs
for i in range(0,1000,50):
        u = test_data_out[i,:,:,0]
        v = test_data_out[i,:,:,1]
        plt.title("Truth at time"+str(i))
        plt.quiver(X,Y, u, v,color='blue')
        plt.savefig('truth/time'+str(i)+'.png',
                    transparent = False,
                    facecolor= 'white') 
        plt.close()

        y_in = test_data_in[i:i+1,:,:,:]
        y_pred = pcnn(y_in)
        u = y_pred[0,:,:,0]
        v = y_pred[0,:,:,1]
        plt.title("PCNN at time"+str(i))
        plt.quiver(X,Y, u, v,color='blue')
        plt.savefig('pcnn/time'+str(i)+'.png',
                    transparent = False,
                    facecolor= 'white') 
        plt.close()

        y_in = test_data_in[i:i+1,:,:,:]
        y_pred = longPcnn(y_in)
        u = y_pred[0,:,:,0]
        v = y_pred[0,:,:,1]
        plt.title("Long PCNN at time"+str(i))
        plt.quiver(X,Y, u, v,color='blue')
        plt.savefig('pcnn_long/time'+str(i)+'.png',
                    transparent = False,
                    facecolor= 'white') 
        plt.close()

        y_in = test_data_in[i:i+1,:,:,:]
        y_pred = cnn(y_in)
        u = y_pred[0,:,:,0]
        v = y_pred[0,:,:,1]
        plt.title("CNN at time"+str(i))
        plt.quiver(X,Y, u, v,color='blue')
        plt.savefig('cnn/time'+str(i)+'.png',
                    transparent = False,
                    facecolor= 'white') 
        plt.close()

# Creating gifs
frames_truth = []
frames_pcnn = []
frames_pcnn_long = []
frames_cnn = []

for i in range(0,1000,50):
    image_t = imageio.v2.imread('truth/time'+str(i)+'.png')
    image_p = imageio.v2.imread('pcnn/time'+str(i)+'.png')
    image_pl = imageio.v2.imread('pcnn_long/time'+str(i)+'.png')
    image_c = imageio.v2.imread('cnn/time'+str(i)+'.png')
    frames_truth.append(image_t)
    frames_pcnn.append(image_p)
    frames_pcnn_long.append(image_pl)
    frames_cnn.append(image_c)

imageio.mimsave('truth.gif',frames_truth,duration=100)
imageio.mimsave('pcnn.gif',frames_pcnn,duration=100)
imageio.mimsave('pcnn_long.gif',frames_pcnn_long,duration=100)
imageio.mimsave('cnn.gif',frames_cnn,duration=100)

        