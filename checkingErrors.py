# -*- coding: utf-8 -*-
"""
Author: Ari Bormanis
Purpose: Here we check the value of our errors in terms of the MSE on the test set.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Loading models
test_size = 1025

test_data_in = np.load('test_data_in.npy')
test_data_out = np.load('test_data_out.npy')


pcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_20epochs_learn1e5')
longPcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_100epochs_learn1e5')
cnn = tf.keras.models.load_model('final_CNN_test_batch_8_20epochs_learn1e5')

# Computing the mean MSE and plotting MSE as a function of time
mse = tf.keras.losses.MeanSquaredError()
x = np.arange(test_size)
MSE_vals_PCCN = np.zeros(test_size)
MSE_vals_PCCN_long = np.zeros(test_size)
MSE_vals_CCN = np.zeros(test_size)
avg_valPCNN = 0
avg_valPCNN_long = 0
avg_valCNN = 0
for i in range(test_size):
    y_in = test_data_in[i:i+1,:,:,:]
    y_out = test_data_out[i:i+1,:,:,:]

    yPCNN = pcnn(y_in)
    yPCNN_long = longPcnn(y_in)
    yCNN = cnn(y_in)

    errPCNN = mse(yPCNN, y_out).numpy()
    errPCNN_long = mse(yPCNN_long, y_out).numpy()
    errCNN = mse(yCNN, y_out).numpy()

    avg_valPCNN += errPCNN 
    avg_valPCNN_long += errPCNN_long
    avg_valCNN += errCNN

    MSE_vals_PCCN[i] = errPCNN
    MSE_vals_PCCN_long[i] = errPCNN_long
    MSE_vals_CCN[i] = errCNN

print()
print('Avg MSE PCNN:', avg_valPCNN / test_size)
print('Avg MSE PCNN long:', avg_valPCNN_long / test_size)
print('Avg MSE CNN:', avg_valCNN / test_size)

plt.plot(x,MSE_vals_PCCN,label='PCNN')
plt.plot(x,MSE_vals_PCCN_long,label='PCNN Long')
plt.plot(x,MSE_vals_CCN,label='CNN')
plt.xlabel('time step')
plt.ylabel('MSE')
plt.legend()
plt.show()