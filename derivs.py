# -*- coding: utf-8 -*-
"""
Author: Ari Bormanis
Purpose: Determining how well the models respect the zero divergence condition 
"""

import tensorflow as tf
import numpy as np
import keras.layers as layers
import matplotlib.pyplot as plt


# Loading models
test_size = 1025

test_data_in = np.load('test_data_in.npy')
test_data_out = np.load('test_data_out.npy')


pcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_20epochs_learn1e5')
longPcnn = tf.keras.models.load_model('final_PCNN_test_batch_8_100epochs_learn1e5')
cnn = tf.keras.models.load_model('final_CNN_test_batch_8_20epochs_learn1e5')


# Just defining fancy ways of taking derivative via convolution
class xdir(layers.Conv2D):
    def __init__(self, delta_x, **kwargs):
        super(xdir,self).__init__(filters=1,kernel_size=(3,3),dtype='float64',**kwargs)
        ker = np.array([[0., 0., 0.],
                         [1/(-2*delta_x), 0.,1/(2*delta_x)],
                         [0., 0. ,0.]],dtype='float64')
        ker = ker.reshape(3,3,1,1)
        self.derivKer = tf.constant(ker)
    
    def build(self, input_shape):
        self.kernel = self.derivKer
        self.built = True
        
    def call(self, inputs):
        result = self.convolution_op(inputs, self.kernel)
        return result
    
class ydir(layers.Conv2D):
    def __init__(self, delta_y, **kwargs):
        super(ydir,self).__init__(filters=1,kernel_size=(3,3),dtype='float64',**kwargs)
        ker = np.array([[0., 1/(2*delta_y), 0.],
                         [0, 0.,0],
                         [0., 1/(-2*delta_y) ,0.]],dtype='float64')
        ker = ker.reshape(3,3,1,1)
        self.derivKer = tf.constant(ker)
    
    def build(self, input_shape):
        self.kernel = self.derivKer
        self.built = True
        
    def call(self, inputs):
        result = self.convolution_op(inputs, self.kernel)
        return result
    

# now we calculate the gradient at interior points
x = np.arange(test_size)
delta_x = 3e-3
delta_y = 3e-3

g_vals_PCCN = np.zeros(test_size)
g_vals_PCCN_long = np.zeros(test_size)
g_vals_CCN = np.zeros(test_size)

sample_size = 320*160
for i in range(test_size):
    y_in = test_data_in[i:i+1,:,:,:]

    yPCNN = pcnn(y_in)
    yPCNN_long = longPcnn(y_in)
    yCNN = cnn(y_in)

    x_p = xdir(delta_x)(tf.reshape(yPCNN[:,:,:,0], (1,162,322,1)))
    y_p = ydir(delta_y)(tf.reshape(yPCNN[:,:,:,1], (1,162,322,1)))
    x_pl = xdir(delta_x)(tf.reshape(yPCNN_long[:,:,:,0], (1,162,322,1)))
    y_pl = ydir(delta_y)(tf.reshape(yPCNN_long[:,:,:,1], (1,162,322,1)))
    x_c = xdir(delta_x)(tf.reshape(yCNN[:,:,:,0], (1,162,322,1)))
    y_c = ydir(delta_y)(tf.reshape(yCNN[:,:,:,1], (1,162,322,1)))


    pcnn_div = tf.math.reduce_sum(x_p+y_p)
    pcnn_long_div = tf.math.reduce_sum(x_pl+y_pl)
    cnn_div = tf.math.reduce_sum(x_c+y_c)

    g_vals_PCCN[i] = pcnn_div / sample_size
    g_vals_PCCN_long[i] = pcnn_long_div / sample_size
    g_vals_CCN[i] = cnn_div / sample_size

plt.plot(x,g_vals_PCCN,label='PCNN')
plt.plot(x,g_vals_PCCN_long,label='PCNN Long')
plt.plot(x,g_vals_CCN,label='CNN')
plt.xlabel('time step')
plt.ylabel('sum of divergences')
plt.legend()
plt.show()