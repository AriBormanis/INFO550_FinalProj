# -*- coding: utf-8 -*-
"""
Author: Ari Bormanis
Purpose: This is what allows us to encode taking the derivative into our model. I.e.
we take derivatives via a second order central differnece scheme using convolution.
"""

import tensorflow as tf
import keras.layers as layers
import numpy as np

class derivLayer(layers.Conv2D):
    def __init__(self, delta_x, delta_y, **kwargs):
        super(derivLayer,self).__init__(filters=2,kernel_size=(3,3),dtype='float64',**kwargs)
        ker1 = np.array([[0., 1/(2*delta_y),0.],
                          [0., 0., 0.],
                          [0., 1/(-2*delta_y), 0.]],dtype='float64')
        ker1 = ker1.reshape(3,3,1,1)
        ker2 = np.array([[0., 0., 0.],
                         [1/(2*delta_x), 0.,1/(-2*delta_x)],
                         [0., 0. ,0.]],dtype='float64')
        ker2 = ker2.reshape(3,3,1,1)
        ker = np.concatenate([ker1,ker2],axis=-1)
        self.derivKer = tf.constant(ker)
    
    def build(self, input_shape):
        self.kernel = self.derivKer
        self.built = True
        
    def call(self, inputs):
        result = self.convolution_op(inputs, self.kernel)
        return result
        
        
