# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:06:49 2024

@author: jfodopsokou
"""

import tensorflow as tf
import numpy as np

# Possible enhancements:
    #After the normalization the last sample is lost. It is not a big deal, but it could be better to mantain
    #length
class SegmentNormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, freq, freq_limits, normalize_index, **kwargs):
        super(SegmentNormalizeLayer, self).__init__(**kwargs)
        # Convert freq to a TensorFlow constant tensor
        self.freq = tf.constant(freq, dtype=tf.float32)
        # Convert freq_limits to a TensorFlow constant tensor
        self.freq_limits = tf.constant(freq_limits, dtype=tf.float32)
        self.normalize_index = normalize_index

    def call(self, inputs):
        n_segments = len(self.freq_limits) - 1
        outputs = []
        for i in range(n_segments):
            # Find start and end indices using TensorFlow operations
            start = tf.where(tf.equal(self.freq, self.freq_limits[i]))[0][0]
            end = tf.where(tf.equal(self.freq, self.freq_limits[i+1]))[0][0]
            segment = inputs[:, start:end, :]

            # Apply normalization only on the specified index in the last dimension
            normalized_segment = tf.concat([
                (segment[:, :, j:j+1] - tf.reduce_min(segment[:, :, j:j+1], axis=1, keepdims=True)) /
                (tf.reduce_max(segment[:, :, j:j+1], axis=1, keepdims=True) - tf.reduce_min(segment[:, :, j:j+1], axis=1, keepdims=True))
                if j == self.normalize_index else segment[:, :, j:j+1]
                for j in range(segment.shape[-1])
            ], axis=-1)

            outputs.append(normalized_segment)
            
        output = tf.concat(outputs, axis=1)
        output.set_shape([inputs.shape[0], inputs.shape[1] - 1, inputs.shape[2]]) #the dynamic process is complex.
                        #The output shape will not be define until the freq and freq_limits are actually explore through
                        #call method. It is thus necessary to specified explicitly the outshape (because we can infered it)
        return output

    def compute_output_shape(self, input_shape):            
        # Assuming the input shape is preserved except for the segmentation
        return (input_shape[0], input_shape[1] - 1, input_shape[2])

    def get_config(self):
        config = super(SegmentNormalizeLayer, self).get_config()
        config.update({
            'freq': self.freq.numpy(),#.tolist(),  # Convert tensor to list for JSON serialization
            'freq_limits': self.freq_limits.numpy(),#.tolist(),
            'normalize_index': self.normalize_index
        })
        return config
    
class SelectiveDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, freq, freq_limits, **kwargs):
        super(SelectiveDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.freq = np.array(freq).reshape(-1, 1)
        self.freq_limits = freq_limits
        self.positions = np.abs(self.freq - self.freq_limits).argmin(axis=0)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        
        # Calculate proportionality for selective connection
        n = input_shape[-1]
        m = len(self.freq)
        start = int((self.positions[0] / m) * n)
        end = int((self.positions[1] / m) * n)
        
        # Calculate indices using TensorFlow to ensure they are tensors
        start = tf.cast(tf.math.round((self.positions[0] / m) * n), tf.int32)
        end = tf.cast(tf.math.round((self.positions[1] / m) * n), tf.int32)

        # Create a mask using TensorFlow operations
        self.mask = tf.Variable(initial_value=tf.zeros((n, self.units)), trainable=False, dtype=tf.float32)
        indices = tf.range(n, dtype=tf.int32)
        mask_values = tf.where((indices >= start) & (indices < end), 1.0, 0.0)
        self.mask.assign(tf.tile(tf.expand_dims(mask_values, axis=-1), [1, self.units]))
        
    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        #return masked_kernel
        return tf.matmul(inputs, masked_kernel) + self.bias    
    
    def compute_output_shape(self):
        return self.units

    def get_config(self):
        config = super(SelectiveDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'freq': self.freq.tolist(),
            'freq_limits': self.freq_limits.tolist(),
        })
        return config
    
#Selective layer is just a selective layer. Without any trainable weight. 
class SelectiveLayer(tf.keras.layers.Layer):
    def __init__(self, freq, freq_limits, **kwargs):
        super(SelectiveLayer, self).__init__(**kwargs)
        self.freq = np.array(freq).reshape(-1, 1)
        self.freq_limits = freq_limits
        self.positions = np.abs(self.freq - self.freq_limits).argmin(axis=0)
        
    def build(self, input_shape):
        super(SelectiveLayer, self).build(input_shape)
        
    def call(self, inputs):
        n = inputs.shape[-1]
        m = len(self.freq)
        
        # Calculate indices using TensorFlow to ensure they are tensors
        start = tf.cast(tf.math.round((self.positions[0] / m) * n), tf.int32)
        end = tf.cast(tf.math.round((self.positions[1] / m) * n), tf.int32)

        # Create a mask using TensorFlow operations
        indices = tf.range(n, dtype=tf.int32)
        mask = (indices >= start) & (indices < end)
        
        output = tf.boolean_mask(inputs, mask, axis=1)
        #return masked_kernel
        
        output.set_shape([inputs.shape[0], output.shape[1]])
        
        return output    
    
    def compute_output_shape(self, input_shape):
        m = len(self.freq)
        n = input_shape[-1]
        start = int((self.positions[0] / m) * n)
        end = int((self.positions[1] / m) * n)
        length = tf.cast(tf.reduce_sum(end - start), tf.int32)
        
        return (input_shape[0], length.numpy())

    def get_config(self):
        config = super(SelectiveDenseLayer, self).get_config()
        config.update({
            'freq': self.freq.tolist(),
            'freq_limits': self.freq_limits.tolist(),
        })
        return config