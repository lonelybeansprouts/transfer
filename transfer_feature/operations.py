# -*- coding: utf-8 -*-
"""Convolutional layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras import backend as K
import tensorflow as tf
from math import sqrt
import numpy as np
from keras.initializers import RandomUniform
from keras.initializers import RandomNormal

from keras.layers import AveragePooling2D, UpSampling2D, Concatenate




############################################
## a convolution layer which implements factorized wieght decay with reparameterized trick   
###########################################


class Convolution2D_pad(Layer):
    """Abstract nD convolution layer (private, used as implementation base).
            
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
            
    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """
            
    def __init__(self, 
                 filters,
                 kernel_size,
                 strides=1,
                 rank=2,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 source_features=None,
                 **kwargs):
                    
        super(Convolution2D_pad, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format ='channels_last'  # conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.source_features = source_features
            
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        a,b,c,d = self.kernel_shape 
          
        if a!=b:
            raise ValueError('kernel width and depth are not equal')

         
        #self.kernel_initializer =  RandomUniform(minval=-2, maxval=2, seed=None)
        # self.kernel_initializer =  RandomNormal(0.0,1.0)

        #print('kkkk')
        # for x in self.source_features:
           # print(K.int_shape(x))

        
        
        if (self.source_features):
            bz,h,w,c = input_shape
            self.out_features = []
            for source_feature in self.source_features:
                h_s = K.int_shape(source_feature)[1]
                if (h<h_s):
                    rate = h_s // h
                    self.out_features.append(AveragePooling2D(pool_size=(rate,rate))(source_feature))
                elif (h>h_s):
                    rate = h // h_s
                    self.out_features.append(UpSampling2D(size=(rate,rate))(source_feature))
                else:
                    self.out_features.append(source_feature)
            
            #卷积核的输入维度增加
            num_source = len(self.source_features)
            self.kernel_shape = list(self.kernel_shape)
            self.kernel_shape[2] = self.kernel_shape[2]+num_source

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
                  
                   
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
            
    def call(self, inputs):     
        if (self.source_features):
            self.out_features.append(inputs)
            print(K.int_shape(inputs))
            inputs = Concatenate(axis=-1)(self.out_features)  #融合特征 
            print(K.int_shape(inputs))

        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
                    
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
            
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
            
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)
            
    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Convolution2D_pad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))













# from keras.initializers import RandomUniform
# import keras

# a = K.variable(np.ones(shape=[4,100,100,7]),dtype='float32')
# conv2d =Convolution2D(16, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.1), kernel_initializer='he_normal', name='block2_conv1')
# b =conv2d(a)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     r = sess.run(b)
#     print(r.shape)





