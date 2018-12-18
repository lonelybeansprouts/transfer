import keras
import numpy as np
import matplotlib.pyplot as plt
#按顺序构成的模型
from keras.models import Sequential, Model
#Dense全连接层
from keras.layers import Dense, Activation, Dropout, Input, Conv2D, Concatenate, AveragePooling2D, UpSampling2D
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import L1L2 as WR
import keras.backend as K
from operation_1 import Convolution2D_pad, Convolution2D_test


def conv_extend(filters, kernel_size, input, source_features):
    input_shape = K.int_shape(input)
    bz,h,w,c = input_shape
    out_features = []
    for source_feature in source_features:
        h_s = K.int_shape(source_feature)[1]
        if (h<h_s):
            rate = h_s // h
            out_features.append(AveragePooling2D(pool_size=(rate,rate))(source_feature))
        elif (h>h_s):
            rate = h // h_s
            out_features.append(UpSampling2D(size=(rate,rate))(source_feature))
        else:
            out_features.append(source_feature)
    out_features.append(input)
    print('feature_shape:',K.int_shape(input))
    input = Concatenate(axis=-1)(out_features)
    print('feature_concat_shape:',K.int_shape(input))

    num_source = len(source_features)
    output = Convolution2D_test(filters=1, kernel_size=(3,3),num_source=num_source)(input)
    return output



input = Input(shape=(32,32,1))
out1 = Conv2D(1,(3,3),padding='same')(input)
out2 = Conv2D(1,(3,3),padding='same')(input)
out3 = Conv2D(1,(3,3),padding='same')(input)
out4 = Conv2D(1,(3,3),padding='same')(input)
out5 = Conv2D(1,(3,3),padding='same')(input)

inp = Concatenate()([input,out1,out2,out3,out4,out5])

source_features = [out1,out2,out3,out4,out5]

# output = Conv2D(1,(3,3))(inp)

# output = Convolution2D_pad(1,(3,3),source_features=source_features)(input)

output = conv_extend(filters=1, kernel_size=(3,3),input=input,source_features=source_features)



model = Model(input,output)

print(model.layers)
# print([layer.name for layer in model.layers])

# Adam()

# a = Conv2D(10,(3,3))

# a.get