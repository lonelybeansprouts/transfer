import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, add, Activation, Flatten, AveragePooling2D, MaxPooling2D, Dropout, Lambda, UpSampling2D, Concatenate, GlobalAveragePooling2D
from keras.layers.merge import Add
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.regularizers import l2
from keras import optimizers
from keras.models import Model,Sequential
from utills import load_cifar
from keras.initializers import he_normal
from keras.layers import Conv2D
from keras import backend as K 
#from operations import Convolution2D as Conv2D
import numpy
# from dataloader.mnist_dataloader import read_mnist
from keras import backend as K
from utills import load_weight_by_weight_name
from keras import regularizers 
from operation_1 import Convolution2D_pad, Convolution2D_test






DEPTH              = 28
WIDE               = 10
IN_FILTERS         = 16

CLASS_NUM          = 10
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 1

BATCH_SIZE         = 128
EPOCHS             = 200
ITERATIONS         = 50000 // BATCH_SIZE + 1
weight_decay       =  0.0005
LOG_FILE_PATH      = './w_resnet/'
dropout            =0.5





##########################
# source model
##########################
def residual_network(img_input,classes_num=10,stack_n=8):
    
    def residual_block(x,o_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
        
    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)



    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

input = Input(shape=(32,32,1))
output = residual_network(img_input=input,classes_num=10,stack_n=8)
source_model = Model(input,output)
print(source_model.summary())


for layer in source_model.layers:
    layer.trainable = False


conv_layers = [layer for layer in source_model.layers]

conv_layer_16 = []
conv_layer_32 = []
conv_layer_64 = []
for layer in conv_layers:
    print(layer.output_shape)
    print(layer.name)
    if "conv2d" in layer.name:
        if layer.output_shape[-1]==16:
            conv_layer_16.append(layer)
        if layer.output_shape[-1]==32:
            conv_layer_32.append(layer)
        if layer.output_shape[-1]==64:
            conv_layer_64.append(layer)
num_layer_16 = len(conv_layer_16)
num_layer_32 = len(conv_layer_32)
num_layer_64 = len(conv_layer_64)
out_block_1 = conv_layer_16[0:num_layer_16//2]
out_block_2 = conv_layer_16[num_layer_16//2:num_layer_16]
out_block_3 = conv_layer_32[0:num_layer_32//2]
out_block_4 = conv_layer_32[num_layer_32//2:num_layer_32]
out_block_5 = conv_layer_64[0:num_layer_64//2]
out_block_6 = conv_layer_64[num_layer_64//2:num_layer_64]


out_block_1 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_1]
out_block_2 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_2]
out_block_3 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_3]
out_block_4 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_4]
out_block_5 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_5]
out_block_6 = [Conv2D(7,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(layer.output) for layer in out_block_6]

out_block_1 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_1))
out_block_2 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_2))
out_block_3 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_3))
out_block_4 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_4))
out_block_5 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_5))
out_block_6 = Conv2D(1,kernel_size=(3,3),strides=(1,1), padding='same',
                        kernel_initializer="he_normal", activation = 'tanh',
                        kernel_regularizer=regularizers.l2(weight_decay))(add(out_block_6))

#source_features = [out_block_1, out_block_2, out_block_3, out_block_4, out_block_5, out_block_6]
source_features = None


# ####################################################
# #   target  model
# ####################################################
def conv_extend( filters, kernel_size, input, source_features):
    output=None
    if source_features:
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
        output = Convolution2D_test(filters=filters, kernel_size=kernel_size, num_source=num_source, 
                                  padding='same', kernel_initializer=he_normal(), activation = 'relu')(input)
    else:
        output = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                         kernel_initializer=he_normal(), activation='relu')(input)
    return output



out = conv_extend(filters=32, kernel_size=(3,3), input=input, source_features=source_features)
#out = Conv2D(filters=32, kernel_size=(3,3), padding='same',
#                                kernel_initializer=he_normal(), activation='relu')(input)
out = MaxPooling2D((2, 2), strides=(2, 2))(out)
out = conv_extend(filters=64, kernel_size=(3,3), input=out, source_features=source_features)
out = MaxPooling2D((2, 2), strides=(2, 2))(out)
out = Flatten()(out)
out = Dense(1024, activation = 'relu', kernel_initializer='he_normal')(out)
out = Dense(1024, activation = 'relu', kernel_initializer='he_normal')(out)
out = Dropout(0.5)(out)
output = Dense(10, activation = 'softmax', kernel_initializer='he_normal')(out)
target_model = Model(input,output)
print(target_model.layers)



##################################
#loading data
###################################
dataset = numpy.load('../../data/usps/usps_15.npz')
#dataset = numpy.load('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz')
train_set_x = dataset['train_set_x']
train_set_y = dataset['train_set_y']
test_set_x = dataset['test_set_x']
test_set_y = dataset['test_set_y']
train_set_x = train_set_x.reshape(-1,32,32,1)
test_set_x = test_set_x.reshape(-1,32,32,1)

##################################
#载入源模型参数及验证参数
##################################
print('start loading weights')
load_weight_by_weight_name(source_model,'../../data/output/model_resnet_minist/ckpt.h5')
print("source model loading weight finished")

print("validate source model")
#sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=.1)
source_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
r=source_model.evaluate(test_set_x,test_set_y)
print("测试集结果：",r)
r=source_model.evaluate(train_set_x,train_set_y)
print("训练集结果：",r)
print("validate finished")



##################################
#训练目标模型
##################################
def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.001
    return 0.0001
print("xxxxxxx")
#print(source_model.layers[1].get_weights()[0])
print("xxxxxxx")
#print(Conv2D_test.get_weights()[0])




#     # set callback
tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
#cbks = [change_lr,tb_cb]
cbks = [change_lr]
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
#adam = optimizers.Adam(lr=.1)
target_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print("model structure:")
print(target_model.summary())


# start train
target_model.fit(train_set_x, train_set_y,
                 batch_size=15,
                 epochs=200,
                 callbacks=cbks,
                 validation_data=(test_set_x, test_set_y),
                 shuffle=True)


print("xxxxxxx")
#print(source_model.layers[1].get_weights()[0])
print("xxxxxxx")
#print(Conv2D_test.get_weights()[0])








# out1 = Convolution2D_pad(32, (5, 5), padding='same',  \
#         kernel_initializer=he_normal(), activation = 'relu', source_features=source_features)(model.layers[0].input)
# out2 = MaxPooling2D((2, 2), strides=(2, 2))(out1)
# out3 = Convolution2D_pad(64, (5, 5), padding='same',  \
#         kernel_initializer=he_normal(), activation = 'relu', source_features=source_features)(out2)
# out4 = MaxPooling2D((2, 2), strides=(2, 2))(out3)
# out5 = Flatten()(out4)
# out6 = Dense(120, activation = 'relu', kernel_initializer='he_normal')(out5)
# out7 = Dense(84, activation = 'relu', kernel_initializer='he_normal')(out6)
# dropout = Dropout(0.5)(out7)
# out8 = Dense(10, activation = 'softmax', kernel_initializer='he_normal')(dropout)
# lenet = Model(model.layers[0].input,out8)




# print('start loading weights')
# # source_model.load_weights("C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5",by_name=True)
# #load_weight_by_weight_name(source_model,"C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5")
# # load_weight_by_weight_name(model,'../../data/output/model_vgg19_minist/model_at_epoch_190.h5')
# load_weight_by_weight_name(source_model,'C:\\Users\\beansprouts\\Desktop\\transfer\\data\\model_at_epoch_190.h5')
# print("source model loading weight finished")
