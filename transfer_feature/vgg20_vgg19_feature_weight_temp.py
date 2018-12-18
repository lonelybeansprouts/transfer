import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, add, Activation, Flatten, AveragePooling2D, MaxPooling2D, Dropout, Lambda
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


from keras import backend as K
from utills import load_weight_by_weight_name




input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))

# build model
out = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1')(input)
out = BatchNormalization(name='batch_normalization_1')(out)
out = Activation('relu')(out)
out = Conv2D(64, (3, 3),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2')(out)
out = BatchNormalization(name='batch_normalization_2')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)

# Block 2
out = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1')(out)
out = BatchNormalization(name='batch_normalization_3')(out)
out = Activation('relu')(out)
out = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2')(out)
out = BatchNormalization(name='batch_normalization_4')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)

# Block 3
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1')(out)
out = BatchNormalization(name='batch_normalization_5')(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2')(out)
out = BatchNormalization(name='batch_normalization_6')(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3')(out)
out = BatchNormalization(name='batch_normalization_7')(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4')(out)
out = BatchNormalization(name='batch_normalization_8')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

# Block 4
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1')(out)
out = BatchNormalization(name='batch_normalization_9')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2')(out)
out = BatchNormalization(name='batch_normalization_10')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3')(out)
out = BatchNormalization(name='batch_normalization_11')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4')(out)
out = BatchNormalization(name='batch_normalization_12')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(out)

# Block 5
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1')(out)
out = BatchNormalization(name='batch_normalization_13')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2')(out)
out = BatchNormalization(name='batch_normalization_14')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3')(out)
out = BatchNormalization(name='batch_normalization_15')(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4')(out)
out = BatchNormalization(name='batch_normalization_16')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(out)

# model modification for cifar-10
out = Flatten(name='flatten')(out)
out = Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10')(out)
out = BatchNormalization(name='batch_normalization_17')(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)
out = Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2')(out)
out = BatchNormalization(name='batch_normalization_18')(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)     
out = Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10')(out)      
out = BatchNormalization(name='batch_normalization_19')(out)
output = Activation('softmax')(out)
source_model = Model(input,output)
print('start loading weights')
# source_model.load_weights("C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5",by_name=True)
#load_weight_by_weight_name(source_model,"C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5")
# load_weight_by_weight_name(model,'../../data/output/model_vgg19_minist/model_at_epoch_190.h5')
load_weight_by_weight_name(source_model,'C:\\Users\\beansprouts\\Desktop\\transfer\\data\\model_at_epoch_190.h5')
print("source model loading weight finished")



#
for layer in source_model.layers:
    layer.trainable = False

#
conv_layers = []
for layer in source_model.layers:
    if "conv" in layer.name:
        conv_layers.append(layer)

conv1_1,conv1_2, conv2_1,conv2_2, conv3_1,conv3_2,conv3_3,conv3_4, conv4_1,conv4_2,conv4_3,conv4_4, \
                                                 conv5_1, conv5_2, conv5_3, conv5_4 = conv_layers

conv_down_layers = []
for i in range(len(conv_layers)):
    conv_down_layer = Conv2D(21, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv_layers[i].output)
    conv_down_layers.append(conv_down_layer)


conv1_1_down,conv1_2_down, conv2_1_down,conv2_2_down, conv3_1_down,conv3_2_down,conv3_3_down,conv3_4_down, \
        conv4_1_down,conv4_2_down,conv4_3_down,conv4_4_down,conv5_1_down, conv5_2_down, conv5_3_down, conv5_4_down = conv_down_layers



block1_out = Conv2D(1, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv1_1_down + conv1_2_down)
block2_out = Conv2D(1, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv2_1_down + conv2_2_down)
block3_out = Conv2D(1, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv3_1_down + conv3_2_down + conv3_3_down + conv3_4_down)
block4_out = Conv2D(1, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv4_1_down + conv4_2_down + conv4_3_down + conv4_4_down)
block5_out = Conv2D(1, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), \
                             kernel_initializer=he_normal())(conv5_1_down + conv5_2_down + conv5_3_down + conv5_4_down)

source_features = [block1_out, block2_out, block3_out, block4_out, block5_out]
# source_features = None


#from operations import Convolution2D_pad  #randomly initialize conv weight of source model
from operation_1 import Convolution2D_pad  #initialize conv weights by zero

# build model


# Block 1
out = Convolution2D_pad(64, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1')(input)
out = BatchNormalization(name='batch_normalization_1')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(64, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2')(out)
out = BatchNormalization(name='batch_normalization_2')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)

# Block 2
out = Convolution2D_pad(128, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1')(out)
out = BatchNormalization(name='batch_normalization_3')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(128, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2')(out)
out = BatchNormalization(name='batch_normalization_4')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)

# Block 3
out = Convolution2D_pad(256, (3, 3), source_features=source_features, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1')(out)
out = BatchNormalization(name='batch_normalization_5')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(256, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2')(out)
out = BatchNormalization(name='batch_normalization_6')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(256, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3')(out)
out = BatchNormalization(name='batch_normalization_7')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(256, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4')(out)
out = BatchNormalization(name='batch_normalization_8')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

# Block 4
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1')(out)
out = BatchNormalization(name='batch_normalization_9')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2')(out)
out = BatchNormalization(name='batch_normalization_10')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3')(out)
out = BatchNormalization(name='batch_normalization_11')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4')(out)
out = BatchNormalization(name='batch_normalization_12')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(out)

# Block 5
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1')(out)
out = BatchNormalization(name='batch_normalization_13')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2')(out)
out = BatchNormalization(name='batch_normalization_14')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3')(out)
out = BatchNormalization(name='batch_normalization_15')(out)
out = Activation('relu')(out)
out = Convolution2D_pad(512, (3, 3), source_features=source_features,padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4')(out)
out = BatchNormalization(name='batch_normalization_16')(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(out)

# model modification for cifar-10
out = Flatten(name='flatten')(out)
out = Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10')(out)
out = BatchNormalization(name='batch_normalization_17')(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)
out = Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2')(out)
out = BatchNormalization(name='batch_normalization_18')(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)     
out = Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10')(out)      
out = BatchNormalization(name='batch_normalization_19')(out)
output = Activation('softmax')(out)
target_model = Model(input,output)
# print('xx')
# ll = [layer.name for layer in model.layers]
# print(ll)

print("start loading weights of target model")
# target_model.load_weights("C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5",by_name=True)
#load_weight_by_weight_name(target_model,"C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5")
# load_weight_by_weight_name(model,"../../data/output/model_vgg19_minist/model_at_epoch_190.h5")
load_weight_by_weight_name(target_model,'C:\\Users\\beansprouts\\Desktop\\transfer\\data\\model_at_epoch_190.h5')
print("target model loading weight finished")

'''
from utills import load_weight_by_weight_name
print('start loading weights')
# source_model.load_weights("C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5",by_name=True)
#load_weight_by_weight_name(source_model,"C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5")
load_weight_by_weight_name(source_model,'../../data/output/model_vgg19_minist/model_at_epoch_190.h5')
print("source model loading weight finished")
# target_model.load_weights("C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5",by_name=True)
#load_weight_by_weight_name(target_model,"C:\\Users\\beansprouts\\Desktop\\transfer\\transfer\\model_at_epoch_190.h5")
load_weight_by_weight_name(target_model,"../../data/output/model_vgg19_minist/model_at_epoch_190.h5")
print("target model loading weight finished")
'''

'''
s = source_model.layers[0].get_weights()
t = target_model.layers[0].get_weights()
print(len(s),s[0])
print(len(t),t[0],t[1])
print("sss")
'''


# dataset = numpy.load('../../data/usps/usps_15.npz')
dataset = numpy.load('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz')
train_set_x = dataset['train_set_x']
train_set_y = dataset['train_set_y']
test_set_x = dataset['test_set_x']
test_set_y = dataset['test_set_y']
train_set_x = train_set_x.reshape(-1,32,32,1)
test_set_x = test_set_x.reshape(-1,32,32,1)
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
source_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
r=source_model.evaluate(test_set_x,test_set_y)
print(r)
target_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
rr=target_model.evaluate(test_set_x,test_set_y)
print(rr)








# def scheduler(epoch):
#     if epoch < 100:
#         return 0.01
#     if epoch < 150:
#         return 0.005
#     return 0.001

# if __name__ == '__main__':

#     # load data

#     # dataset = numpy.load('../../data/usps/usps_15.npz')
#     dataset = numpy.load('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz')
#     train_set_x = dataset['train_set_x']
#     train_set_y = dataset['train_set_y']
#     test_set_x = dataset['test_set_x']
#     test_set_y = dataset['test_set_y']
#     print(train_set_x.shape)
#     print(train_set_y.shape)
#     print(test_set_x.shape)
#     print(test_set_y.shape)

#     train_set_x = train_set_x.reshape(-1,32,32,1)
#     test_set_x = test_set_x.reshape(-1,32,32,1)


#     # set callback
#     tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
#     change_lr = LearningRateScheduler(scheduler)
#     #cbks = [change_lr,tb_cb]
#     cbks = [change_lr]


#     # start train
#     target_model.fit(train_set_x, train_set_y,
#               batch_size=20,
#               epochs=200,
#               callbacks=cbks,
#               validation_data=(test_set_x, test_set_y),
#               shuffle=True)

