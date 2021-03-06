import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation, Input, add, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy
# from dataloader.mnist_dataloader import read_mnist
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras import regularizers 
from dataloader.mnist_dataloader import read_mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



weight_decay       =  0.0005
dropout            =0.5

# build model

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
model = Model(input,output)
print(model.summary())


# #多gpu并行
# # model = multi_gpu_model(model,2)

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 100:
         return 0.01
    if epoch < 150:
         return 0.005
    return 0.001

if __name__ == '__main__':

     # load data

     dataset =  read_mnist(dataset='/home/xddz/dejian.zhong/data/mnist/mnist.pkl')
     train_set_x, train_set_y = dataset[0]
     valid_set_x, valid_set_y = dataset[1]
     test_set_x, test_set_y = dataset[2]
     print(train_set_x.shape)
     print(train_set_y.shape)
     print(test_set_x.shape)
     print(test_set_y.shape)

     train_set_x = train_set_x.reshape(-1,32,32,1)
     test_set_x = test_set_x.reshape(-1,32,32,1)

     # build network
     print(model.summary())

     # set callback
     #tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
     change_lr = LearningRateScheduler(scheduler)
     ckpt = ModelCheckpoint('./output/ckpt.h5', save_weights_only=True, save_best_only=False, mode='auto', period=10)
     cbks = [change_lr,ckpt]

     # start train
     model.fit(train_set_x, train_set_y,
               batch_size=128,
               epochs=200,
               callbacks=cbks,
               validation_data=(test_set_x, test_set_y),
               shuffle=True)

