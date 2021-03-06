import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation, Input, add, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.layers import concatenate, AveragePooling2D
import numpy
# from dataloader.mnist_dataloader import read_mnist
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras import regularizers 
from dataloader.mnist_dataloader import read_mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

growth_rate        = 12 
depth              = 100
compression        = 0.5

weight_decay       =  0.0005
dropout            =0.5

# build model

out_block_1 = []
out_block_2 = []
out_block_3 = []

def densenet(img_input,classes_num):
    
    def conv(x, out_filters, k_size):

        return Conv2D(filters=out_filters,
                    kernel_size=k_size,
                    strides=(1,1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x,out_blocks):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1,1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        out_blocks.append(x)
        return x

    def single(x):
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x, blocks, nchannels, out_blocks):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat, out_blocks)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2
    
    x = conv(img_input, nchannels, (3,3))
    x, nchannels = dense_block(x,nblocks,nchannels,out_block_1)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels,out_block_2)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels,out_block_3)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

input = Input(shape=(32,32,1))
output = densenet(img_input=input,classes_num=10)
model = Model(input,output)
print(model.summary())


print(len(out_block_1))
print(len(out_block_2))
print(len(out_block_3))

# #多gpu并行
# # model = multi_gpu_model(model,2)


sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 100:
         return 0.1
    if epoch < 150:
         return 0.01
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

