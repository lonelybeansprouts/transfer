import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy
from dataloader.mnist_dataloader import read_mnist
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.utils import multi_gpu_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



weight_decay       =  0.0005
dropout            =0.5



class MyCbk(keras.callbacks.Callback):

    def __init__(self, model,output_dir,freq=None):
        self.model_to_save = model
        self.freq = freq
        self.output_dir = output_dir
    def on_epoch_end(self, epoch, logs=None):
        if self.freq:
            if epoch % self.freq==0:
                self.model_to_save.save_weights(self.output_dir+'/'+'model_at_epoch_%d.h5' % epoch)
        else:
            self.model_to_save.save_weights(self.output_dir+'/'+'model_at_epoch_%d.h5' % epoch)

# build model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1' , input_shape=(32,32,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# model modification for cifar-10
model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))      
model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))        
model.add(BatchNormalization())
model.add(Activation('softmax'))

#multi gpu 

source_model = model

#model = multi_gpu_model(model,1)

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
    
    
    '''
    # set callback
    tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    #ckpt = ModelCheckpoint('./output/ckpt.h5', save_weights_only=True, save_best_only=False, mode='auto', period=10)
    myckpt = MyCbk(model=source_model,output_dir='./output',freq=10)
    cbks = [change_lr,tb_cb,myckpt]
    
    # start train
    model.fit(train_set_x, train_set_y,
              batch_size=128,
              epochs=200,
              callbacks=cbks,
              validation_data=(test_set_x, test_set_y),
              shuffle=True)
    
    '''
    model.load_weights("./output/model_at_epoch_190.h5")
    r = model.evaluate(test_set_x,test_set_y)
    source_model.save_weights("./output/weights.h5")
    print(r)
    
