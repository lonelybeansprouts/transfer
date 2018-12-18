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
import keras
import os
import keras.preprocessing.image as image
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

growth_rate        = 12 
depth              = 100
compression        = 0.5

weight_decay       =  0.0005
dropout            =0.5

# build model


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

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1,1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
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

    def dense_block(x, blocks, nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2
    
    x = conv(img_input, nchannels, (3,3))
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x



input = Input(shape=(64,64,3))
output = densenet(img_input=input,classes_num=31)
base_model = Model(input,output)
print(base_model.summary())


#
model = multi_gpu_model(base_model,2)

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 100:
         return 0.1
    if epoch < 150:
        return 0.01
    return 0.001

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

if __name__ == '__main__':

     # load data

    data_gen_source = image.ImageDataGenerator()\
              .flow_from_directory(
                      #directory='C:\\Users\\beansprouts\\Desktop\\office31_raw_image\\Original_images\\amazon\\images',
                      directory='../../data/office31_raw_image/Original_images/amazon/images',
                      class_mode='sparse',
                      target_size=(64, 64),
                      shuffle=True,
                      batch_size=16)
    
    data_gen_target = image.ImageDataGenerator()\
              .flow_from_directory(
                     #directory='C:\\Users\\beansprouts\\Desktop\\office31_raw_image\\Original_images\\amazon\\images',
                     directory='../../data/office31_raw_image/Original_images/dslr/images',
                     class_mode='sparse',
                     target_size=(64, 64),
                     shuffle=False,
                     batch_size=16)




    '''
    change_lr = LearningRateScheduler(scheduler)
    # ckpt = ModelCheckpoint('./output/ckpt.h5', save_weights_only=True, save_best_only=False, mode='auto', period=10)
    ckpt = MyCbk(model=base_model,output_dir='./output',freq=10)
    cbks = [change_lr,ckpt]
    model.fit_generator(generator=data_gen_source,
                        steps_per_epoch=data_gen_source.n//16,
                        callbacks=cbks,
                        workers=4,
                        validation_data=data_gen_target,
                        validation_steps=data_gen_target.n//16,
                        epochs=200)
    '''



    base_model.load_weights("./output/model_at_epoch_150.h5")
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    base_model.compile(loss='sparse_categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

    r=base_model.evaluate_generator(generator=data_gen_source,
                                    steps=data_gen_source.n//16)
    print(r)
    r=base_model.evaluate_generator(generator=data_gen_target,
                                    steps=data_gen_target.n//16)
    print(r)

