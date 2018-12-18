import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import LearningRateScheduler, TensorBoard
import numpy

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
#    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001

if __name__ == '__main__':

    # load data
    dataset = numpy.load('../../data/usps/usps_15.npz')
    #dataset = numpy.load('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz')
    train_set_x = dataset['train_set_x']
    train_set_y = dataset['train_set_y']
    test_set_x = dataset['test_set_x']
    test_set_y = dataset['test_set_y']
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)

    train_set_x = train_set_x.reshape(-1,32,32,1)
    test_set_x = test_set_x.reshape(-1,32,32,1)

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    #tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]#,tb_cb]

    # start train
    model.fit(train_set_x, train_set_y,
              batch_size=20,
              epochs=200,
              callbacks=cbks,
              validation_data=(test_set_x, test_set_y),
              shuffle=True)

    # save model
    #model.save('lenet.h5',)

