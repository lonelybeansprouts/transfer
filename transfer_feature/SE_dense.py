from keras.layers import Dense,Lambda,Concatenate,Reshape,GlobalAveragePooling1D,Input,BatchNormalization,Dropout,MaxPooling2D,Activation,Conv2D,Flatten
import keras.regularizers as regularizers
import keras.backend as K
from keras.layers.merge import multiply
import tensorflow as tf
import numpy as np
from keras.models import Model
import keras
from keras.initializers import he_normal



weight_decay = 0.0005
dropout = 0.5



def add_common_layer(x):
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def dense_SE(input, size, group1_num, group2_num, ratio):
    input_size = K.int_shape(input)[-1]
    group1_input_size = input_size // group1_num
    group2_output_size = size // group1_num    
    group1 = []
    for i in range(group1_num):
        group = Lambda(lambda z: z[:, i * group1_input_size : i * group1_input_size + group1_input_size])(input)
        group1.append(Dense(group2_output_size,activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(group))
    output = Concatenate()(group1)

    output = add_common_layer(output)
    
    output = Reshape([size//group2_num, group2_num])(output)  
    se = GlobalAveragePooling1D()(output)
    se = Reshape([1,group2_num])(se)
    se = Dense(group2_num // ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
    se = Dense(group2_num, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
    output = multiply([output, se])
    return Reshape([size])(output)



input = Input(shape=(32,32,3))
# build model
out = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1')(input)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(64, (3, 3),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)

# Block 2
out = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)

# Block 3
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(256, (3, 3),  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

# Block 4
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(out)

# Block 5
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(out)

# model modification for cifar-10
out = Flatten(name='flatten')(out)
out = dense_SE(input=out, size=4096, group1_num=32, group2_num=256, ratio=16)
#out = Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)
out = dense_SE(input=out, size=4096, group1_num=32, group2_num=256, ratio=16)
#out = Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2')(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = Dropout(dropout)(out)     
out = Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10')(out)      
out = BatchNormalization()(out)
output = Activation('softmax')(out)
model = Model(input,output)



a = np.ones(shape = [3,32,32,3])
r = model.predict(a)
print(r)


mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test
   
(x_train, y_train), (x_test, y_test) = load_cifar()    
# #     # color preprocessing
x_train, x_test = color_preprocessing(x_train, x_test)  
# build network  
print(model.summary())
# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
#     sgd = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# set callback
cbks = [TensorBoard(log_dir='./resnet_32/', histogram_freq=0),
        LearningRateScheduler(scheduler),
        ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto',save_weights_only=True, period=10)]
# set data augmentation
print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
                                width_shift_range=0.125,
                                height_shift_range=0.125,
                                fill_mode='constant',cval=0.)
datagen.fit(x_train)
# start training
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))