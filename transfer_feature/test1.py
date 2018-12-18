# import keras 
# from keras.layers import Dense, Input, add, Activation, Flatten, AveragePooling2D, MaxPooling2D, Dropout, Lambda, Conv2D, Conv2DTranspose
# from keras.models import Model
# import numpy as np
# from keras.layers import UpSampling2D
# from keras import backend as K

# input = Input(shape=(32,32,3))

# output = Conv2D(filters=64,kernel_size=3,padding='same',strides=(2,2))(input)

# output = MaxPooling2D(pool_size=(2,2))(output)

# print(output.shape)

# output = UpSampling2D(size=(2,2))(output)

# print(K.int_shape(output))

# # tran_output = Conv2DTranspose(filters=3,kernel_size=3,padding='same',strides=(4,4))(output)

# # print(tran_output.shape)

# model = Model(input,output)
# o = model.predict(np.ones(shape=(3,32,32,3),dtype='float32'))
# print(o.shape)

# model = Model(input,tran_output)
# o = model.predict(np.ones(shape=(3,32,32,3),dtype='float32'))
# print(o.shape)


# i = 6
# ss = "abc"
# s = ss+"ssss%d"%6

# print(s)

# a = [3,"12",2,"12",4]

# if "12" in a:
#     print(a.index("12"))










import keras
import numpy as np
import matplotlib.pyplot as plt
#按顺序构成的模型
from keras.models import Sequential
#Dense全连接层
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import L1L2 as WR



np.random.seed(0)
x_data=np.linspace(-10,10,20)
noise=np.random.normal(0,1,x_data.shape)
y_data=np.square(x_data)+noise+x_data
# y_data=y_data/np.max(y_data)

#构建一个顺序模型
model=Sequential()

#在模型中添加一个全连接层
#units是输出维度,input_dim是输入维度(shift+两次tab查看函数参数)
#输入1个神经元,隐藏层10个神经元,输出层1个神经元
model.add(Dense(units=200,input_dim=1,kernel_initializer='normal',kernel_regularizer=WR(l1=0.1, l2=0.01)))
model.add(Activation('sigmoid'))   #增加非线性激活函数
model.add(Dense(200,kernel_initializer='normal',kernel_regularizer=WR(l1=0.01, l2=0.01)))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))   #默认连接上一层input_dim=10
# model.add(Activation('tanh'))

#定义优化算法(修改学习率)
adam = Adam(lr=0.1)

#编译模型
model.compile(optimizer=adam,loss='mse')   #optimizer参数设置优化器,loss设置目标函数


est = EarlyStopping(monitor='tt')
#训练模型
model.fit(x_data,y_data,epochs=100,callbacks=[est])

#打印权值和偏置值
# W,b=model.layers[0].get_weights()   #layers[0]只有一个网络层
# print('W:',W,'b:',b)

#x_data输入网络中，得到预测值y_pred
y_pred=model.predict(x_data)
print(y_pred)

plt.scatter(x_data,y_data)

plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()
