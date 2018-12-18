from scipy.io.matlab.mio import loadmat
import numpy
from PIL import Image
import random




def make_dataset(file='usps.mat',num_train_each_class=15,num_test=None):
#图片大小为16*16，输出时将图片缩放为了28*28,数据值没有归一化，而minist是归一化数据
#数据值跨度为-1.0到1.0 而mnist数据值跨度为0到1.0
#数据集的label跨度为1到10，而mnist为0-9, 实际图像的取值

    data = loadmat(file)

    test_set_x = data.get('Xte')
    test_set_y = data.get('Yte')
    train_set_x = data.get('Xtr')
    train_set_y = data.get('Ytr')
    test_set_x = test_set_x.toarray()
    train_set_x = train_set_x.toarray()

    train_set_x = (train_set_x+1)/2.0
    test_set_x = (test_set_x+1)/2.0
    train_set_y = train_set_y.ravel()-1
    test_set_y = test_set_y.ravel()-1
    
    l_tr = []
    for i in range(train_set_x.shape[0]):
        img = train_set_x[i].reshape(16,16)*255
        img = Image.fromarray(img)
        img = img.resize(size=(32,32))
        img = numpy.array(img)
        img = img.ravel()/255.0
        l_tr.append(img)
    train_set_x = numpy.vstack(l_tr)
    print(train_set_x.shape)

    l_te = []
    for i in range(test_set_x.shape[0]):
        img = test_set_x[i].reshape(16,16)*255
        img = Image.fromarray(img)
        img = img.resize(size=(32,32))
        img = numpy.array(img)
        img = img.ravel()/255.0
        l_te.append(img)
    test_set_x = numpy.vstack(l_te)
    print(test_set_x.shape)

    # img = Image.fromarray(train_set_x[0].reshape(32,32)*255)
    # img.show()

    # level sampling for training set
    train_L = []
    class_L = []
    print(train_set_x.shape)
    print(train_set_y.shape)
    for i in range(10):
        for j in range(train_set_y.shape[0]):
            if train_set_y[j]==i:
                class_L.append((train_set_x[j],train_set_y[j]))
        print(len(class_L))
        random.shuffle(class_L)  #乱序采样
        train_L.extend(class_L[0:num_train_each_class])
        class_L = []
    # print(len(train_L))
    # for x,y in train_L:
    #     print(y)
    random.shuffle(train_L)
    # print(len(train_L))
    # for x,y in train_L:
    #     print(y)
    x,y = zip(*train_L)
    train_set_x = numpy.vstack(x)
    train_set_y = numpy.hstack(y)

    # print(x.shape)
    # print(y.shape)
    # img = Image.fromarray(train_set_x[4].reshape(32,32)*255)
    # img.show()
    # print(train_set_y[4])

    numpy.savez('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz',train_set_x=train_set_x,train_set_y=train_set_y,\
                                                               test_set_x=test_set_x,test_set_y=test_set_y)
    #loading data
    #
    # dataset = numpy.load('G:\\AAA_workspace\\dataset\\usps\\usps_15.npz')
    # print(type(dataset))
    # print(dataset.files)
    # train_set_x = dataset['train_set_x']
    # train_set_y = dataset['train_set_y']
    # test_set_x = dataset['test_set_x']
    # test_set_y = dataset['test_set_y']
    # print(train_set_x.shape)
    # print(train_set_y.shape)
    # print(test_set_x.shape)
    # print(test_set_y.shape)
    # img = Image.fromarray(test_set_x[0].reshape(32,32)*255)
    # img.show()
    # print(test_set_y[0])
  
#######################
# make usps dataset 
#######################
file = "G:\\AAA_workspace\\dataset\\usps.mat"
make_dataset(file=file)