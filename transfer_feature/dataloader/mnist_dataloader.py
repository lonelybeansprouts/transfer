import pickle
from PIL import Image
import numpy


def read_mnist(dataset='mnist.pkl'):
    with open(dataset,'rb') as f:
        data = pickle.load(f,encoding='latin1')
    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x, test_set_y = data[2]        

    l_tr = []
    for i in range(train_set_x.shape[0]):
        img = train_set_x[i].reshape(28,28)*255
        img = Image.fromarray(img)
        img = img.resize(size=(32,32))
        img = numpy.array(img)
        img = img.ravel()/255.0
        l_tr.append(img)
    train_set_x = numpy.vstack(l_tr)
    print(train_set_x.shape) 

    l_val = []
    for i in range(valid_set_x.shape[0]):
        img = valid_set_x[i].reshape(28,28)*255
        img = Image.fromarray(img)
        img = img.resize(size=(32,32))
        img = numpy.array(img)
        img = img.ravel()/255.0
        l_val.append(img)
    valid_set_x = numpy.vstack(l_val)
    print(valid_set_x.shape)  

    l_te = []
    for i in range(test_set_x.shape[0]):
        img = test_set_x[i].reshape(28,28)*255
        img = Image.fromarray(img)
        img = img.resize(size=(32,32))
        img = numpy.array(img)
        img = img.ravel()/255.0
        l_te.append(img)
    test_set_x = numpy.vstack(l_te)
    print(test_set_x.shape)           

    return [[train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]]

# dataset =  read_mnist(dataset='G:\AAA_workspace\dataset\mnist\mnist.pkl')
# train_set_x, train_set_y = dataset[0]
# valid_set_x, valid_set_y = dataset[1]
# test_set_x, test_set_y = dataset[2]

# print(train_set_x.shape)
# print(train_set_y.shape)
# img = Image.fromarray(train_set_x[0].reshape(32,32)*255)
# img.show()
# print(train_set_y[0])