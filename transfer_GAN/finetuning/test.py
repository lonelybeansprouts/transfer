from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import ResNet as resnet
import os
import collections
import AlexNet
from torch.utils import model_zoo
import data_loader
import math
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
from torch.nn import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = True if torch.cuda.is_available() else False
LEARNING_RATE = 0.001
BATCH_SIZE_SRC = 64
BATCH_SIZE_TAR = 1
L2_DECAY = 5e-4
DROPOUT = 0.5
N_EPOCH = 200
MOMENTUM = 0.9
RES_TRAIN = []
RES_TEST = []



def test(model, data_tar):
    model.eval()
    n_batch = len(data_tar)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_tar):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            print(output.data)
            print(pred)
            print(target.data.view_as(pred))
            print("xxxxxxxxxxxxxxxxxxxxx")

            total_loss += loss.item()
        acc_test = float(correct) * 100. / (n_batch * BATCH_SIZE_SRC)
        tqdm.write('\ttest loss: {:.6f},\tcorrect: [{}/{}],\ttest accuracy: {:.4f}%'.format(
            total_loss / n_batch, correct, len(data_tar.dataset), acc_test
        ))
        RES_TEST.append([ total_loss / n_batch, acc_test])


def load_model(name='alexnet'):
    if name == 'alexnet':
        model = AlexNet.AlexNetFc(pretrained=False, num_classes=31)
        # torch.nn.init.xavier_uniform_(model.nfc.weight)
        # torch.nn.init.constant_(model.nfc.bias, 0.1)
    elif name == 'resnet':
        model = resnet.myresnet(pretrained=False, num_classes=31)
        # torch.nn.init.xavier_uniform_(model.nfc.weight.data)
        # torch.nn.init.constant_(model.nfc.bias.data, 0.01)
    return model


def lr_decay(LR, n_epoch, e):
    return LR / math.pow((1 + 10 * e / n_epoch), 0.75)

def get_optimizer(model_name='alexnet', learning_rate=0.0001):
    optimizer = None
    if model_name == 'alexnet':
        optimizer = optim.SGD(params=[
            {'params': model.features.parameters()},
            {'params': model.classifier.parameters()},
            {'params': model.nfc.parameters(), 'lr': learning_rate * 10}
        ], lr=learning_rate, momentum=MOMENTUM, weight_decay=L2_DECAY)
    elif model_name == 'resnet':
        optimizer = optim.SGD(params=[
            {'params': model.features.parameters()},
            {'params': model.nfc.parameters(), 'lr': learning_rate * 10}
        ], lr=learning_rate, momentum=MOMENTUM, weight_decay=L2_DECAY)
    assert optimizer is not None
    return optimizer

if __name__ == '__main__':
    torch.manual_seed(10)
    root_dir = '../../../../data/Original_images/'
    src, tar = 'amazon/images', 'webcam/images'
#    root_dir = 'C:/Users/beansprouts/Desktop/office31_raw_image/Original_images/'
#    src, tar = 'amazon/images', 'webcam/images'
    #model_name = 'alexnet'
    model_name = 'resnet'
    data_src, data_tar = data_loader.load_training(root_dir, src, BATCH_SIZE_SRC), \
                         data_loader.load_testing(root_dir, tar, BATCH_SIZE_TAR)
    print('Source:{}, target:{}'.format(src, tar))

    model = load_model(model_name).to(DEVICE)

    model.load_state_dict(torch.load('../../../../data/models/params.pkl'))

    test(model=model,data_tar=data_tar)



     

