from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 24
epochs = 200
lr = 0.01
momentum = 0.9
no_cuda =False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "../../../data/Original_images/"
#source_name = "amazon/images"
source_name = "dslr/images"
target_name = "webcam/images"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url,model_dir="../../../data/models")
    model_dict = model.state_dict()
    #print(type(pretrained_dict))
    #print(type(model_dict))
    #print(pretrained_dict.keys())
    #print(model_dict.keys())

    for k, v in model_dict.items():
        if (not "cls_fc" in k) and (not "num_batches_tracked" in k) and (not "__" in k):
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.rec_dense.parameters()},
        {'params': model.rec_feat.parameters()},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_mmd, image_rec_s, image_rec_t = model(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        print("gamma:",gamma)
        
        loss_rec_s = F.mse_loss(image_rec_s,data_source)
        loss_rec_t = F.mse_loss(image_rec_t,data_target)

        loss_rec = loss_rec_s + loss_rec_t

        #print(image_rec)

        loss = loss_cls + gamma * loss_mmd #+ loss_rec
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\trec_Loss: {:.6f}'\
                    .format(epoch, i * len(data_source), len_source_dataset,
                             100. * i / len_source_loader, loss.data[0], loss_cls.data[0], loss_mmd.data[0],loss_rec.data[0]))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output, _, __, ___ = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    model = models.DANNet_Rec(num_classes=31)
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain(model)
    
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, target_name, correct, 100. * correct / len_target_dataset ))
    
