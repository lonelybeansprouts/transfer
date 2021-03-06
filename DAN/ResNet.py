import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch
import numpy as np

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class DANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(DANNet, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        '''
        if self.training == True:
            target = self.sharedNet(target)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        '''
        source = self.cls_fc(source)
        #target = self.cls_fc(target)

        return source, loss

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



class DANNet_Rec(nn.Module):

    def __init__(self, num_classes=31):
        super(DANNet_Rec, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = nn.Linear(2048, num_classes)

        self.rec_dense = nn.Sequential()
        self.rec_dense.add_module('__fc5_', nn.Linear(in_features=2048, out_features=2048))
        self.rec_dense.add_module('__relu5_', nn.ReLU(True))
        self.rec_dense.add_module('__fc4_', nn.Linear(in_features=2048, out_features=512 * 7 * 7))
        self.rec_dense.add_module('__relu4_', nn.ReLU(True))

        self.rec_feat = nn.Sequential()
        self.rec_feat.add_module('__conv3_', nn.Conv2d(in_channels=512, out_channels=256,
                                                kernel_size=3, padding=1))
        self.rec_feat.add_module('__relu3_', nn.ReLU(True))
        self.rec_feat.add_module('__pool3_', nn.Upsample(scale_factor=4))
        self.rec_feat.add_module('__conv2_', nn.Conv2d(in_channels=256, out_channels=128,
                                                     kernel_size=3, padding=1))
        self.rec_feat.add_module('__relu2_', nn.ReLU(True))
        self.rec_feat.add_module('__pool2_', nn.Upsample(scale_factor=2))
        self.rec_feat.add_module('__conv1_', nn.Conv2d(in_channels=128, out_channels=64,
                                                     kernel_size=3, padding=1))
        self.rec_feat.add_module('__relu1_', nn.ReLU(True))
        self.rec_feat.add_module('__pool1_', nn.Upsample(scale_factor=2))
        self.rec_feat.add_module('__conv0_', nn.Conv2d(in_channels=64, out_channels=32,
                                                          kernel_size=3, padding=1))
        self.rec_feat.add_module('__relu0_', nn.ReLU(True))
        self.rec_feat.add_module('__pool0_', nn.Upsample(scale_factor=2))
        self.rec_feat.add_module('__conv_last_', nn.Conv2d(in_channels=32, out_channels=3,
                                                                  kernel_size=3, padding=1))

    def forward(self, source, target):
        loss = 0
        img_rec_s = 0
        img_rec_t = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
            
            feat_encode = self.rec_dense(source)
            feat_encode = feat_encode.view(-1, 512, 7, 7)
            img_rec_s = self.rec_feat(feat_encode)
            
            feat_encode = self.rec_dense(target)
            feat_encode = feat_encode.view(-1, 512, 7, 7)
            img_rec_t = self.rec_feat(feat_encode)
        
        source = self.cls_fc(source)
        #target = self.cls_fc(target)

        return source, loss, img_rec_s, img_rec_t
    
'''
a = np.ones([10,3,224,224])
a = torch.Tensor(a)
model = DANNet_Rec(31)
model.train()
a,b,c = model(a,a)

print(c.shape)
'''
