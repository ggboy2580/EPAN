from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d

__all__ = ['ResNet50', 'ResNet101']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super().__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)

        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #self.base = nn.Sequential(*list(resnet50.children())[:])
        #self.base = nn.Sequential(*list(resnet50.children())[:])
        self.epanlinear=nn.Linear(524288,2048)
        self.epanlinear2=nn.Linear(4096,2048)
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        ex = x
        x = self.base(x)
        # 获取前5个模块的输出
        ex = self.base[:5](ex)
        print(ex.shape)
        ex=ex.view(ex.size(0),-1)
        print(ex.shape)
        ex=self.epanlinear(ex)
        print(ex.shape)

        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
            lf = self.conv2(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)#这一步是需要张平的，作用还是把（32,2048,1,1，）转换成（32，2048）
        f = torch.cat([ex, f], dim=1)
        f=self.epanlinear2(f)
        f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)#这一步是数据归一化
        print(f.shape)

        if not self.training:#如果不是训练
            return f,lf

        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=False)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.base2 = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


if __name__ == '__main__':
    resnet50=ResNet50(751)
    imgs=torch.Tensor(32,3,256,128)
    f=resnet50(imgs)
    print(f.shape)


