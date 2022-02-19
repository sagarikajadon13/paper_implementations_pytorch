import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception_block(nn.Module):
  def __init__(self, input_channels, k1, k2, k3, k4):
    super(Inception_block, self).__init__()
    
    self.p1= nn.Conv2d(input_channels, k1, kernel_size= 1)

    self.p2_1= nn.Conv2d(input_channels, k2[0], kernel_size= 1, stride=1)
    self.p2_2= nn.Conv2d(k2[0], k2[1], kernel_size= 3, stride=1, padding= 1)

    self.p3_1= nn.Conv2d(input_channels, k3[0], kernel_size= 1, stride=1)
    self.p3_2= nn.Conv2d(k3[0], k3[1], kernel_size= 5, stride=1, padding= 2)

    self.p4_1= nn.MaxPool2d(kernel_size= 3, stride= 1, padding=1)
    self.p4_2= nn.Conv2d(input_channels, k4, kernel_size= 1, stride=1)

  def forward(self, x):
    out1= F.relu(self.p1(x))

    out2= F.relu(self.p2_1(x))
    out2= F.relu(self.p2_2(out2))

    out3= F.relu(self.p3_1(x))
    out3= F.relu(self.p3_2(out3))

    out4= self.p4_1(x)
    out4= F.relu(self.p4_2(out4))

    out= torch.cat((out1, out2, out3, out4), dim= 1)
    return out


class GoogLeNet(nn.Module):
  def __init__(self, input_dims):
    super(GoogLeNet, self).__init__()

    self.features= nn.Sequential(
        nn.Conv2d(3, 64, kernel_size= 7, stride= 2, padding= 3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 3, stride=2, padding= 1),
        nn.BatchNorm2d(64),

        nn.Conv2d(64, 64, kernel_size= 1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, stride=1, padding= 1),
        nn.ReLU(),
        nn.BatchNorm2d(192),
        nn.MaxPool2d(kernel_size= 3, stride=2, padding= 1)
    )

    self.inceptions= nn.Sequential(
        Inception_block(192, 64, [96, 128], [16, 32], 32),
        Inception_block(input_channels= 256, k1= 128, k2= [128, 192], k3= [32, 96], k4= 64),
        nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),

        Inception_block(480, 192, [96, 208], [16, 48], 64),
        Inception_block(512, 160, [112, 224], [24, 64], 64),
        Inception_block(512, 128, [128, 256], [24, 64], 64),
        Inception_block(512, 112, [144, 288], [32, 64], 64),
        Inception_block(528, 256, [160, 320], [32, 128], 128),
        nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),

        Inception_block(832, 256, [160, 320], [32, 128], 128),
        Inception_block(832, 384, [192, 384], [48, 128], 128),
        nn.AdaptiveAvgPool2d((1, 1))
    )

    self.classifier= nn.Sequential(
        nn.Dropout(p= 0.4),
        nn.Linear(1024, 1000)
    )

  def forward(self, x):
    pred= self.features(x)
    pred= self.inceptions(pred)
    pred= pred.reshape(pred.shape[0], -1)
    pred= self.classifier(pred)
    return pred


if __name__== '__main__':
    test= torch.randn(1, 3, 224, 224)
    model= GoogLeNet(3)
    out= model(test)
    print(out.shape)