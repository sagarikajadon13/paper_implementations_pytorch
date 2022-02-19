import torch
import torch.nn as nn

class VGG(nn.Module):
  def __init__(self, features, output_dim):
    super(VGG, self).__init__()

    self.features= features
    self.avgpool= nn.AdaptiveAvgPool2d(7)

    self.classifier= nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(inplace= True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace= True),
        nn.Dropout(0.5),
        nn.Linear(4096, output_dim)
    )

  def forward(self, x):
    x= self.features(x)
    x= self.avgpool(x)
    x= x.reshape(x.shape[0], -1)
    x= self.classifier(x)
    return x


VGG_11= [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

VGG_13= [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

VGG_16= [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

vgg_19= [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def VGG_features(config, batchnorm= True):
  layers= []
  in_channels= 3

  for i in config:
    assert i == 'M' or isinstance(i, int)

    if i == 'M':
      layers += [nn.MaxPool2d(kernel_size= 2)]
    else:
      if batchnorm:
        layers += [nn.Conv2d(in_channels, i, kernel_size= 3, padding= 1), nn.BatchNorm2d(i), nn.ReLU(inplace= True)]
      else:
        layers += [nn.Conv2d(in_channels, i, kernel_size= 3, padding= 1), nn.ReLU(inplace= True)]
      in_channels= i
  
  return nn.Sequential(*layers)


if __name__== '__main__':
    test= torch.randn(1, 3, 299, 299)
    features= VGG_features(VGG_11, batchnorm= True)
    model= VGG(features= features, output_dim= 1000)
    out= model(test)
    print(out.shape)