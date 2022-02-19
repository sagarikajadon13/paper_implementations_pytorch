import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_convolutions(nn.Module):
  '''
  input_channels= #no. of input channels for depthwise convolution
  kernels_per_channel= #filters to convolve with each input channel for depthwise convolution
  output_channels= #no. of ouput channels for pointwise convolution
  kernel_size= kernel_size for depthwise convolution
  stride= stride for depthwise convolution
  padding= padding for depthwise convolution
  '''

  def __init__(self, input_channels, kernels_per_channel, output_channels, kernel_size, stride, padding):
    super(depthwise_separable_convolutions, self).__init__()

    self.depthwise= nn.Conv2d(in_channels= input_channels, 
                              out_channels= input_channels*kernels_per_channel, 
                              kernel_size= kernel_size, 
                              stride= stride, 
                              padding= padding,
                              groups= input_channels)
    
    self.bn1= nn.BatchNorm2d(input_channels*kernels_per_channel)
    
    self.pointwise= nn.Conv2d(in_channels= input_channels*kernels_per_channel,
                              out_channels= output_channels,
                              kernel_size= 1,
                              stride= 1,
                              padding= 0)
    
    self.bn2= nn.BatchNorm2d(output_channels)
  
  def forward(self, x):
    out= self.depthwise(x)
    out= F.relu(self.bn1(out))
    out= self.pointwise(out)
    out= F.relu(self.bn2(out))
    return out


class MobileNetV1(nn.Module):
  def __init__(self):
    super(MobileNetV1, self).__init__()

    self.features= nn.Sequential(nn.Conv2d(3, 32, kernel_size= 3, stride=2, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 depthwise_separable_convolutions(32, 1, 64, 3, 1, 1),
                                 depthwise_separable_convolutions(64, 1, 128, 3, 2, 1),
                                 depthwise_separable_convolutions(128, 1, 128, 3, 1, 1),
                                 depthwise_separable_convolutions(128, 1, 256, 3, 2, 1),
                                 depthwise_separable_convolutions(256, 1, 256, 3, 1, 1),
                                 depthwise_separable_convolutions(256, 1, 512, 3, 2, 1),
                                 depthwise_separable_convolutions(512, 1, 512, 3, 1, 1),
                                 depthwise_separable_convolutions(512, 1, 512, 3, 1, 1),
                                 depthwise_separable_convolutions(512, 1, 512, 3, 1, 1),
                                 depthwise_separable_convolutions(512, 1, 512, 3, 1, 1),
                                 depthwise_separable_convolutions(512, 1, 512, 3, 1, 1),
                                 depthwise_separable_convolutions(512, 1, 1024, 3, 2, 1),
                                 depthwise_separable_convolutions(1024, 1, 1024, 3, 1, 1)
                                 )
    
    self.classifier= nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(1024, 1000))
  
  def forward(self, x):
    out= self.features(x)
    out= self.classifier(out)
    return out

if __name__== '__main__':
    test= torch.randn(1, 3, 224, 224)
    model= MobileNetV1()
    out= model(test)
    print(out.shape)