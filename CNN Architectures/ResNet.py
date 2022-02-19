import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet_block(nn.Module):
  def __init__(self, input_dim, output_dim, strides, proj= False):
    super(resnet_block, self).__init__()

    self.conv1= nn.Conv2d(input_dim, output_dim, kernel_size= 3, padding=1, stride= strides)
    self.conv2= nn.Conv2d(output_dim, output_dim, kernel_size= 3, padding= 1, stride= 1)

    #proj = True only when the no. of channels double
    if proj:
      self.conv3= nn.Conv2d(input_dim, output_dim, kernel_size= 1, padding= 0, stride= strides)
    else:
      self.conv3= None
    
    self.bn= nn.BatchNorm2d(output_dim)

  def forward(self, x):
    out= F.relu(self.bn(self.conv1(x)))
    out= self.bn(self.conv2(out))

    if self.conv3:
      x= self.bn(self.conv3(x))
    
    out += x
    out= F.relu(out)
    return out

    
class bottleneck_block(nn.Module):
  def __init__(self, input_dim, intermediate_dim, output_dim, strides, proj= False):
    super(bottleneck_block, self).__init__()
    self.conv1= nn.Conv2d(input_dim, intermediate_dim, kernel_size= 1, padding= 0, stride= strides)
    self.conv2= nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size= 3, padding= 1, stride= 1)
    self.conv3= nn.Conv2d(intermediate_dim, output_dim, kernel_size= 1, padding= 0, stride=1)

    if proj:
      self.conv4= nn.Conv2d(input_dim, output_dim, kernel_size= 1, padding=0, stride= strides)
    else:
      self.conv4= None

    self.bn1= nn.BatchNorm2d(intermediate_dim)
    self.bn2= nn.BatchNorm2d(output_dim)

  def forward(self, x):
    out= F.relu(self.bn1(self.conv1(x)))
    out= F.relu(self.bn1(self.conv2(out)))
    out= self.bn2(self.conv3(out))

    if self.conv4:
      x= self.bn2(self.conv4(x))
    
    out += x
    out= F.relu(out)
    return out


def make_block(input_dim, output_dim, num_blocks, intermediate_dim= 0, bottleneck= False, first_block= False):
  '''
  make blocks for both resnet34 and resnet50 
  bottleneck= is True for resnet50
  first_block= #channels double in the first block, set proj= True 
  '''

  layers= []

  for i in range(num_blocks):
    if bottleneck:
      if i==0 and first_block:
        if input_dim== 64:
          #no. of channels double but no downsampling in the first block in resnet50
          layers.append(bottleneck_block(input_dim, intermediate_dim, output_dim, strides= 1, proj= True)) 
        else:
          layers.append(bottleneck_block(input_dim, intermediate_dim, output_dim, strides= 2, proj= True))
      else:
        layers.append(bottleneck_block(output_dim, intermediate_dim, output_dim, strides= 1, proj= False))
    else:
      if i== 0 and first_block:
        layers.append(resnet_block(input_dim, output_dim, strides= 2, proj= True))
      else:
        layers.append(resnet_block(output_dim, output_dim, strides= 1, proj= False))
  
  return layers


class ResNet34(nn.Module):
  def __init__(self, input_dim):
    super(ResNet34, self).__init__()
    self.features= nn.Sequential(
        nn.Conv2d(input_dim, 64, kernel_size= 7, stride= 2, padding= 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
    )

    self.blocks= nn.Sequential(*make_block(64, 64, num_blocks= 3),
                               *make_block(64, 128, num_blocks= 4, first_block= True),
                               *make_block(128, 256, num_blocks= 6, first_block= True),
                               *make_block(256, 512, num_blocks= 3, first_block= True)
    )

    self.classifier= nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 1000)
    )
  
  def forward(self, x):
    x= self.features(x)
    x= self.blocks(x)
    x= self.classifier(x)
    return x


class ResNet50(nn.Module):
  def __init__(self, input_dim):
    super(ResNet50, self).__init__()

    self.features= nn.Sequential(
        nn.Conv2d(input_dim, 64, kernel_size= 7, stride= 2, padding= 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
    )

    self.blocks= nn.Sequential(*make_block(64, 256, intermediate_dim= 64, num_blocks= 3, bottleneck= True, first_block= True),
                               *make_block(256, 512, intermediate_dim= 128, num_blocks= 4, bottleneck= True, first_block= True),
                               *make_block(512, 1024, intermediate_dim= 256, num_blocks= 6, bottleneck= True, first_block= True),
                               *make_block(1024, 2048, intermediate_dim= 512, num_blocks= 4, bottleneck= True, first_block= True)
    )

    self.classifier= nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 1000)
    )
  
  def forward(self, x):
    x= self.features(x)
    x= self.blocks(x)
    x= self.classifier(x)
    return x


if __name__== '__main__':
    test_eg= torch.randn([1, 3, 224, 224])
    
    model1= ResNet34(input_dim= 3)
    pred1= model1(test_eg)
    print(pred1.shape)
    
    model2= ResNet50(input_dim= 3)
    pred= model2(test_eg)
    print(pred.shape)