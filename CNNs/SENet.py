import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicSEResnetBlock(nn.Module):
  def __init__(self, input_dim, output_dim, stride, reduction= 16, proj= False):
    '''proj = True only when the no. of channels double''' 
    super().__init__()

    self.conv1= nn.Conv2d(input_dim, output_dim, kernel_size= 3, padding=1, stride= stride)
    self.bn1= nn.BatchNorm2d(output_dim)
    self.conv2= nn.Conv2d(output_dim, output_dim, kernel_size= 3, padding= 1, stride= 1)
    self.bn2= nn.BatchNorm2d(output_dim)
    
    if proj:
      self.conv3= nn.Conv2d(input_dim, output_dim, kernel_size= 1, stride= stride)
      self.bn3= nn.BatchNorm2d(output_dim)
    else:
      self.conv3= None
      
    self.squeeze= nn.AdaptiveAvgPool2d((1, 1))
    reduced_dim= output_dim// reduction
    self.excitation= nn.Sequential(nn.Linear(output_dim, reduced_dim),
                                   nn.ReLU(inplace= True),
                                   nn.Linear(reduced_dim, output_dim),
                                   nn.Sigmoid())
    
    
  def forward(self, x):
    residual= F.relu(self.bn1(self.conv1(x)))
    residual= self.bn2(self.conv2(residual))

    if self.conv3:
      x= self.bn3(self.conv3(x))
      
    attention= self.squeeze(residual)
    attention= attention.reshape(attention.shape[0], -1)
    attention= self.excitation(attention)
    attention= attention.reshape((attention.shape[0], attention.shape[1], 1, 1))
    
    out= residual*attention + x
    out= F.relu(out)
    return out

    
class BottleneckSEResnetBlock(nn.Module):
  def __init__(self, input_dim, intermediate_dim, output_dim, stride, reduction= 16, proj= False):
    super().__init__()
    self.conv1= nn.Conv2d(input_dim, intermediate_dim, kernel_size= 1, stride= stride)
    self.bn1= nn.BatchNorm2d(intermediate_dim)
    
    self.conv2= nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size= 3, padding= 1, stride= 1)
    self.bn2= nn.BatchNorm2d(intermediate_dim)
    
    self.conv3= nn.Conv2d(intermediate_dim, output_dim, kernel_size= 1, stride=1)
    self.bn3= nn.BatchNorm2d(output_dim)

    if proj:
      self.conv4= nn.Conv2d(input_dim, output_dim, kernel_size= 1, stride= stride)
      self.bn4= nn.BatchNorm2d(output_dim)
    else:
      self.conv4= None
      
    self.squeeze= nn.AdaptiveAvgPool2d((1, 1))
    reduced_dim= output_dim// reduction
    self.excitation= nn.Sequential(nn.Linear(output_dim, reduced_dim),
                                   nn.ReLU(inplace= True),
                                   nn.Linear(reduced_dim, output_dim),
                                   nn.Sigmoid())


  def forward(self, x):
    residual= F.relu(self.bn1(self.conv1(x)))
    residual= F.relu(self.bn2(self.conv2(residual)))
    residual= self.bn3(self.conv3(residual))

    if self.conv4:
      x= self.bn4(self.conv4(x))
    
    attention= self.squeeze(residual)
    attention= attention.reshape(attention.shape[0], -1)
    attention= self.excitation(attention)
    attention= attention.reshape((attention.shape[0], attention.shape[1], 1, 1))
    
    out= residual*attention + x
    out= F.relu(out)
    return out

def make_block(input_dim, output_dim, num_blocks, intermediate_dim= 0, bottleneck= False, first_block= False):
  '''
  make blocks for both seresnet34 and seresnet50 
  bottleneck= is True for seresnet50
  first_block= #channels double in the first block, set proj= True 
  '''

  layers= []

  for i in range(num_blocks):
    if bottleneck:
      if i==0 and first_block:
        if input_dim== 64:
          #no. of channels double but no downsampling in the first block in seresnet50
          layers.append(BottleneckSEResnetBlock(input_dim, intermediate_dim, output_dim, stride= 1, proj= True)) 
        else:
          layers.append(BottleneckSEResnetBlock(input_dim, intermediate_dim, output_dim, stride= 2, proj= True))
      else:
        layers.append(BottleneckSEResnetBlock(output_dim, intermediate_dim, output_dim, stride= 1, proj= False))
    else:
      if i== 0 and first_block:
        layers.append(BasicSEResnetBlock(input_dim, output_dim, stride= 2, proj= True))
      else:
        layers.append(BasicSEResnetBlock(output_dim, output_dim, stride= 1, proj= False))
  
  return layers


class SEResNet34(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
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
    print(x.shape)
    x= self.classifier(x)
    return x


class SEResNet50(nn.Module):
  def __init__(self, input_dim):
    super().__init__()

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
    print(x.shape)
    x= self.classifier(x)
    return x


if __name__== '__main__':
    test_eg= torch.randn([1, 3, 224, 224])
    
    model1= SEResNet34(input_dim= 3)
    pred1= model1(test_eg)
    print(pred1.shape)
    
    model2= SEResNet50(input_dim= 3)
    pred= model2(test_eg)
    print(pred.shape)
