import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self, output_dim):
    super(LeNet, self).__init__()
    self.conv1= nn.Conv2d(in_channels= 1, 
                          out_channels= 6,
                          kernel_size= 5,
                          padding= 2,
                          stride= 1)
    self.conv2= nn.Conv2d(in_channels= 6, 
                          out_channels= 16,
                          kernel_size= 5,
                          padding= 0,
                          stride= 1)
    
    self.pool= nn.MaxPool2d(kernel_size= 2, stride= 2)
    self.fc1= nn.Linear(5*5*16, 120)
    self.fc2= nn.Linear(120, 84)
    self.fc3= nn.Linear(84, output_dim)


  def forward(self, x):
    x= F.relu(self.conv1(x))
    x= self.pool(x)

    x= F.relu(self.conv2(x))
    x= self.pool(x)
   
    x= x.reshape(x.shape[0], -1)

    x= F.relu(self.fc1(x))
    x= F.relu(self.fc2(x))
    x= self.fc3(x)
    return x


if __name__== '__main__':
    test= torch.randn(1, 1, 28, 28)
    model= LeNet(10)
    out= model(test)
    print(out.shape)