import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionLayer(nn.Module):
    '''
    1x1 transition layers to reduce the no. of feature maps before the dense blocks
    '''
    def __init__(self, input_channels, theta, bias= False, **kwargs):
        super().__init__()
        '''
        theta= reduction factor
        '''
        out_channels= int(math.floor(theta* input_channels))
        
        self.bn= nn.BatchNorm2d(input_channels)
        self.conv= nn.Conv2d(input_channels, out_channels, kernel_size= 1)
        self.pool= nn.AvgPool2d(kernel_size= 2, stride= 2)
    
    def forward(self, x):
        x= self.conv(F.relu(self.bn(x)))
        x= self.pool(x)
        return x
    
    
class BottleneckLayer(nn.Module):
    '''
    1. dense layers with additional 1x1 bottleneck layer to reduce the no. of feature maps internally
    before the 3x3 convolution.
    2. #internal feature maps= 4*growth_rate
    3. returns feature maps(= growth_rate), concatenated with the input
    '''
    def __init__(self, input_channels, growth_rate, bias= False, **kwargs):
        super().__init__()
        inner_channels= 4*growth_rate
        
        self.bn1= nn.BatchNorm2d(input_channels)
        self.bottleneck= nn.Conv2d(input_channels, inner_channels, kernel_size= 1)
        
        self.bn2= nn.BatchNorm2d(inner_channels)
        self.conv= nn.Conv2d(inner_channels, growth_rate, kernel_size= 3, padding= 1, stride= 1)
    
    def forward(self, x):
        out= self.bottleneck(F.relu(self.bn1(x)))
        out= self.conv(F.relu(self.bn2(out)))
        return torch.cat([out, x], axis= 1)
    
    
class SingleDenseLayer(nn.Module):
    '''
    dense layer without the additional bottleneck layer
    returns feature maps(= growth_rate), concatenated with the input
    '''
    def __init__(self, input_channels, growth_rate, bias= False, **kwargs):
        super().__init__()
        self.bn= nn.BatchNorm2d(input_channels)
        self.conv= nn.Conv2d(input_channels, growth_rate, kernel_size= 3, padding= 1, stride= 1)
    
    def forward(self, x):
        out= self.conv(F.relu(self.bn(x)))
        return torch.cat([out, x], axis= 1)
    
        
class DenseNet(nn.Module):
    def __init__(self, input_channels, num_classes, blocks, growth_rate= 32, bottleneck= False, reduction= False, bias= False, **kwargs):
        ''' 
        blocks= a list containing the no. of layers in each dense block
        bottleneck= include bottleneck layer or not
        reduction= reduce the no. of feature maps in transtion layers or not
        '''
        super().__init__()
        theta= 0.5 if reduction else 1
        
        self.conv1= nn.Conv2d(3, 2*growth_rate, kernel_size= 7, padding= 3, stride= 2)
        self.pool1= nn.MaxPool2d(kernel_size= 3, padding= 1, stride= 2)
        c= 2*growth_rate
        
        self.dense1= self._make_dense_block(c, growth_rate, blocks[0], bottleneck)
        c+= growth_rate* blocks[0]
        self.transition1= TransitionLayer(c, theta)
        c= int(math.floor(c*theta))
        
        self.dense2= self._make_dense_block(c, growth_rate, blocks[1], bottleneck)
        c+= growth_rate*blocks[1]
        self.transition2= TransitionLayer(c, theta)
        c= int(math.floor(c*theta))
        
        self.dense3= self._make_dense_block(c, growth_rate, blocks[2], bottleneck)
        c+= growth_rate*blocks[2]
        self.transition3= TransitionLayer(c, theta)
        c= int(math.floor(c*theta))
        
        self.dense4= self._make_dense_block(c, growth_rate, blocks[3], bottleneck)
        c+= growth_rate*blocks[3]
        self.pool2= nn.AdaptiveAvgPool2d(output_size= (1, 1))
        
        self.linear= nn.Linear(c, num_classes)
        
        
    def _make_dense_block(self, input_channels, growth_rate, n_layers, bottleneck):
        '''
        n_layers= no. of layers in each dense block
        '''
        layers= []
        channels= input_channels
        
        while n_layers>0:
            if bottleneck:
                layers.append(BottleneckLayer(channels, growth_rate))
            else:
                layers.append(SingleDenseLayer(channels, growth_rate))
            channels+= growth_rate
            n_layers-= 1
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x= self.conv1(x)
        x= self.pool1(x)
       
        x= self.dense1(x)
        x= self.transition1(x)
        
        x= self.dense2(x)
        x= self.transition2(x)
        
        x= self.dense3(x)
        x= self.transition3(x)
        
        x= self.dense4(x)
        x= self.pool2(x)
        
        x= x.reshape(x.shape[0], -1)
        x= self.linear(x)
        return x
    
    
def DensenetBC_121():
    return DenseNet(3, 1000, [6, 12, 24, 16], bottleneck= True, reduction= True)

def DensenetBC_169():
    return DenseNet(3, 1000, [6, 12, 32, 32], bottleneck= True, reduction= True)

def DensenetBC_201():
    return DenseNet(3, 1000, [6, 12, 48, 32], bottleneck= True, reduction= True)

def DensenetBC_264():
    return DenseNet(3, 1000, [6, 12, 64, 48], bottleneck= True, reduction= True)



if __name__== '__main__':
    test= torch.randn(1, 3, 224, 224)
    model= DensenetBC_121()
    out= model(test)
    print(out.shape)

    