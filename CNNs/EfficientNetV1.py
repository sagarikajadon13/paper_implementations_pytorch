import torch
import torch.nn as nn
from math import ceil

class InvertedLinearBottleneck(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 expansion_ratio, 
                 kernel_size, 
                 stride, 
                 survival_prob= 0.8,
                 residual= True):
        
        super(InvertedLinearBottleneck, self).__init__()
        self.stride= stride
        self.residual= residual
        self.survival_prob= survival_prob
        
        assert self.stride== 1 or 2
        inner_channels= input_channels* expansion_ratio
        
        self.layers= nn.Sequential(nn.Conv2d(input_channels, inner_channels, kernel_size= 1),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.SiLU(inplace= True),
                                    
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size= kernel_size, padding= kernel_size//2, stride= stride, groups= inner_channels),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.SiLU(inplace= True),
                                    
                                   nn.Conv2d(inner_channels, output_channels, kernel_size= 1),
                                   nn.BatchNorm2d(output_channels))
        
        #squeeze and excitation
        self.squeeze= nn.AdaptiveAvgPool2d((1, 1))
        reduced_dim= output_channels// 4
        self.excitation= nn.Sequential(nn.Linear(output_channels, reduced_dim),
                                   nn.SiLU(inplace= True),
                                   nn.Linear(reduced_dim, output_channels),
                                   nn.Sigmoid())
        
        
    def forward(self, x):
        out= self.layers(x)
        
        attention= self.squeeze(out)
        attention= attention.reshape(attention.shape[0], -1)
        attention= self.excitation(attention)
        attention= attention.reshape((attention.shape[0], attention.shape[1], 1, 1))
        out= out*attention
        
        if self.stride== 1 and self.residual:
            #stochastic depth
            if self.training:
                binary_tensor= torch.rand(x.shape[0], 1, 1, 1)< self.survival_prob
                out= torch.div(out, self.survival_prob)* binary_tensor
            out+= x
            
        return out
    
def BottleneckBlock(input_channels, output_channels, n_layers, expansion_ratio, kernel_size, stride):
    layers= []
    layers.append(InvertedLinearBottleneck(input_channels, output_channels, expansion_ratio, kernel_size, stride, residual= False))
    n_layers-= 1
    
    while n_layers> 0:
        layers.append(InvertedLinearBottleneck(output_channels, output_channels, expansion_ratio, kernel_size, 1))
        n_layers-= 1
    
    return nn.Sequential(*layers)

class EfficientNetV1(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1= nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size= 3, padding= 1, stride= 2),
                                  nn.BatchNorm2d(32),
                                  nn.SiLU(inplace= True))
        
        #input_channels, output_channels, expansion_ratio, kernel_size, stride
        self.block0= InvertedLinearBottleneck(32, 16, 1, 3, 1, residual= False)
        
        #input_channels, output_channels, n_layers, expansion_ratio, kernel_size, stride
        self.block1= BottleneckBlock(16, 24, 2, 6, 3, 2)
        self.block2= BottleneckBlock(24, 40, 2, 6, 5, 2)
        self.block3= BottleneckBlock(40, 80, 3, 6, 3, 2)
        self.block4= BottleneckBlock(80, 112, 3, 6, 5, 1)
        self.block5= BottleneckBlock(112, 192, 4, 6, 5, 2)
        self.block6= InvertedLinearBottleneck(192, 320, 6, 3, 1, residual= False)
        
        self.conv2= nn.Sequential(nn.Conv2d(320, 1280, kernel_size= 1),
                                  nn.BatchNorm2d(1280),
                                  nn.SiLU(inplace= True))
        
        self.pool= nn.AvgPool2d(kernel_size= 7)
        self.conv3= nn.Conv2d(1280, num_classes, kernel_size= 1)
    
    def forward(self, x):
        x= self.conv1(x)
        print(x.shape)
        x= self.block0(x)
        print(x.shape)
        x= self.block1(x)
        print(x.shape)
        x= self.block2(x)
        print(x.shape)
        x= self.block3(x)
        print(x.shape)
        x= self.block4(x)
        print(x.shape)
        x= self.block5(x)
        print(x.shape)
        x= self.block6(x)
        print(x.shape)
        x= self.conv2(x)
        print(x.shape)
        x= self.pool(x)
        print(x.shape)
        x= self.conv3(x)
        print(x.shape)
        x= x.reshape(x.shape[0], -1)
        return x
    
if __name__== '__main__':
    test= torch.randn(10, 3, 224, 224)
    model= EfficientNetV1(3, 1000)
    out= model(test)
    print(out.shape)
