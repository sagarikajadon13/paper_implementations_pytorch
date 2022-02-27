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

#output_channels, n_layers, expansion_ratio, kernel_size, stride
bottleneck_config= [[16, 1, 1, 3, 1],
                    [24, 2, 6, 3, 2],
                    [40, 2, 6, 5, 2],
                    [80, 3, 6, 3, 2],
                    [112, 3, 6, 5, 1],
                    [192, 4, 6, 5, 2],
                    [320, 1, 6, 3, 1]]

class efficientnet(nn.Module):
    def __init__(self, input_channels, num_classes, width_mult, depth_mult, dropout_rate):
        super().__init__()
        c_in= input_channels
        c_out= ceil(32* width_mult)   
        
        self.conv1= nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size= 3, padding= 1, stride= 2),
                                  nn.BatchNorm2d(c_out),
                                  nn.SiLU(inplace= True))
        
        c_in= c_out
        blocks= []
        for output_channels, n_layers, expansion_ratio, kernel_size, stride in bottleneck_config:
            c_out= ceil(output_channels* width_mult)
            layers= ceil(n_layers* depth_mult)
            
            blocks.append(BottleneckBlock(c_in, c_out, layers, expansion_ratio, kernel_size, stride))
            c_in= c_out
       
        self.bottleneckblocks= nn.Sequential(*blocks)
        c_out= 4*(ceil(1280* width_mult)//4)
        
        self.conv2= nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size= 1),
                                  nn.BatchNorm2d(c_out),
                                  nn.SiLU(inplace= True))
        
        self.pool= nn.AvgPool2d(kernel_size= 7)
        self.dropout= nn.Dropout(p= dropout_rate, inplace= True)
        self.conv3= nn.Conv2d(c_out, num_classes, kernel_size= 1)
    
    def forward(self, x):
        x= self.conv1(x)
        x= self.bottleneckblocks(x)
        x= self.conv2(x)
        print(x.shape)
        x= self.pool(x)
        x= self.dropout(x)
        x= self.conv3(x)
        x= x.reshape(x.shape[0], -1)
        return x
    
#version: width_mult, depth_mult, dropout
versions= {'b0': [1, 1, 0.2],
           'b1': [1, 1.1, 0.2],
           'b2': [1.1, 1.2, 0.3],
           'b3': [1.2, 1.4, 0.3],
           'b4': [1.4, 1.8, 0.4],
           'b5': [1.6, 2.2, 0.4],
           'b6': [1.8, 2.6, 0.5],
           'b7': [2, 3.1, 0.5]}

def EfficientNetV1(input_channels, num_classes, version= 'b0'):
    width_mult, depth_mult, dropout_rate= versions[version]
    return efficientnet(input_channels, num_classes, width_mult, depth_mult, dropout_rate)
    
if __name__== '__main__':
    test= torch.randn(10, 3, 224, 224)
    model= EfficientNetV1(3, 1000, version= 'b4')
    out= model(test)
    print(out.shape)
