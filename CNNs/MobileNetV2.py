import torch
import torch.nn as nn

class InvertedLinearBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, expansion_ratio, stride, residual= True):
        super(InvertedLinearBottleneck, self).__init__()
        self.stride= stride
        self.residual= residual
        
        assert self.stride== 1 or 2
        inner_channels= input_channels* expansion_ratio
        
        self.layers= nn.Sequential(nn.Conv2d(input_channels, inner_channels, kernel_size= 1),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.ReLU6(inplace= True),
                                    
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size= 3, padding= 1, stride= stride, groups= inner_channels),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.ReLU6(inplace= True),
                                    
                                   nn.Conv2d(inner_channels, output_channels, kernel_size= 1),
                                   nn.BatchNorm2d(output_channels))
        
    def forward(self, x):
        if self.stride== 1 and self.residual:
            out= x+ self.layers(x)
        else:
            out= self.layers(x)
        return out
            

def BottleneckBlock(input_channels, output_channels, n_layers, expansion_ratio, stride):
    layers= []
    layers.append(InvertedLinearBottleneck(input_channels, output_channels, expansion_ratio, stride, residual= False))
    n_layers-= 1
    
    while n_layers> 0:
        layers.append(InvertedLinearBottleneck(output_channels, output_channels, expansion_ratio, 1))
        n_layers-= 1
    
    return nn.Sequential(*layers)

    
class MobileNetV2(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1= nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size= 3, padding= 1, stride= 2),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU6(inplace= True))
        
        self.block0= InvertedLinearBottleneck(32, 16, 1, 1, residual= False)
        self.block1= BottleneckBlock(16, 24, 2, 6, 2)
        self.block2= BottleneckBlock(24, 32, 3, 6, 2)
        self.block3= BottleneckBlock(32, 64, 4, 6, 2)
        self.block4= BottleneckBlock(64, 96, 3, 6, 1)
        self.block5= BottleneckBlock(96, 160, 3, 6, 2)
        self.block6= InvertedLinearBottleneck(160, 320, 6, 1, residual= False)
        
        self.conv2= nn.Sequential(nn.Conv2d(320, 1280, kernel_size= 1),
                                  nn.BatchNorm2d(1280),
                                  nn.ReLU6(inplace= True))
        
        self.pool= nn.AvgPool2d(kernel_size= 7)
        self.conv3= nn.Conv2d(1280, num_classes, kernel_size= 1)
    
    def forward(self, x):
        x= self.conv1(x)
        x= self.block0(x)
        x= self.block1(x)
        x= self.block2(x)
        x= self.block3(x)
        x= self.block4(x)
        x= self.block5(x)
        x= self.block6(x)
        x= self.conv2(x)
        x= self.pool(x)
        x= self.conv3(x)
        x= x.reshape(x.shape[0], -1)
        return x
        
        
if __name__== '__main__':
    test= torch.randn(10, 3, 224, 224)
    model= MobileNetV2(3, 1000)
    out= model(test)
    print(out.shape)

        
        
        
        
        
        
        
        
        
        