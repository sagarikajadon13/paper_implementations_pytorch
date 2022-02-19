import torch
import torch.nn as nn
import torch.nn.functional as F

#basic convolution block
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, bias= False, **kwargs):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn= nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        return F.relu(x)

#figure 5 of the paper
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        '''
        in_channels= no. of input channels
        pool_features= no. of output channels after pooling (in pooling branch)
        '''
        super().__init__()
        self.branch3x3stack= nn.Sequential(BasicConv2D(in_channels, 64, kernel_size= 1),
                                           BasicConv2D(64, 96, kernel_size= 3, padding= 1),
                                           BasicConv2D(96, 96, kernel_size= 3, padding= 1))
        
        self.branch3x3= nn.Sequential(BasicConv2D(in_channels, 48, kernel_size= 1),
                                      BasicConv2D(48, 64, kernel_size= 3, padding= 1))
        
        self.branch1x1= BasicConv2D(in_channels, 64, kernel_size= 1)
        
        self.branchpool= nn.Sequential(nn.AvgPool2d(kernel_size= 3, stride=1, padding= 1), 
                                       BasicConv2D(in_channels, pool_features, kernel_size= 1))
        
    def forward(self, x):
        out1= self.branch3x3stack(x)
        out2= self.branch1x1(x)
        out3= self.branch3x3(x)
        out4= self.branchpool(x)
        out= torch.cat([out1, out2, out3, out4], dim= 1)
        return out
    
#reduction block 
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3stack= nn.Sequential(BasicConv2D(in_channels, 64, kernel_size= 1),
                                           BasicConv2D(64, 96, kernel_size= 3, padding= 1),
                                           BasicConv2D(96, 96, kernel_size= 3, stride= 2))
        
        self.branch3x3= nn.Sequential(BasicConv2D(in_channels, 48, kernel_size= 1),
                                      BasicConv2D(48, 384, kernel_size= 3, stride= 2))
        
        self.branchpool= nn.MaxPool2d(kernel_size= 3, stride=2)
        
    def forward(self, x):
        out1= self.branch3x3stack(x)
        out2= self.branch3x3(x)
        out3= self.branchpool(x)
        out= torch.cat([out1, out2, out3], dim= 1)
        return out
    
#figure 6 in the paper
class InceptionC(nn.Module):
    def __init__(self, in_channels, channels7x7):
        super().__init__()
        c7= channels7x7

        self.branch7x7stacka= nn.Sequential(BasicConv2D(in_channels, c7, kernel_size= 1),
                                            BasicConv2D(c7, c7, kernel_size= (1, 7), padding= (0, 3)),
                                            BasicConv2D(c7, c7, kernel_size= (7, 1), padding= (3, 0)),
                                            BasicConv2D(c7, c7, kernel_size= (1, 7), padding= (0, 3)),
                                            BasicConv2D(c7, 192, kernel_size= (7, 1), padding= (3, 0)))
    
        self.branch7x7stackb= nn.Sequential(BasicConv2D(in_channels, c7, kernel_size= 1),
                                            BasicConv2D(c7, c7, kernel_size= (1, 7), padding= (0, 3)),
                                            BasicConv2D(c7, 192, kernel_size= (7, 1), padding= (3, 0)))
        
        self.branch1x1= BasicConv2D(in_channels, 192, kernel_size= 1)
        
        self.branchpool= nn.Sequential(nn.AvgPool2d(kernel_size= 3, stride=1, padding= 1), 
                                       BasicConv2D(in_channels, 192, kernel_size= 1))
        
    def forward(self, x):
        out1= self.branch7x7stacka(x)
        out2= self.branch7x7stackb(x)
        out3= self.branch1x1(x)
        out4= self.branchpool(x)
        out= torch.cat([out1, out2, out3, out4], dim= 1)
        return out

#reduction block
class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch7x7stack= nn.Sequential(BasicConv2D(in_channels, 192, kernel_size= 1),
                                           BasicConv2D(192, 192, kernel_size= (1, 7), padding= (0, 3)),
                                           BasicConv2D(192, 192, kernel_size= (7, 1), padding= (3, 0)),
                                           BasicConv2D(192, 192, kernel_size= 3, stride= 2))
        
        self.branch3x3= nn.Sequential(BasicConv2D(in_channels, 192, kernel_size= 1),
                                      BasicConv2D(192, 320, kernel_size= 3, stride= 2))
        
        self.branchpool= nn.MaxPool2d(kernel_size= 3, stride=2)
        
    def forward(self, x):
        out1= self.branch7x7stack(x)
        out2= self.branch3x3(x)
        out3= self.branchpool(x)
        out= torch.cat([out1, out2, out3], dim= 1)
        return out

#figure 7 in the paper 
class InceptionE(nn.Module):
       def __init__(self, in_channels):
           super().__init__()
           self.branch1x1= BasicConv2D(in_channels, 320, kernel_size= 1)
           
           self.branch3x3_1= BasicConv2D(in_channels, 384, kernel_size= 1)
           self.branch3x3_2a= BasicConv2D(384, 384, kernel_size= (1, 3), padding= (0, 1))
           self.branch3x3_2b= BasicConv2D(384, 384, kernel_size= (3, 1), padding= (1, 0))
           
           self.branch3x3stack_1= BasicConv2D(in_channels, 448, kernel_size= 1)
           self.branch3x3stack_2= BasicConv2D(448, 384, kernel_size= 3, padding= 1)
           self.branch3x3stack_3a= BasicConv2D(384, 384, kernel_size= (1, 3), padding= (0, 1))
           self.branch3x3stack_3b= BasicConv2D(384, 384, kernel_size= (3, 1), padding= (1, 0))
           
           self.branchpool= nn.Sequential(nn.AvgPool2d(kernel_size= 3, stride=1, padding= 1), 
                                          BasicConv2D(in_channels, 192, kernel_size= 1))
       
       def forward(self, x):
           out1= self.branch1x1(x)
           
           out2= self.branch3x3_1(x)
           out2a= self.branch3x3_2a(out2)
           out2b= self.branch3x3_2b(out2)
           
           out3= self.branch3x3stack_1(x)
           out3= self.branch3x3stack_2(out3)
           out3a= self.branch3x3stack_3a(out3)
           out3b= self.branch3x3stack_3b(out3)
           
           out4= self.branchpool(x)
           out= torch.cat([out1, out2a, out2b, out3a, out3b, out4], dim= 1)
           return out
       
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool1= nn.AvgPool2d(kernel_size= 5, stride= 3)
        
        self.conv1= BasicConv2D(in_channels, 128, kernel_size= 1)
        self.conv2= BasicConv2D(128, 768, kernel_size= 5)
        
        self.pool2= nn.AdaptiveAvgPool2d((1, 1))
        self.dropout= nn.Dropout()
        self.linear= nn.Linear(768, num_classes)
        
    def forward(self, x):
        x= self.pool1(x)
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.pool2(x)
        x= self.dropout(x)
        x= x.reshape(x.shape[0], -1)
        x= self.linear(x)
        return x
       
class InceptionV3(nn.Module):
    def __init__(self, num_classes= 1000):
        super().__init__()
        self.conv1= BasicConv2D(3, 32, kernel_size= 3, stride= 2)
        self.conv2= BasicConv2D(32, 32, kernel_size= 3)
        self.conv3= BasicConv2D(32, 64, kernel_size= 3, padding= 1)
        self.pool1= nn.MaxPool2d(kernel_size= 3, stride= 2)
        self.conv4= BasicConv2D(64, 80, kernel_size= 1)
        self.conv5= BasicConv2D(80, 192, kernel_size= 3)
        self.pool2= nn.MaxPool2d(kernel_size= 3, stride= 2)
        
        self.inceptionA1= InceptionA(192, 32)
        self.inceptionA2= InceptionA(256, 64)
        self.inceptionA3= InceptionA(288, 64)
        
        self.inceptionB= InceptionB(288)
        
        self.inceptionC1= InceptionC(768, 128)
        self.inceptionC2= InceptionC(768, 160)
        self.inceptionC3= InceptionC(768, 160)
        self.inceptionC4= InceptionC(768, 192)
        
        self.inceptionaux= InceptionAux(768, num_classes)
        self.inceptionD= InceptionD(768)
        
        self.inceptionE1= InceptionE(1280)
        self.inceptionE2= InceptionE(2048)
        
        self.avgpool= nn.AdaptiveAvgPool2d((1, 1))
        self.dropout= nn.Dropout()
        self.linear= nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        #N x 3 x 299 x 299
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.pool1(x)
        x= self.conv4(x)
        x= self.conv5(x)
        x= self.pool2(x)
        
        #N x 192 x 35 x 35
        x= self.inceptionA1(x)
        x= self.inceptionA2(x)
        x= self.inceptionA3(x)
        x= self.inceptionB(x)
        
        #N x 768 x 17 x 17
        x= self.inceptionC1(x)
        x= self.inceptionC2(x)
        x= self.inceptionC3(x)
        x= self.inceptionC4(x)
        
        #N x 768 x 17 x 17
        auxlogits= self.inceptionaux(x)
        
        x= self.inceptionD(x)
        
        #N x 1280 x 8 x 8
        x= self.inceptionE1(x)
        x= self.inceptionE2(x)
        
        #N x 2048 x 8 x 8
        x= self.avgpool(x)
        x= self.dropout(x)
        x= x.reshape(x.shape[0], -1)
        x= self.linear(x)
        
        return x, auxlogits
        


if __name__== '__main__':
    test= torch.randn(8, 3, 299, 299)
    model= InceptionV3()
    out, aux= model(test)
    print(out.shape)
    print(aux.shape)