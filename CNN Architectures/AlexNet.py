import torch
import torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self, output_dim):
    super(AlexNet, self).__init__()

    self.features= nn.Sequential(
        nn.Conv2d(3, 96, kernel_size= (11, 11), stride= 4),
        nn.ReLU(inplace= True),
        nn.LocalResponseNorm(size= 5, k= 2),
        nn.MaxPool2d(kernel_size=(3, 3), stride= 2),

        nn.Conv2d(96, 256, 5, padding= 2),
        nn.ReLU(inplace= True),
        nn.LocalResponseNorm(size= 5, k= 2),
        nn.MaxPool2d(kernel_size=(3, 3), stride= 2),

        nn.Conv2d(256, 384, (3, 3), padding= 1),
        nn.ReLU(inplace= True),

        nn.Conv2d(384, 384, kernel_size= (3, 3), padding= 1),
        nn.ReLU(inplace= True),

        nn.Conv2d(384, 256, kernel_size= (3, 3), padding= 1),
        nn.ReLU(inplace= True),
        nn.MaxPool2d(kernel_size=(3, 3), stride= 2)
    )

    self.classifier= nn.Sequential(
        nn.Linear(6400, 4096),
        nn.ReLU(inplace= True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace= True),
        nn.Dropout(0.5),
        nn.Linear(4096, output_dim)
    )
  
  def forward(self, x):
    x= self.features(x)
    x= x.reshape(x.shape[0], -1)
    pred= self.classifier(x)
    return pred

if __name__== '__main__':
    test= torch.randn(1, 3, 224, 224)
    model= AlexNet(output_dim=1000)
    out= model(test)
    print(out.shape)