2013,引入了块的概念,被后续模型广泛使用,结构清晰,发现更深的模型比更宽的模型好

# VGG块

n个1填充3*3卷积接一个步长为2的2*2最大池化层



使用很多VGG块堆叠,能获得更深的模型,使用不同的块和超参数能获得不同的变种



# code

```python
import torch.nn as nn


class VggBlock(nn.Module):#vgg块会把长宽减半,但这是在pooling中做的,conv不会使长宽变化,第一个conv会改变通道数
    def __init__(self, input_kernel_size, output_kernel_size, n):
        super(VggBlock, self).__init__()
        layers=[]
        layers.append(nn.Conv2d(input_kernel_size,output_kernel_size,3,padding=1))
        layers.append(nn.ReLU())
        for i in range(n-1):#各个卷积层是不共享参数的,所以需要在init中就构建好
            layers.append(nn.Conv2d(output_kernel_size,output_kernel_size,3,padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2,2))
        self.model=nn.Sequential(*layers)
    def forward(self, x):
        x=self.model(x)
        return x
```