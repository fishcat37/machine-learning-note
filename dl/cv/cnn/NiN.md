网络中的网络,因为全连接层太贵,参数量太多,而且会带来过拟合,NiN选择彻底抛弃全连接层,与后续一个选择抛弃卷积层的网络都是为了考虑某个性能而做出的权衡

# NiN块

卷积层后面接两个1*1卷积层,来取代那两层全连接层,给每个像素增加了非线性性



交替使用NiN块和步幅为2的3*3MaxPool,逐步减小长宽和增大通道数,最后以类别数的通道数给全局平均池化层获得输出



# 特点

不易过拟合,有较少的参数个数



# code

```python
import torch.nn as nn


class NiNBlock(nn.Module):
    def __init__(self, kernel_size,stride,output_kernel_size,input_kernel_size):
        super(NiNBlock, self).__init__()
        self.conv2d1=nn.Conv2d(input_kernel_size,output_kernel_size,kernel_size,stride)
        self.mlp_conv2d1=nn.Conv2d(output_kernel_size,output_kernel_size,1)
        self.mlp_conv2d2=nn.Conv2d(output_kernel_size,output_kernel_size,1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(3,2)
    def forward(self, x):
        x=self.relu(self.conv2d1(x))
        x=self.relu(self.mlp_conv2d1(x))
        x=self.pool(self.relu(self.mlp_conv2d2(x)))
        return x

input_kernel_size=3
output_kernel_size=96
kernel_size=11
stride=4

ninBlock=nn.Sequential(nn.Conv2d(input_kernel_size,output_kernel_size,kernel_size,stride),nn.ReLU(),nn.Conv2d(output_kernel_size,output_kernel_size,1),nn.ReLU(),nn.Conv2d(output_kernel_size,output_kernel_size,1),nn.ReLU(),nn.MaxPool2d(3,2))
```