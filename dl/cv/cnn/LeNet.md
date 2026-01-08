# 简介

二十世纪的产物,使用了卷积层,结构简单,曾经是用来手写数字识别的

# 架构

input:$32*32$

## 1. 6*1*5*5卷积层

## 2. 步幅为2的2*2Avgpooling

## 3. 16*6*5*5的卷积层

## 4. 步幅为2的2*2Avgpooling

## 5. 两层全连接

## 6. 10个sigmoid,因为要分类10个数字



# code

```python
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.pool=nn.AvgPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,output_dim)
        self.softmax=nn.Softmax(dim=1)
        self.flatten=nn.Flatten()
        self.tanh=nn.Tanh()
    def forward(self, x):#32*32
        x=self.pool(self.tanh(self.conv1(x)))#->28*28->14*14
        x=self.pool(self.tanh(self.conv2(x)))#->10*10->5*5
        x=self.flatten(x)
        x=self.tanh(self.fc1(x))
        x=self.tanh(self.fc2(x))
        x=self.softmax(self.fc3(x))
        
lenet=nn.Sequential(nn.Conv2d(1,6,5),nn.Tanh(),nn.AvgPool2d(2,2),nn.Conv2d(6,16,5),nn.Tanh(),nn.AvgPool2d(2,2),nn.Flatten(),nn.Linear(400,120),nn.Tanh(),nn.Linear(120,84),nn.Tanh(),nn.Linear(84,10),nn.Softmax(dim=1))
```