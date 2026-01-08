2012年,由核方法重新转向深度学习的开启,更深更大,它还使用了图像的数据增强

# 改进

- 使用dropout
- 使用Relu
- 使用maxpooling

# 架构

input:$3*224*224$ 

## 1. stride4的11*11卷积

## 2. stride2的3*3MaxPool

## 3. 5*5卷积,填充2

## 4. stride2的3*3MaxPool

## 5. 堆叠3个3*3卷积,都填充1

## 6. stride2的3*3MaxPool

## 7. 两个全连接层,后面有dropout

## 8. 1000类输出



# code

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, output_dim=1000):
        super(AlexNet, self).__init__()
        self.conv2d1 = nn.Conv2d(3,96,11,4)
        self.maxpool=nn.MaxPool2d(3,2)
        self.conv2d2 = nn.Conv2d(96,256,5,padding=2)
        self.conv2d3=nn.Conv2d(256,384,3,padding=1)
        self.conv2d4=nn.Conv2d(384,384,3,padding=1)
        self.conv2d5=nn.Conv2d(384,256,3,padding=1)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(256*6*6,4096)
        self.linear2=nn.Linear(4096,4096)
        self.linear3=nn.Linear(4096,output_dim)
        self.dropout=nn.Dropout(0.5)
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.maxpool(self.relu(self.conv2d1(x)))
        x=self.maxpool(self.relu(self.conv2d2(x)))
        x=self.maxpool(self.relu(self.conv2d5(self.relu(self.conv2d4(self.relu(self.conv2d3(x)))))))
        x=self.dropout(self.relu(self.linear1(self.flatten(x))))
        x=self.dropout(self.relu(self.linear2(x)))
        return self.softmax(self.linear3(x))
    
alexnet=nn.sequential(nn.Conv2d(3,96,11,4),nn.ReLU(),nn.MaxPool2d(3,2),nn.Conv2d(95,256,5,padding=2),nn.ReLU(),nn.MaxPool2d(3,2),\
    nn.Conv2d(256,384,padding=1),nn.ReLU(),nn.conv2d(384,384,3,padding=1),nn.ReLU(),nn.Conv2d(384,256,3,padding=1),nn.ReLU(),nn.MaxPool2d(3,2),\
        nn.Flatten(),nn.Linear(256*6*6,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),\
            nn.Linear(4096,10),nn.Softmax(dim=1))
```