# 代码
```python
import torch
from torch.nn import Module, Linear
import torch.nn as nn


class MLP(Module):
    def __init__(self, input_dim,hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model=nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        self.model(x)
```

