长短期记忆神经网络



![](https://cdn.nlark.com/yuque/0/2025/png/54671003/1753243130689-9e773b40-ae9d-47a8-ae18-7f7f43a3626f.png)

# 概念
1. 长期记忆(细胞状态)
2. 短期记忆
3. 遗忘门
4. 输入门
5. 输出门



# 遗忘门
$ Sigmoid(input*w_1+\textsf{short term memory}*w_2+b)*\textsf{long term memory} $

决定长期记忆的保留程度

# 输入门
$ Sigmoid(input*w_1+\textsf{short term memory}*w_2+b_1)*Tanh(input*w_3+\textsf{short term memory}*w_4+b_2)+\textsf{forget gate output} $

决定输入,输入是长期记忆和短期记忆的和,输入门控制了短期记忆

# 输出门
$ Sigmoid(input*w_1+\textsf{short term mamory}*w_2+b_1)*Tanh(\textsf{input Gate output}) $

决定输出

# sigmoid
用来控制数值比例

# tanh
用来将数值缩放到-1~1

# 双向
一般的LSTM是单向的，这意味着他无法获得完整的注意力，如果使用两个LSTM组合，每个获得一个方向就能得到双向注意力

# padding
LSTM的填充实际上是针对时间序列长度的填充，单个时间步的序列与他无关，他的填充只需要普通的0填充，但是他要知道那是填充，这样他才能在计算损失等时候不带上填充部分，所以我们需要让他知道实际的length，这就是lstm填充时需要注意的。

使用 pack_padded_sequence 和 pad_packed_sequence

前者用来指定需要学习的部分，即未填充的部分，让模型忽略填充部分

而这样就会让模型不会输出padding部分输出，如果还想要输出保证长度一致方便后续处理就要用到第二个函数，而第一个函数就需要传入lengths，而且是降序排序，除非通过enforce_sorted=False来制定不用排序，同时lengths得是CPU类型的长整型，这同样适用于gru

