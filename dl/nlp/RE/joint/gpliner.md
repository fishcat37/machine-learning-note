# GPLinker关键概念

## global pointer

gplinker中应该使用3个global pointer

一个用来处理实体(此处直接使用非efficient)，他的输出logits应当是这样的`[btz, 2, seq_len, seq_len]`，其中的2是heads，分别代表了**主体(头实体head)和客体(尾实体tail)**，对于某个位置(i,j)来说，i代表了该段起始位置，j代表了该段末尾位置

还有两个(此处应当使用efficient global pointer)用来处理关系，其输出logits应当是`[btz, relation_counts, seq_len, seq_len] `的

一个用来预测一种关系下主客体的起始token位置，对于(i,j)来说，i代表主体的起始位置，j代表了客体的起始位置

一个用于预测一种关系下主客体的末尾token位置，对于(i,j)来说，i代表主体的末尾位置，j代表了客体的末尾位置

## RoPE

GlobalPointer 需要计算 **任意 token_i 和 token_j 的配对分数**，用 RoPE 后，得分矩阵本身就带有位置差的信息（i − j），所以模型能更自然地捕捉 **span 边界和关系**。

## Mask

### 下三角Mask

在识别关系中有`[seq_len, seq_len]`的二维矩阵，其中(i,j)的位置代表了从i到j的子句，所以在i>j的位置是没有实际意义的，因为这里是倒置的，不可能有实体，所以在实体识别的在global pointer的forward中需要将他mask掉，让模型不关注这部分，这往往是通过生成一个和logits一样大的01矩阵，下三角区域(不包括)对角线为1，其他位置为0，然后给这个矩阵乘以一个很大的数(1e12)再从原矩阵中减去，使得这部分在预测中被省略，这部分只需要在gplinker的forward里进行mask就可以。**下三角Mask只用在实体识别的global pointer中**，因为在关系识别中，主体可以在客体的后面。

### padding部分Mask

由于padding部分不应该算作正常部分进行实体识别和关系识别，所以在global pointer的forward中需要将这部分mask掉，通过`mask.unsqueeze(1).unsqueeze(3)`和`mask.unsqueeze(1).unsqueeze(2)`可以分别将原本[batch_size, seq_len的01mask向量扩展到logits的维度再通过广播机制就能mask掉以padding部分为开始和以padding为末尾的矩阵部分，将这部分设为**-inf**就能在后续预测中忽略这些位置，padding部分的mask还需要在**计算损失**的时候标成**inf**，这样才能在计算损失的时候忽略这部分。

### 特殊token的Mask

还有两个位置需要注意，一个是0，一个是-1，这两个位置会填充特殊token指明句首句末，所以在进行extract spoes的时候也要进行mask

## 稀疏多标签交叉熵损失函数

由于论文中的原损失函数方程(下式1)计算时容易**溢出或产生极大值**，我们往往使用它的数学等价形式(下式2)来构建损失函数,使用**log-sum-exp trick**来防止溢出
$$
\log \left( 1 + \sum_{i \in \Omega_{neg}} e^{s_i} \right) + \log \left( 1 + \sum_{j \in \Omega_{pos}} e^{-s_j} \right)
$$

$$
$\log \left( \sum_k e^{x_k} \right) = m + \log \left( \sum_k e^{x_k - m} \right), \quad m = \max_k x_k$
$$




## Efficient

在原版gplinker的判别关系的global pointer中如果增加一个类别，就得增加`[hidden_size, seq_len, seq_len]`的参数，Efficient 版的核心改动是：把原本的 **二次打分权重矩阵**，拆解为 **低秩的向量形式**，Efficient 版里，**共享一套投影 + RoPE 处理**，最后只用一个简单的线性层把 hidden 转成 `2 * num_classes` 的向量：

- 前一半表示 token 作为“起点”的打分；
- 后一半表示 token 作为“终点”的打分；

- 再通过 outer product（外积）组合，得到 (i, j) 的 span score。

这样就不用为每个关系维护一整块 `[hidden_size, seq_len, seq_len]` 参数，而是压缩到 `[hidden_size, num_classes]` 级别。
