# Norm

存在多个Norm方式，这些方式可按位置和作用对象进行区分

# 位置

在transformer的Attention和FFN都需要加Norm，原始的transformer中使用的是post-norm，根据Norm位置不同可分为两种

## Post-Norm

先做子层运算，再加残差，最后整体 LayerNorm。

### 特点

$x_{l+1} = \text{LN}(x_l + F(x_l))$

- 优点：残差路径里是“原始信号”，信息保真。
- 缺点：随着深度增加，容易导致梯度消失/爆炸，深层模型训练不稳定。

## Pre-Norm

$x_{l+1} = x_l + F(\text{LN}(x_l))$

先 LayerNorm 再进子层，最后加残差。

### 特点

- 优点：梯度更容易传递，深层网络更稳定。
- 缺点：残差路径中不是“纯净的原始输入”，可能影响模型表示能力；另外 PreNorm 有时在浅层表现不如 PostNorm。

# 作用对象

## BatchNorm(BN)

他会使得一个batch中所有样本的同一个维度的均值为零，方差为1，所以它**依赖batch_size**，当batch_size过小时就会效果变差

## LayerNorm(LN)

适用于rnn和transformer这种处理可变序列的模型

在可变序列问题中，一个batch的样本可能并不等长，即使我们对它进行截断和填充，但他实际长度依旧不变，所以我们使用batchNorm就会面临样本无法对齐的问题，这是我们就会做layerNorm，他能使得单个token内部均值为0，方差为1

## GroupNorm(GN)

从batch的channel独立换成了channel分组，将这个组内的联合起来了
