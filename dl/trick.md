# EMA(指数移动平均)

其思想与对比学习中的momentum update很相似，他们都考虑到了深度学习中梯度下降更新模型的缺点，即更新的模型往往更适合晚训练的batch而不适合先训练的batch，因为他每次更新都是彻底的，当模型对一个批次进行反向传播计算梯度然后梯度下降后，它实际上就拟合了这个batch的数据，所以当这个时候，这个模型对这个batch的效果往往更好，而由于深度学习小批量梯度下降方法的问题，模型往往更拟合后面的batch，而EMA就是使用了一种动量式的更新方式，这类似于残差连接以及对比学习和调度器中的动量更新，它通过给上一个模型状态一定比例来保留它对原本batch的拟合，往往这个以往状态的比例会很大

# 影子参数

EMA中极其重要的概念就是**影子参数**，他会保留一份上次更新时的参数，在普通参数更新中，我们直接对模型进行梯度下降，但使用EMA之后，我们得到梯度下降后的新模型参数后，会用它来动态更新影子参数
$$
\theta_{ema}^{(t)} = \alpha \cdot \theta_{ema}^{(t-1)} + (1 - \alpha) \cdot \theta^{(t)}
$$
这里的$\theta_{ema}^{(t)}$就是影子参数，这里的$\theta^{(t)}$是正常模型参数，这里的$\alpha$是衰减率，一般是0.99，0.999，0.9999

# 使用

```python
from torch_ema import ExponentialMovingAverage


#################### 初始化 EMA #####################
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

##################### epoch中 ########################

loss = model(x).loss(y)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# 更新 EMA 参数
ema.update()
###################################################


# ---------- 推理 ----------
# 保存权重
torch.save({
    "model": model.state_dict(),
    "ema": ema.state_dict(),
}, "checkpoint.pth")

# 加载权重
checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
ema.load_state_dict(checkpoint["ema"])

# 应用 EMA 参数推理
ema.store()            # 备份原始参数
ema.copy_to()          # 把模型参数替换成 EMA 版本
model.eval()
```
