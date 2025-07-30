![](https://cdn.nlark.com/yuque/0/2025/png/54671003/1753244377140-c53d06d1-8d23-4119-8c38-edafb048e277.png)

# 重置门
$ r_i=sigmoid(x_i*w_1+h_{i-1}*w_2+b) $

$ \tilde{h}_t=r_i*h_{i-1} $

# 更新门
$ z_i=sigmoid(x_i*w_1+h_{i-1}*w_2+h) $

$ h_i=z_i*h_{i-1}+(1-z_i)*tanh(x_i*w_1+\tilde{h}_t*w_2+b) $

