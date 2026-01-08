![image.png](https://raw.githubusercontent.com/fishcat37/pic-bed/main/20251220142658853.png)
线性分类器的几种解释视角

![image.png](https://raw.githubusercontent.com/fishcat37/pic-bed/main/20251220153212401.png)
adam optimizer的实现，同时包含动量和RMSProp，同时为了解决第一步过大。加入了bias correction，它的基础是SGD，结合了动量和RMSProp，这两个单拎出来也是两种优化器
![image.png](https://raw.githubusercontent.com/fishcat37/pic-bed/main/20251220153420273.png)
adamw的改进，只是将正则化从计算梯度取出来，放到最后更新的时候了，因为前放的话会受学习率和其他参数影响