## MP(马尔可夫过程)
## MRP(马尔可夫奖励过程)
假设V为某状态价值，R为状态转移的直接奖励，则：
$$
V=R+\gamma V
$$
可化为解析解
$$
V=(1-\gamma)^{-1}R
$$
由于此处要求逆，但求逆时间复杂度过高，所以一般不直接求，而是用迭代逼近
$$
V_k(s)=R(s)+\sum_{s' \in S}P(s' \mid s)V_k(s')
$$
## MDP(马尔可夫决策过程)
对于$MRP(S,R^\pi,P^\pi,\gamma)$ 策略pi及其行为的引入：
$$R^\pi(s) = \sum_{a \in A} \pi(a|s)R(s, a)$$
$$P^\pi(s'|s) = \sum_{a \in A} \pi(a|s)P(s'|s, a)$$
在MDP中我们一样不使用解析解，因为需要求逆
非确定性策略即智能体以概率分布选择行为的情况下的迭代策略：
$$
V_{k}^{\pi}(s) = \sum_{a} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in S} p(s'|s, a) V_{k-1}^{\pi}(s') \right]
$$
确定性策略下的迭代策略：
$$
V_{k}^{\pi}(s) = R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s, \pi(s)) V_{k-1}^{\pi}(s')
$$
最优策略：
$$
\pi^*(s) = \arg \max_{\pi} V^\pi(s)
$$
每个v关于a的动作-价值函数：
$$
Q^{\pi}(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{\pi}(s')
$$
其中$R(s, a)$是指的即时奖励，假如一个场景是走迷宫，规定只有出口有10价值，那么只有能到达出口的那单个状态价值组有即时奖励，所以$R(s, a)$是不会进行更新的
$$
V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s, a)
$$
## 策略迭代
每次迭代i中，我们都算出V的迭代解，然后由于V与Q间的关系，我们实际上也求出了Q，然后用Q中每个s下Q最高的a去更新a，或者用这个新的关于Q的a分布更新a，得到新的策略$\pi$ ，迭代过程公式如下：

#### 策略评估
$$
B^\pi V(s) = V_{k+1}(s) = \sum_{a \in A} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V_k(s') \right]
$$
$$
Q^{\pi_i}(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{\pi_i}(s')
$$
#### 策略改进
$$
\pi_{i+1}(s) = \arg \max_{a} Q^{\pi_i}(s, a) \ \forall s \in S
$$
### 可收敛性
$$
V^{\pi_i}(s) \le \max_{a} Q^{\pi_i}(s, a) = Q^{\pi_i}(s, \pi_{i+1}(s)) = R(s, \pi_{i+1}(s)) + \gamma \sum_{s' \in S} P(s'|s, \pi_{i+1}(s)) V^{\pi_i}(s')
$$
既每次更新后的Q总是不低于上一个策略的V，所以这说明这种优化是一种单调非递减有界函数，所以我们能保证，迭代中每种策略只出现一次，同时新出现的策略至少不比之前的差，当找到最优策略或者陷入局部循环时，会出现策略不变或者策略不变优，找到最优策略的迭代次数最多为$|A|^{|S|}$次，即总的可能的策略个数，因为迭代中策略不重复出现
## 价值迭代
在该方法中我们不直接维护策略$\pi$，我们直接基于bellman方程和他的bellman最优公式来更新V
bellman方程：
$$
V^{\pi}(s) = R^{\pi}(s) + \gamma \sum_{s' \in S} P^{\pi}(s'|s) V^{\pi}(s')
$$
bellman最优公式：
$$
BV(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s' \in S} p(s'|s, a) V(s') \right]
$$
- $\pi$：对应策略 $\pi$。
    
- $\gamma$：对应折扣因子 $\gamma$。
    
- $\sum_{s' \in S}$：对应对所有可能状态 $s'$ 的求和。
    
- $\max_{a}$：对应在所有可选动作中取最大值。
### 可收敛性
已知贝尔曼算子是一种缩放算子，对于缩放算子O，有对于任意范数i

$$
\|O(V)-O(V')\|_i<=\|V-V'\|_i
$$
对于bellman算子中，两种不同的函数估计
$$
\|BV - BV'\| \le \gamma \|V - V'\|
$$
![image.png](https://raw.githubusercontent.com/fishcat37/pic-bed/main/20260127172525217.png)

![image.png](https://raw.githubusercontent.com/fishcat37/pic-bed/main/20260127171957682.png)

在这两种方式中，我们都会循环直到策略函数和价值函数无变化