# 附录B：倒向随机微分方程 (BSDE) 速成




倒向随机微分方程（Backward Stochastic Differential Equations, BSDE）是理解扩散模型反向过程的重要数学工具。虽然许多扩散模型的实践者可以在不深入BSDE理论的情况下使用这些模型，但理解BSDE能帮助我们更深刻地认识扩散模型的数学本质，特别是在连续时间框架下。



本附录将快速介绍BSDE的核心概念，重点关注与扩散模型相关的部分。我们假设读者已经熟悉附录A中的测度论和随机过程基础。




## B.1 从前向到倒向：问题的提出



### B.1.1 前向SDE回顾



在深入BSDE之前，让我们先回顾标准的（前向）随机微分方程。前向SDE描述了一个随机系统如何从已知的初始状态演化到未来：



 $$dX_t = b(t, X_t)dt + \sigma(t, X_t)dW_t, \quad X_0 = x_0$$



这个方程的特点是：



 - **因果性**：给定初始条件 $X_0 = x_0$，我们可以逐步计算未来的状态
 - **适应性**：$X_t$ 只依赖于到时刻 $t$ 为止的随机性（布朗运动的历史）
 - **信息流向**：从过去流向未来，这符合我们的物理直觉
 - **马尔可夫性**：未来只依赖于当前状态，不依赖于历史路径




在扩散模型的语境中，前向SDE描述了数据如何逐渐被噪声破坏的过程。例如，在DDPM中，前向过程将清晰的图像 $x_0$ 逐步转化为纯噪声 $x_T \sim \mathcal{N}(0, I)$。


### B.1.2 倒向问题的动机



然而，在许多实际问题中，我们面临的是相反的情况：我们知道（或期望）系统在未来某个时刻的状态，需要推断现在应该采取什么行动，或者系统现在应该处于什么状态。这就引出了倒向问题。



 终端值问题的直观例子

想象你要在时刻 $T$ 到达某个目的地，但路径是随机的（受到风、交通等随机因素影响）。倒向问题问的是：



 - 现在（时刻 $t 期权定价：知道期权到期时的支付，求现在的价格
 - 对冲策略：如何动态调整投资组合以复制期权支付
 - 风险度量：从未来的损失分布推断现在的风险


 
 - **随机控制**：


 动态规划的值函数满足BSDE
 - 最优控制策略可以从BSDE的解中提取


 
 - **偏微分方程**：


 非线性PDE的概率表示
 - 高维PDE的数值求解


 
 - **扩散模型**：


 反向扩散过程的严格数学刻画
 - 分数函数（score function）的演化
 - 最优传输路径的构造


 





在扩散模型中，BSDE的视角特别有启发性。前向过程将数据破坏成噪声，而我们真正关心的是反向过程：如何从噪声重构数据。这个反向过程的数学描述自然地导向BSDE理论。具体来说：



 - 终端条件：$Y_T$ 对应于纯噪声状态的对数概率密度
 - $Y_t$ 过程：表示中间时刻的对数概率密度
 - $Z_t$ 过程：与分数函数 $\nabla \log p_t$ 密切相关，指导去噪方向



## B.2 BSDE的数学定义



### B.2.1 线性BSDE



最简单的BSDE形式是线性BSDE，它为理解一般BSDE奠定基础。让我们从积分形式开始，这样更容易理解：



 $$Y_t = \xi + \int_t^T [a(s)Y_s + b(s)Z_s + f(s)]ds - \int_t^T Z_s dW_s$$



改写成微分形式（注意负号）：



 $$\begin{cases}
 -dY_t = [a(t)Y_t + b(t)Z_t + f(t)]dt - Z_t dW_t \\
 Y_T = \xi
 \end{cases}$$



 直观理解：为什么是"倒向"？

考虑一个具体例子：你知道一支股票在时刻 $T$ 的价格是 $\xi$（可能是随机的），你想知道它在更早时刻 $t 在金融中：对冲组合中的股票数量
 - 在控制中：最优控制策略
 - 在扩散模型中：与分数函数 $\nabla \log p_t$ 相关


 
 - **$\xi$**：终端条件，$\mathcal{F}_T$-可测（即在时刻 $T$ 已知）
 - **$a(t), b(t), f(t)$**：系数过程，描述系统的动态特性




#### 为什么需要 $Z_t$？三个视角



**1. 数学视角（鞅表示定理）**


根据鞅表示定理，任何关于布朗运动的平方可积鞅 $M_t$ 都可以唯一表示为：


 $$M_t = M_0 + \int_0^t H_s dW_s$$


在BSDE中，$Y_t - \int_0^t [a(s)Y_s + b(s)Z_s + f(s)]ds$ 是一个鞅，因此必然存在某个过程使其可以表示为随机积分，这个过程就是 $Z_t$。



**2. 金融视角（完备市场）**


在Black-Scholes模型中：



 - $Y_t$ = 期权在时刻 $t$ 的价值
 - $Z_t$ = 对冲组合中需要持有的股票数量（Delta对冲）
 - 通过动态调整股票持仓 $Z_t$，可以完美复制期权的支付




**3. PDE视角（梯度信息）**


如果 $Y_t = u(t, X_t)$ 其中 $u$ 满足某个PDE，则通过Itô公式可以证明：


 $$Z_t = \sigma(t, X_t)^T \nabla_x u(t, X_t)$$


即 $Z_t$ 编码了解关于空间变量的梯度信息。




 最简单的线性BSDE例子

考虑最简单的情况：$a(t) = b(t) = 0$，$f(t) = r$（常数），则BSDE变为：


 $$-dY_t = r \, dt - Z_t dW_t, \quad Y_T = \xi$$



这个BSDE的解是：


 $$Y_t = \mathbb{E}[\xi | \mathcal{F}_t] + r(T-t)$$



物理意义：如果终端收益是 $\xi$，无风险利率是 $r$，那么时刻 $t$ 的公平价值是终端收益的条件期望加上期间的利息。



对应的 $Z_t$ 过程可以通过鞅表示定理得到。如果 $\xi = g(W_T)$，则：


 $$Z_t = \mathbb{E}[g'(W_T) | \mathcal{F}_t]$$




### B.2.2 一般（非线性）BSDE



从线性到非线性的推广开启了BSDE理论的广阔应用空间。非线性BSDE不仅是数学上的推广，更重要的是它能够刻画许多线性模型无法描述的现象。



 $$\begin{cases}
 -dY_t = f(t, Y_t, Z_t)dt - Z_t dW_t \\
 Y_T = \xi
 \end{cases}$$



这里 $f: [0,T] \times \mathbb{R} \times \mathbb{R}^d \to \mathbb{R}$ 称为**驱动函数**（driver），它决定了BSDE的特性。




#### 驱动函数的重要性


驱动函数 $f$ 就像是BSDE的"灵魂"，不同的 $f$ 对应不同的应用：



 - **$f(t,y,z) = 0$**：鞅，对应线性期望
 - **$f(t,y,z) = g(z)$**：风险度量，如期望缺口（Expected Shortfall）
 - **$f(t,y,z) = \frac{1}{2}|z|^2$**：指数效用，对应风险敏感控制
 - **$f(t,y,z) = |z|^2 - \text{div}(z)$**：扩散模型中的分数演化





 非线性的必要性：一个金融例子

考虑一个面临**流动性约束**的交易者。他不能无限制地买卖资产，交易成本随交易量非线性增长。



此时，期权的对冲成本不再是线性的，必须用非线性BSDE来描述：


 $$-dY_t = \left[\frac{\gamma}{2}|Z_t|^2 + h(t,Z_t)\right]dt - Z_t dW_t$$



其中：



 - $\gamma > 0$ 表示风险厌恶系数
 - $h(t,z)$ 表示交易成本（通常是 $z$ 的凸函数）
 - 解 $Y_t$ 表示考虑交易成本后的期权价值




这个BSDE的解通常**高于**Black-Scholes价格，差额反映了流动性成本。




 非线性的必要性：扩散模型例子

在扩散模型中，对数密度 $Y_t = \log p_t(X_t)$ 的演化遵循非线性BSDE。考虑标准的扩散过程：



 $$dX_t = -\frac{1}{2}\beta(t)X_t dt + \sqrt{\beta(t)}dW_t$$



应用Itô公式到 $Y_t = \log p_t(X_t)$，可以得到：



 $$-dY_t = \left[\|Z_t\|^2 + \frac{1}{2}\beta(t)\text{div}(Z_t) + \text{其他项}\right]dt - Z_t \cdot dW_t$$



这里的非线性项 $\|Z_t\|^2$ 至关重要：



 - 它来自于扩散项的二阶效应（Itô修正）
 - 它确保了概率密度的归一化
 - 它与分数函数的能量有关





#### 驱动函数的性质与分类



根据驱动函数的性质，BSDE可以分为几类：



 1. Lipschitz驱动

如果 $f$ 满足Lipschitz条件：


 $$|f(t,y_1,z_1) - f(t,y_2,z_2)| \leq L(|y_1-y_2| + |z_1-z_2|)$$


则BSDE存在唯一解。这是最经典的情况，由Pardoux-Peng (1990)建立。




 2. 二次增长驱动

如果 $f$ 关于 $z$ 有二次增长：


 $$|f(t,y,z)| \leq C(1 + |y| + |z|^2)$$


这类BSDE更具挑战性，但在许多应用中自然出现（如指数效用、扩散模型）。需要额外的技术处理，如BMO方法。




 3. 单调驱动

如果 $f$ 关于 $y$ 单调：


 $$(y_1 - y_2)(f(t,y_1,z) - f(t,y_2,z)) \leq 0$$


这保证了比较定理成立，在风险度量和最优控制中很重要。




#### g-期望理论



非线性BSDE引出了Peng提出的**g-期望**理论，这是对经典期望的非线性推广：



 g-期望

给定驱动函数 $g$（通常记为 $g$ 而非 $f$），定义g-期望：


 $$\mathcal{E}_g[\xi | \mathcal{F}_t] := Y_t$$


其中 $(Y,Z)$ 是BSDE的解：


 $$-dY_s = g(s,Y_s,Z_s)ds - Z_s dW_s, \quad Y_T = \xi$$



g-期望的性质依赖于 $g$ 的性质：



 - 如果 $g \equiv 0$，则 $\mathcal{E}_g[\xi] = \mathbb{E}[\xi]$（经典期望）
 - 如果 $g$ 是凸的，则 $\mathcal{E}_g$ 是凸的（风险厌恶）
 - 如果 $g$ 满足某些条件，$\mathcal{E}_g$ 可以表示模型不确定性下的稳健期望





### B.2.3 解的概念



理解BSDE解的概念需要仔细考虑几个微妙之处。与前向SDE不同，BSDE的解是一对过程 $(Y, Z)$，而不仅仅是 $Y$。



 BSDE的解（严格定义）

一对适应过程 $(Y, Z)$ 称为BSDE的解，如果：



 - **正则性条件**：


 $Y: [0,T] \times \Omega \to \mathbb{R}$ 是连续适应过程
 - $Z: [0,T] \times \Omega \to \mathbb{R}^d$ 是循序可测过程
 - $\mathbb{E}\left[\sup_{0 \leq t \leq T} |Y_t|^2\right] 
 - **积分方程**：对所有 $t \in [0,T]$，几乎必然有
 $$Y_t = \xi + \int_t^T f(s, Y_s, Z_s)ds - \int_t^T Z_s dW_s$$

 - **终端条件**：$Y_T = \xi$ 几乎必然成立






#### 为什么需要这些条件？



**1. 适应性（Adaptedness）**


$(Y_t, Z_t)$ 必须是 $\mathcal{F}_t$-适应的，这意味着它们不能"预见未来"。这是因果性的数学表述。



**2. 平方可积性**


条件 $\mathbb{E}[\int_0^T |Z_t|^2 dt] 时间离散化（Euler格式）
 - 空间离散化（有限差分、蒙特卡洛）
 - 深度学习方法（神经网络逼近）


 



## B.3 基本理论结果



### B.3.1 存在唯一性定理



 定理（Pardoux-Peng, 1990）

假设：



 - $\xi \in L^2(\mathcal{F}_T)$
 - $f$ 关于 $y, z$ 满足Lipschitz条件：
 $$|f(t,y_1,z_1) - f(t,y_2,z_2)| \leq L(|y_1-y_2| + |z_1-z_2|)$$

 - $f(t,0,0) \in L^2([0,T] \times \Omega)$



则BSDE存在唯一的平方可积解 $(Y,Z)$。




### B.3.2 比较定理



BSDE的一个重要性质是比较定理，它允许我们比较不同BSDE解的大小关系：



 比较定理

设 $(Y^1, Z^1)$ 和 $(Y^2, Z^2)$ 分别是以下两个BSDE的解：



 - BSDE 1: $-dY^1_t = f_1(t, Y^1_t, Z^1_t)dt - Z^1_t dW_t$, $Y^1_T = \xi_1$
 - BSDE 2: $-dY^2_t = f_2(t, Y^2_t, Z^2_t)dt - Z^2_t dW_t$, $Y^2_T = \xi_2$



如果 $\xi_1 \leq \xi_2$ a.s. 且 $f_1(t, y, z) \leq f_2(t, y, z)$ 对所有 $(t, y, z)$ 成立，则 $Y^1_t \leq Y^2_t$ a.s. 对所有 $t \in [0,T]$ 成立。




## B.4 BSDE与PDE的联系：Feynman-Kac公式



BSDE与偏微分方程之间存在深刻的联系，这个联系通过Feynman-Kac公式体现。


### B.4.1 线性情况



考虑线性PDE：



 $$\begin{cases}
 \frac{\partial u}{\partial t}(t,x) + \mathcal{L}u(t,x) + f(t,x) = 0 \\
 u(T,x) = g(x)
 \end{cases}$$



其中 $\mathcal{L}$ 是椭圆算子：



 $$\mathcal{L}u = b(t,x) \cdot \nabla u + \frac{1}{2}\text{Tr}[\sigma\sigma^T(t,x) \nabla^2 u]$$



 Feynman-Kac公式（线性情况）

PDE的解可以表示为：

 $$u(t,x) = \mathbb{E}\left[g(X_T^{t,x}) + \int_t^T f(s,X_s^{t,x})ds \,\Big|\, X_t = x\right]$$

其中 $X^{t,x}$ 是从时刻 $t$、位置 $x$ 出发的SDE的解。




### B.4.2 非线性情况



对于非线性PDE：



 $$\begin{cases}
 \frac{\partial u}{\partial t} + \mathcal{L}u + f(t,x,u,\sigma^T\nabla u) = 0 \\
 u(T,x) = g(x)
 \end{cases}$$



解可以通过BSDE表示：



 非线性Feynman-Kac公式

设 $(Y^{t,x}, Z^{t,x})$ 是BSDE的解：

 $$-dY_s = f(s, X_s^{t,x}, Y_s, Z_s)ds - Z_s dW_s, \quad Y_T = g(X_T^{t,x})$$

则 $u(t,x) = Y_t^{t,x}$，且 $\nabla u(t,x) = Z_t^{t,x}/\sigma(t,x)$。




## B.5 BSDE在扩散模型中的应用



### B.5.1 反向扩散过程的BSDE表示



在连续时间扩散模型中，前向过程是：



 $$dX_t = f(X_t, t)dt + g(t)dW_t$$



Anderson (1982) 证明了反向过程可以写成：



 $$dX_t = [f(X_t, t) - g^2(t)\nabla_x \log p_t(X_t)]dt + g(t)d\bar{W}_t$$



其中 $\bar{W}_t$ 是反向布朗运动。这里的关键是分数函数 $\nabla_x \log p_t(x)$。


### B.5.2 分数函数的BSDE刻画



定义 $Y_t = \log p_t(X_t)$，则可以证明 $(Y_t, Z_t)$ 满足某个BSDE，其中：



 - $Y_t$ 对应对数概率密度
 - $Z_t$ 与分数函数 $\nabla \log p_t$ 相关




# BSDE视角下的分数匹配
import torch
import torch.nn as nn

class BSDEScoreMatching(nn.Module):
 """
 使用BSDE框架的分数匹配

 核心思想：
 - Y_t 表示对数密度
 - Z_t 表示分数函数
 - 通过最小化BSDE残差来学习
 """
 def __init__(self, score_model, T=1.0):
 super().__init__()
 self.score_model = score_model
 self.T = T

 def bsde_loss(self, x_0, t):
 """
 计算BSDE形式的损失函数

 理论基础：
 如果 (Y, Z) 满足BSDE，则残差应该为0
 我们最小化这个残差来学习分数函数
 """
 # 前向扩散采样
 noise = torch.randn_like(x_0)
 x_t = self.forward_diffusion(x_0, t, noise)

 # 预测分数（对应Z过程）
 score_pred = self.score_model(x_t, t)

 # BSDE残差：这来自于Itô公式应用于 log p_t(X_t)
 # 具体形式依赖于扩散系数的选择
 residual = self.compute_bsde_residual(x_t, t, score_pred, noise)

 return residual.pow(2).mean()

 def compute_bsde_residual(self, x_t, t, score, noise):
 """计算BSDE残差（简化版本）"""
 # 这里的具体形式依赖于所选择的SDE
 # 对于标准的VP-SDE，残差与去噪分数匹配目标相关
 return score + noise # 简化示例



### B.5.3 最优传输视角


 BSDE理论还提供了扩散模型与最优传输之间的联系：



 扩散桥与BSDE

考虑连接两个分布 $\mu_0$ 和 $\mu_T$ 的扩散桥。相应的Schrödinger桥问题可以通过求解耦合的前向-倒向SDE系统来解决：



 - 前向SDE描述从 $\mu_0$ 出发的扩散
 - 倒向SDE施加终端条件 $\mu_T$
 - 最优控制（漂移）由BSDE的解给出





## B.6 数值方法



求解BSDE的数值方法对于实际应用至关重要。


### B.6.1 时间离散化



最简单的方法是Euler格式的倒向版本：




def backward_euler_bsde(terminal_condition, driver_f, dt, num_steps):
 """
 BSDE的倒向Euler格式

 Args:
 terminal_condition: 终端条件 ξ
 driver_f: 驱动函数 f(t, y, z)
 dt: 时间步长
 num_steps: 时间步数
 """
 # 初始化
 Y = [terminal_condition]
 Z = []

 # 倒向迭代
 for i in range(num_steps):
 t = (num_steps - i) * dt

 # 条件期望的近似（这里需要具体的数值方法）
 Y_prev, Z_curr = compute_conditional_expectation(
 Y[-1], t, dt, driver_f
 )

 Y.append(Y_prev)
 Z.append(Z_curr)

 return list(reversed(Y)), list(reversed(Z))



### B.6.2 深度学习方法


 现代方法使用神经网络来参数化BSDE的解：




class DeepBSDE(nn.Module):
 """
 深度BSDE求解器

 使用神经网络逼近Z过程，通过最小化终端条件误差来训练
 """
 def __init__(self, dim, hidden_dim=256):
 super().__init__()
 # Z过程的神经网络逼近器
 self.z_net = nn.Sequential(
 nn.Linear(dim + 1, hidden_dim), # x 和 t
 nn.ReLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, dim) # 输出 Z
 )

 def solve(self, x_0, T, num_steps, terminal_g):
 """
 求解BSDE

 通过前向模拟和神经网络预测Z，
 最小化终端条件误差
 """
 dt = T / num_steps
 x = x_0
 y = self.initial_value(x_0) # Y_0的初始猜测

 for i in range(num_steps):
 t = i * dt

 # 预测Z
 z = self.z_net(torch.cat([x, t.expand_as(x[:, :1])], dim=1))

 # 前向演化
 dw = torch.randn_like(x) * torch.sqrt(dt)
 x = x + self.drift(x, t) * dt + self.diffusion(t) * dw
 y = y - self.driver(t, y, z) * dt + torch.sum(z * dw, dim=1, keepdim=True)

 # 终端条件误差
 terminal_error = (y - terminal_g(x)).pow(2).mean()

 return terminal_error



## B.7 练习题




练习 B.1：线性BSDE的显式解
 考虑线性BSDE：

 $$-dY_t = (aY_t + f_t)dt - Z_t dW_t, \quad Y_T = \xi$$

其中 $a$ 是常数，$f_t$ 是确定性函数。求解 $(Y_t, Z_t)$ 的显式表达式。

使用变量替换 $\tilde{Y}_t = e^{at}Y_t$，可以得到：

 $$Y_t = e^{-a(T-t)}\mathbb{E}\left[\xi + \int_t^T e^{a(T-s)}f_s ds \,\Big|\, \mathcal{F}_t\right]$$

利用鞅表示定理，可以得到 $Z_t$ 的表达式。具体地，如果 $\xi = g(W_T)$，则：

 $$Z_t = e^{-a(T-t)}\mathbb{E}[g'(W_T) | \mathcal{F}_t]$$






练习 B.2：BSDE与热方程

证明热方程的解可以用BSDE表示。具体地，设 $u(t,x)$ 满足：

 $$\frac{\partial u}{\partial t} + \frac{1}{2}\Delta u = 0, \quad u(T,x) = g(x)$$

证明 $u(t,x) = Y_t$，其中 $Y_t$ 是某个BSDE的解。

考虑布朗运动 $X_t^x = x + W_t$，定义 $Y_t = u(t, X_t^x)$。


应用Itô公式：

 $$dY_t = \left(\frac{\partial u}{\partial t} + \frac{1}{2}\Delta u\right)dt + \nabla u \cdot dW_t = \nabla u \cdot dW_t$$

因此 $Y_t$ 满足BSDE：

 $$-dY_t = 0 \cdot dt - Z_t dW_t, \quad Y_T = g(X_T^x)$$

其中 $Z_t = -\nabla u(t, X_t^x)$。






练习 B.3：扩散模型中的BSDE

设前向扩散过程为 $dX_t = \sqrt{2}dW_t$（标准布朗运动的缩放）。



 - 写出对应的Fokker-Planck方程
 - 证明反向过程涉及分数函数 $\nabla \log p_t$
 - 将分数函数的演化写成BSDE形式

1. Fokker-Planck方程：

 $$\frac{\partial p}{\partial t} = \Delta p$$


2. 反向过程（Anderson, 1982）：

 $$dX_t = -2\nabla \log p_{T-t}(X_t)dt + \sqrt{2}d\bar{W}_t$$


3. 定义 $Y_t = \log p_{T-t}(X_t)$，$Z_t = \nabla \log p_{T-t}(X_t)$，则：

 $$-dY_t = \left(\|Z_t\|^2 - \text{div}(Z_t)\right)dt - Z_t \cdot dW_t$$

这是一个非线性BSDE，其驱动函数为 $f(z) = \|z\|^2 - \text{div}(z)$。






## 本章小结


在本附录中，我们快速介绍了BSDE的核心概念：



 - **基本定义**：BSDE是从终端条件出发的随机微分方程
 - **存在唯一性**：在Lipschitz条件下，BSDE有唯一解
 - **与PDE的联系**：通过Feynman-Kac公式连接
 - **在扩散模型中的应用**：刻画反向过程和分数函数
 - **数值方法**：从经典的Euler格式到现代的深度学习方法




BSDE理论为理解扩散模型提供了严格的数学框架，特别是在连续时间设定下。虽然实践中我们通常使用离散化的版本，但BSDE的视角帮助我们理解模型的本质，并启发新的算法设计。



要深入学习BSDE理论，推荐阅读Pardoux和Peng的原始论文，以及El Karoui等人的综述文章。对于在机器学习中的应用，可以参考E, Han和Jentzen的深度BSDE工作。