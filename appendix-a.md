[← 返回目录](index.md)
 附录A
 [附录B →](appendix-b.md)



# 附录A：测度论与随机过程速成



 本附录为理解第5章（连续时间扩散模型）提供必要的数学基础。我们将快速回顾测度论的核心概念，介绍布朗运动和随机微分方程的基本知识。这不是完整的数学课程，而是为理解扩散模型所需的最小知识集。



## A.1 测度论基础



### A.1.1 为什么需要测度论？



让我们从一个简单但深刻的问题开始：**在区间[0,1]上随机选一个点，这个点恰好是有理数的概率是多少？**



你的直觉可能会说："有理数有无穷多个，所以概率应该很大。"但实际上，这个概率是0！这个反直觉的结果揭示了我们需要一个比传统概率论更精细的数学框架。



 **直观理解：**想象你在数轴上随机投掷飞镖。虽然有理数密密麻麻地分布在数轴上（任意两个不同的实数之间都有无穷多个有理数），但它们太"稀疏"了——就像在无限的海洋中撒了无限多粒沙子，但每粒沙子都是孤立的点，没有"体积"。



测度论正是为了严格处理这类"无穷"而诞生的。它不仅能处理"长度"、"面积"、"体积"，还能处理更抽象的"大小"概念。在随机过程中，我们需要测度论来：



 - 定义什么是"随机事件"（可测集）
 - 给事件赋予"概率"（测度）
 - 处理连续时间上的无穷多个随机变量
 - 严格定义条件期望和鞅




### A.1.2 σ-代数：可测量的事件



在概率论中，不是所有的子集都能被赋予概率。我们需要一个"合理"的集合族来定义哪些事件是可测的。



> **定义**
> 定义 A.1（σ-代数）

 设 $\Omega$ 是一个非空集合（样本空间）。$\Omega$ 的子集族 $\mathcal{F}$ 称为 σ-代数，如果：


 - $\Omega \in \mathcal{F}$（全集可测）
 - 若 $A \in \mathcal{F}$，则 $A^c \in \mathcal{F}$（对补运算封闭）
 - 若 $A_1, A_2, \ldots \in \mathcal{F}$，则 $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$（对可数并封闭）





σ-代数的三个条件可以理解为：



 - **条件1**：必然事件是可测的
 - **条件2**：如果事件A可测，那么"A不发生"也应该可测
 - **条件3**：如果每个事件都可测，那么"至少有一个发生"也应该可测





**例子：掷硬币的σ-代数**


考虑掷一次硬币，$\Omega = \{H, T\}$（H=正面，T=反面）


最小的σ-代数（平凡σ-代数）：$\mathcal{F}_1 = \{\emptyset, \{H, T\}\}$


最大的σ-代数（幂集）：$\mathcal{F}_2 = \{\emptyset, \{H\}, \{T\}, \{H, T\}\}$


注意：$\{\emptyset, \{H\}\}$ 不是σ-代数，因为缺少 $\{H\}^c = \{T\}$




### A.1.3 测度：给集合赋予"大小"



> **定义**
> 定义 A.2（测度）

 测度 $\mu: \mathcal{F} \rightarrow [0, \infty]$ 是满足以下条件的函数：


 - $\mu(\emptyset) = 0$（空集的测度为0）
 - 可数可加性：对于两两不交的 $A_1, A_2, \ldots \in \mathcal{F}$，
 $$\mu\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mu(A_i)$$





测度推广了我们熟悉的概念：



 - **长度**：实数轴上的Lebesgue测度，$\mu([a,b]) = b - a$
 - **面积**：平面上的二维Lebesgue测度
 - **概率**：当 $\mu(\Omega) = 1$ 时，$\mu$ 称为概率测度 $\mathbb{P}$
 - **计数**：计数测度，$\mu(A) = |A|$（集合中元素的个数）




 **关键洞察：**测度的可数可加性是其最重要的性质。它说明了为什么有理数在实数中的测度为0：


 - 每个单点集 $\{r\}$ 的Lebesgue测度为0
 - 有理数集 $\mathbb{Q} \cap [0,1]$ 是可数个单点的并
 - 由可数可加性：$\mu(\mathbb{Q} \cap [0,1]) = \sum_{r \in \mathbb{Q} \cap [0,1]} \mu(\{r\}) = \sum 0 = 0$





### A.1.4 可测函数与随机变量



> **定义**
> 定义 A.3（随机变量）

 设 $(\Omega, \mathcal{F}, \mathbb{P})$ 是概率空间。函数 $X: \Omega \rightarrow \mathbb{R}$ 称为随机变量，如果对任意 Borel 集 $B \subseteq \mathbb{R}$，有
 $$X^{-1}(B) = \{\omega \in \Omega : X(\omega) \in B\} \in \mathcal{F}$$



这个定义看起来抽象，但其核心思想很简单：**随机变量必须与我们的σ-代数兼容**。




**例子：掷骰子的随机变量**


$\Omega = \{1, 2, 3, 4, 5, 6\}$，$\mathcal{F} = 2^{\Omega}$（幂集）


定义随机变量 $X(\omega) = \omega$（点数本身）


事件"点数大于4" = $\{\omega: X(\omega) > 4\} = \{5, 6\} \in \mathcal{F}$ ✓


定义另一个随机变量 $Y(\omega) = \begin{cases} 1 & \text{if } \omega \text{ 是偶数} \\ 0 & \text{if } \omega \text{ 是奇数} \end{cases}$


$Y$ 将骰子结果映射到"奇偶性"，仍然是可测的




在连续情况下，Borel σ-代数包含了所有"常见"的集合（开区间、闭区间、单点等），所以实践中几乎所有函数都是可测的。



### A.1.5 条件期望与滤波



在随机过程中，我们经常需要基于"部分信息"进行预测。这就是条件期望的作用。



> **定义**
> 定义 A.4（滤波）

 滤波（filtration）$\{\mathcal{F}_t\}_{t \geq 0}$ 是一族递增的σ-代数：
 $$\mathcal{F}_s \subseteq \mathcal{F}_t \subseteq \mathcal{F}, \quad \forall s \leq t$$



滤波代表"信息的累积"：$\mathcal{F}_t$ 包含了到时刻 $t$ 为止的所有可观测信息。



 **直观理解：**想象你在观看一场足球比赛：


 - $\mathcal{F}_0$：比赛开始前的信息（球队阵容等）
 - $\mathcal{F}_{45}$：上半场结束时的信息（比分、黄牌等）
 - $\mathcal{F}_{90}$：全场比赛的完整信息


 随着时间推进，你知道的信息越来越多，但不会"忘记"之前的信息。




练习 A.1：构造σ-代数

设 $\Omega = \{1, 2, 3, 4\}$，事件 $A = \{1, 2\}$。



 - 构造包含 $A$ 的最小σ-代数 $\sigma(A)$
 - 如果再加入事件 $B = \{2, 3\}$，最小σ-代数 $\sigma(A, B)$ 是什么？

**解答：**


1. $\sigma(A) = \{\emptyset, \{1,2\}, \{3,4\}, \{1,2,3,4\}\}$


 构造过程：



 - 必须包含 $A = \{1,2\}$
 - 由封闭性，必须包含 $A^c = \{3,4\}$
 - 必须包含 $\emptyset$ 和 $\Omega = \{1,2,3,4\}$



2. $\sigma(A, B)$ 必须包含：



 - $A = \{1,2\}$, $B = \{2,3\}$
 - $A \cap B = \{2\}$, $A \cup B = \{1,2,3\}$
 - 以及它们的补集...



最终：$\sigma(A, B) = 2^{\Omega}$（幂集，包含所有16个子集）





## A.2 布朗运动



布朗运动是随机过程理论的基石，也是扩散模型的数学基础。让我们从物理直觉开始，逐步理解这个美妙的数学对象。



### A.2.1 从花粉到数学：布朗运动的起源



1827年，植物学家罗伯特·布朗在显微镜下观察悬浮在水中的花粉颗粒，发现它们在不停地做无规则运动。这种运动后来被称为布朗运动。1905年，爱因斯坦从理论上解释了这一现象：花粉的运动是由于水分子的随机碰撞。



 **物理直觉：**


 - 每一瞬间，花粉受到来自各个方向的分子碰撞
 - 碰撞是完全随机的，没有特定方向的偏好
 - 大量微小的随机碰撞累积成可观察的随机运动
 - 运动轨迹是连续的，但极其不规则





### A.2.2 数学定义



> **定义**
> 定义 A.5（标准布朗运动）

 随机过程 $\{B_t\}_{t \geq 0}$ 称为标准布朗运动（或维纳过程），如果：


 - **起点确定**：$B_0 = 0$ a.s.（几乎必然）
 - **独立增量**：对 $0 \leq t_1 




### A.2.4 布朗运动的深刻性质



 **基本统计性质：**


 - $\mathbb{E}[B_t] = 0$（期望始终为0）
 - $\text{Var}(B_t) = t$（方差随时间线性增长）
 - $\text{Cov}(B_s, B_t) = \min(s, t)$（协方差等于较早的时间）




 让我们推导协方差公式，这个推导揭示了布朗运动的本质：



 $$\text{Cov}(B_s, B_t) = \mathbb{E}[B_s B_t] - \mathbb{E}[B_s]\mathbb{E}[B_t] = \mathbb{E}[B_s B_t]$$



不失一般性，设 $s  **定义**
> 定义 A.6（带漂移的布朗运动）

 $$X_t = \mu t + \sigma B_t$$
 其中 $\mu$ 是漂移率，$\sigma$ 是扩散系数。



> **定义**
> 定义 A.7（几何布朗运动）

 $$S_t = S_0 \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma B_t\right)$$
 用于建模股票价格等始终为正的量。



> **定义**
> 定义 A.8（布朗桥）

 条件布朗运动 $B_t^{bridge}$，满足 $B_0^{bridge} = 0$ 和 $B_1^{bridge} = 0$。
 $$B_t^{bridge} = B_t - t B_1, \quad t \in [0, 1]$$




练习 A.2：布朗运动的性质

1. 证明：如果 $B_t$ 是标准布朗运动，则 $-B_t$ 也是标准布朗运动。


2. 证明：$W_t = \frac{1}{c}B_{c^2 t}$ 对任意 $c > 0$ 都是标准布朗运动（尺度不变性）。


3. 计算 $\mathbb{E}[B_t^4]$。

**解答：**


1. 验证四个性质：



 - $(-B)_0 = -B_0 = 0$ ✓
 - $(-B)_{t+s} - (-B)_t = -(B_{t+s} - B_t) \sim \mathcal{N}(0, s)$ ✓
 - 独立增量和连续性也保持 ✓



2. 验证 $W_t$ 的增量：

 $$W_{t+s} - W_t = \frac{1}{c}(B_{c^2(t+s)} - B_{c^2 t}) = \frac{1}{c}B_{c^2 s} \sim \mathcal{N}(0, s)$$

3. 使用 $B_t \sim \mathcal{N}(0, t)$ 和正态分布的四阶矩公式：

 $$\mathbb{E}[B_t^4] = 3(\text{Var}(B_t))^2 = 3t^2$$





## A.3 随机积分与伊藤公式



现在我们进入随机微积分的核心——如何对布朗运动进行积分？这个问题比看起来更微妙。



### A.3.1 为什么需要伊藤积分？



考虑积分 $\int_0^t B_s \, dB_s$。在普通微积分中，我们会说这等于 $\frac{1}{2}B_t^2$，但这在随机情况下是**错误的**！




**问题的根源：**


对于黎曼积分，我们用矩形近似：

 $$\int_0^t f(s) \, ds \approx \sum_{i=0}^{n-1} f(\xi_i) \Delta s_i$$

其中 $\xi_i \in [s_i, s_{i+1}]$ 可以任意选择（左端点、右端点、中点等）。


但对于 $\int_0^t B_s \, dB_s$，不同的选择会导致**不同的极限**：



 - 左端点（伊藤）：$\xi_i = s_i$ → 极限 = $\frac{1}{2}B_t^2 - \frac{1}{2}t$
 - 右端点：$\xi_i = s_{i+1}$ → 极限 = $\frac{1}{2}B_t^2 + \frac{1}{2}t$
 - 中点（Stratonovich）：$\xi_i = \frac{s_i + s_{i+1}}{2}$ → 极限 = $\frac{1}{2}B_t^2$





### A.3.2 伊藤积分的定义



> **定义**
> 定义 A.9（伊藤积分）

 对于适应过程 $f_t$（即 $f_t$ 只依赖于到时刻 $t$ 为止的信息），伊藤积分定义为：
 $$\int_0^t f_s \, dB_s = \lim_{n \to \infty} \sum_{i=0}^{n-1} f_{t_i} (B_{t_{i+1}} - B_{t_i})$$
 使用左端点 $f_{t_i}$ 确保了"不能预见未来"。



为什么选择左端点？这确保了积分的**鞅性质**：



 $$\mathbb{E}\left[\int_0^t f_s \, dB_s\right] = 0$$




 **伊藤积分的关键性质：**


 - **线性性**：$\int (af + bg) \, dB = a\int f \, dB + b\int g \, dB$
 - **伊藤等距**：$\mathbb{E}\left[\left(\int_0^t f_s \, dB_s\right)^2\right] = \mathbb{E}\left[\int_0^t f_s^2 \, ds\right]$
 - **鞅性**：如果 $\mathbb{E}[\int_0^t f_s^2 ds] 
例子：计算 $\int_0^t B_s \, dB_s$

使用分部积分的思想，但要加上"伊藤修正项"。

考虑 $f(x) = \frac{1}{2}x^2$，则 $f(B_t) = \frac{1}{2}B_t^2$。


离散近似：

 $$f(B_t) - f(B_0) = \sum_{i=0}^{n-1} [f(B_{t_{i+1}}) - f(B_{t_i})]$$

泰勒展开（保留到二阶）：

 $$f(B_{t_{i+1}}) - f(B_{t_i}) \approx f'(B_{t_i})\Delta B_i + \frac{1}{2}f''(B_{t_i})(\Delta B_i)^2$$

其中 $\Delta B_i = B_{t_{i+1}} - B_{t_i}$。


代入 $f'(x) = x$, $f''(x) = 1$：

 $$\frac{1}{2}B_t^2 = \sum_{i=0}^{n-1} B_{t_i} \Delta B_i + \frac{1}{2}\sum_{i=0}^{n-1} (\Delta B_i)^2$$

当 $n \to \infty$：



 - 第一项 → $\int_0^t B_s \, dB_s$
 - 第二项 → $\frac{1}{2}t$（二次变差）



因此：$\int_0^t B_s \, dB_s = \frac{1}{2}B_t^2 - \frac{1}{2}t$





### A.3.4 伊藤公式：随机微积分的链式法则



伊藤公式是随机微积分最重要的工具，它告诉我们如何对复合函数求微分。与普通微积分的关键区别是多了一个二阶项。



> **定义**
> 定理 A.1（伊藤公式）

 设 $X_t$ 满足 SDE：$dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dB_t$，$f(x, t) \in C^{2,1}$，则：
 $$df(X_t, t) = \left[\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}\right]dt + \sigma \frac{\partial f}{\partial x} dB_t$$




**伊藤公式的直观理解：**


普通微积分：$df = f'(x)dx$（一阶泰勒展开）


随机微积分：需要保留二阶项，因为 $(dB_t)^2 = dt$ 不是高阶无穷小！

 
 
 运算
 $dt \cdot dt$
 $dt \cdot dB_t$
 $dB_t \cdot dB_t$
 
 
 结果
 0（高阶）
 0（高阶）
 $dt$（一阶！）
 
 



### A.3.5 伊藤公式的推导思路



让我们通过一个启发式的推导理解伊藤公式为什么是这样：




 $$f(X_{t+dt}, t+dt) - f(X_t, t) \approx \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial x}dX_t + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(dX_t)^2$$




关键在于计算 $(dX_t)^2$：



 $$(dX_t)^2 = (\mu dt + \sigma dB_t)^2 = \mu^2(dt)^2 + 2\mu\sigma dt \cdot dB_t + \sigma^2(dB_t)^2$$




使用伊藤规则：



 - $(dt)^2 = 0$（高阶无穷小）
 - $dt \cdot dB_t = 0$（高阶无穷小）
 - $(dB_t)^2 = dt$（关键！）




因此 $(dX_t)^2 = \sigma^2 dt$，这就是伊藤修正项的来源。



### A.3.6 伊藤公式的应用




例子1：几何布朗运动

设 $S_t$ 满足 $dS_t = \mu S_t dt + \sigma S_t dB_t$，求 $\log S_t$ 的动态。

设 $f(x) = \log x$，则：



 - $\frac{\partial f}{\partial x} = \frac{1}{x}$
 - $\frac{\partial^2 f}{\partial x^2} = -\frac{1}{x^2}$



应用伊藤公式：

 $$d(\log S_t) = \left[0 + \mu S_t \cdot \frac{1}{S_t} + \frac{1}{2}\sigma^2 S_t^2 \cdot \left(-\frac{1}{S_t^2}\right)\right]dt + \sigma S_t \cdot \frac{1}{S_t} dB_t$$
 $$= \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma dB_t$$

积分得：$\log S_t = \log S_0 + \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t$


因此：$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]$


**注意**：指数中出现了 $-\frac{\sigma^2}{2}$ 项，这是伊藤修正！






例子2：Ornstein-Uhlenbeck过程的平方

设 $X_t$ 满足 $dX_t = -\theta X_t dt + \sigma dB_t$，求 $X_t^2$ 的SDE。

设 $f(x) = x^2$，则 $f'(x) = 2x$，$f''(x) = 2$。


应用伊藤公式：

 $$d(X_t^2) = \left[-\theta X_t \cdot 2X_t + \frac{1}{2}\sigma^2 \cdot 2\right]dt + \sigma \cdot 2X_t dB_t$$
 $$= (-2\theta X_t^2 + \sigma^2)dt + 2\sigma X_t dB_t$$

这给出了 $X_t^2$ 的动态方程。注意漂移项中包含了 $\sigma^2$，这来自伊藤修正。





### A.3.7 多维伊藤公式



对于多个布朗运动的情况，伊藤公式需要考虑所有的二阶交叉项：



> **定义**
> 定理 A.2（多维伊藤公式）

 设 $X_t^i$ 满足 $dX_t^i = \mu^i dt + \sum_j \sigma^{ij} dB_t^j$，其中 $B_t^j$ 是独立的布朗运动。
 对于 $f(x^1, \ldots, x^n, t)$：
 $$df = \frac{\partial f}{\partial t}dt + \sum_i \frac{\partial f}{\partial x^i}dX_t^i + \frac{1}{2}\sum_{i,j} \frac{\partial^2 f}{\partial x^i \partial x^j}d\langle X^i, X^j \rangle_t$$



其中二次共变差 $d\langle X^i, X^j \rangle_t = \sum_k \sigma^{ik}\sigma^{jk} dt$。




练习 A.3：伊藤公式的综合应用

1. 设 $B_t$ 是标准布朗运动，计算 $d(B_t^3)$。


2. 证明 $e^{B_t - \frac{t}{2}}$ 是鞅。


3. 设 $X_t = t B_t$，求 $dX_t$ 的表达式（提示：这是一个时变函数）。

**解答：**


1. 对 $f(x) = x^3$：



 - $f'(x) = 3x^2$, $f''(x) = 6x$
 - $d(B_t^3) = [0 + 0 + \frac{1}{2} \cdot 1 \cdot 6B_t]dt + 3B_t^2 dB_t = 3B_t dt + 3B_t^2 dB_t$



2. 设 $Y_t = e^{B_t - \frac{t}{2}}$，令 $f(x,t) = e^{x - \frac{t}{2}}$：



 - $\frac{\partial f}{\partial t} = -\frac{1}{2}e^{x - \frac{t}{2}}$
 - $\frac{\partial f}{\partial x} = e^{x - \frac{t}{2}}$
 - $\frac{\partial^2 f}{\partial x^2} = e^{x - \frac{t}{2}}$



应用伊藤公式：

 $$dY_t = \left[-\frac{1}{2} + 0 + \frac{1}{2}\right]Y_t dt + Y_t dB_t = Y_t dB_t$$

因此 $Y_t$ 是鞅（漂移项为0）。


3. 对 $f(x,t) = tx$：



 - $\frac{\partial f}{\partial t} = x = B_t$
 - $\frac{\partial f}{\partial x} = t$
 - $\frac{\partial^2 f}{\partial x^2} = 0$


 $$dX_t = d(tB_t) = B_t dt + t dB_t$$

这展示了乘积规则在随机情况下的形式。





## A.4 随机微分方程



随机微分方程（SDE）描述了受随机扰动影响的动态系统。它们是扩散模型的数学基础，让我们能够精确描述前向和反向扩散过程。



### A.4.1 什么是SDE？



一个典型的SDE具有形式：



 $$dX_t = b(X_t, t)dt + \sigma(X_t, t)dB_t$$




这可以理解为：



 - **确定性部分**：$b(X_t, t)dt$ — 漂移项，描述系统的平均行为
 - **随机部分**：$\sigma(X_t, t)dB_t$ — 扩散项，描述随机波动





**物理类比：**


想象一个在流动河水中的树叶：



 - 河水的流速 → 漂移项 $b(X_t, t)$
 - 水流的湍流扰动 → 扩散项 $\sigma(X_t, t)dB_t$
 - 树叶的轨迹 → 解 $X_t$





### A.4.2 SDE的积分形式



SDE实际上是积分方程的简写：



 $$X_t = X_0 + \int_0^t b(X_s, s)ds + \int_0^t \sigma(X_s, s)dB_s$$




第二个积分是伊藤积分，这使得求解SDE变得非平凡。



### A.4.3 存在性与唯一性



> **定义**
> 定理 A.3（存在唯一性定理）

 考虑 SDE：$dX_t = b(X_t, t)dt + \sigma(X_t, t)dB_t$，$X_0 = x_0$。
 如果系数满足：


 - **Lipschitz 条件**：存在常数 $K$ 使得
 $$|b(x,t) - b(y,t)| + |\sigma(x,t) - \sigma(y,t)| \leq K|x-y|$$
 - **线性增长条件**：存在常数 $K$ 使得
 $$|b(x,t)|^2 + |\sigma(x,t)|^2 \leq K^2(1 + |x|^2)$$


 则 SDE 存在唯一的强解。



 **条件的直观理解：**


 - **Lipschitz条件**：系数不能变化太剧烈，保证了解的唯一性
 - **线性增长条件**：系数增长不能太快，保证了解不会在有限时间内爆炸





### A.4.4 经典SDE及其解



让我们深入研究几个在理论和应用中都非常重要的SDE：



#### 1. Ornstein-Uhlenbeck (OU) 过程



> **定义**
> OU过程

 $$dX_t = -\theta X_t dt + \sigma dB_t$$
 其中 $\theta > 0$ 是回归速度，$\sigma > 0$ 是波动率。



**物理意义**：描述带有恢复力的布朗粒子，$-\theta X_t$ 项将粒子拉回原点。




推导OU过程的解

使用积分因子法。令 $Y_t = e^{\theta t} X_t$，应用伊藤公式：

 $$dY_t = e^{\theta t} dX_t + \theta e^{\theta t} X_t dt$$
 $$= e^{\theta t}(-\theta X_t dt + \sigma dB_t) + \theta e^{\theta t} X_t dt$$
 $$= \sigma e^{\theta t} dB_t$$

积分得：$Y_t = Y_0 + \sigma \int_0^t e^{\theta s} dB_s$


因此：

 $$X_t = e^{-\theta t}X_0 + \sigma e^{-\theta t}\int_0^t e^{\theta s} dB_s$$
 $$= e^{-\theta t}X_0 + \sigma \int_0^t e^{-\theta(t-s)} dB_s$$

**性质**：



 - $\mathbb{E}[X_t] = e^{-\theta t}X_0$ （指数衰减到0）
 - $\text{Var}(X_t) = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$ （收敛到 $\frac{\sigma^2}{2\theta}$）
 - 稳态分布：$X_\infty \sim \mathcal{N}(0, \frac{\sigma^2}{2\theta})$





#### 2. 几何布朗运动 (GBM)



> **定义**
> 几何布朗运动

 $$dS_t = \mu S_t dt + \sigma S_t dB_t$$
 其中 $\mu$ 是漂移率，$\sigma$ 是波动率。



**应用**：Black-Scholes模型中的股票价格模型。



**解**（使用伊藤公式对 $\log S_t$）：



 $$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]$$




 **为什么有 $-\frac{\sigma^2}{2}$ 项？**

这是伊藤修正！如果没有这一项，$\mathbb{E}[S_t] \neq S_0 e^{\mu t}$。正是这个修正保证了：

 $$\mathbb{E}[S_t] = S_0 e^{\mu t}$$



#### 3. Cox-Ingersoll-Ross (CIR) 过程



> **定义**
> CIR过程

 $$dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}dB_t$$
 条件：$2\kappa\theta > \sigma^2$ 保证 $X_t > 0$。



**特点**：用于建模利率，保证非负性。扩散项 $\sigma\sqrt{X_t}$ 使得波动率随水平变化。



### A.4.5 SDE的解法技巧



虽然大多数SDE没有解析解，但有几种常用的求解技巧：



 **常用技巧：**


 - **积分因子法**：适用于线性SDE
 - **变量替换**：通过伊藤公式简化方程
 - **Feynman-Kac公式**：将SDE与PDE联系起来
 - **数值方法**：Euler-Maruyama、Milstein等






练习 A.4：求解时变OU过程

求解 $dX_t = -\theta(t) X_t dt + \sigma(t) dB_t$，其中 $\theta(t), \sigma(t)$ 是已知函数。

定义 $\Phi(t) = \exp\left(\int_0^t \theta(s)ds\right)$，令 $Y_t = \Phi(t)X_t$。


应用伊藤公式：

 $$dY_t = \Phi'(t)X_t dt + \Phi(t)dX_t = \sigma(t)\Phi(t)dB_t$$

积分得：

 $$Y_t = Y_0 + \int_0^t \sigma(s)\Phi(s)dB_s$$

因此：

 $$X_t = \frac{1}{\Phi(t)}\left[X_0 + \int_0^t \sigma(s)\Phi(s)dB_s\right]$$
 $$= e^{-\int_0^t \theta(s)ds}X_0 + \int_0^t \sigma(s)e^{-\int_s^t \theta(u)du}dB_s$$





## A.5 与扩散模型的深层联系



现在我们终于可以理解扩散模型的数学基础了。扩散模型的核心是两个相互关联的SDE：前向过程和反向过程。



### A.5.1 前向扩散过程



扩散模型的前向过程是一个精心设计的SDE：



 $$dx = f(x, t)dt + g(t)dB_t$$




在DDPM中，选择 $f(x, t) = -\frac{1}{2}\beta(t)x$ 和 $g(t) = \sqrt{\beta(t)}$，得到：



 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dB_t$$




 **为什么这样选择？**


 - 线性漂移 $-\frac{1}{2}\beta(t)x$ 使得方程可解
 - 时变系数 $\beta(t)$ 控制扩散速度
 - 最终分布收敛到标准正态分布 $\mathcal{N}(0, I)$






关键推导：前向过程的解

证明前向SDE的解具有形式：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

这是一个时变OU过程。定义 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=0}^t \alpha_s$（离散情况）或 $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)ds\right)$（连续情况）。


使用积分因子 $\Phi(t) = 1/\sqrt{\bar{\alpha}_t}$：



 - 令 $y_t = x_t / \sqrt{\bar{\alpha}_t}$
 - 应用伊藤公式得 $dy_t = \frac{\sqrt{\beta(t)}}{\sqrt{\bar{\alpha}_t}} dB_t$
 - 积分：$y_t = y_0 + \int_0^t \frac{\sqrt{\beta(s)}}{\sqrt{\bar{\alpha}_s}} dB_s$
 - 因此：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{\bar{\alpha}_t} \int_0^t \frac{\sqrt{\beta(s)}}{\sqrt{\bar{\alpha}_s}} dB_s$



关键观察：积分 $\int_0^t \frac{\sqrt{\beta(s)}}{\sqrt{\bar{\alpha}_s}} dB_s$ 是均值为0的高斯随机变量，其方差为：

 $$\text{Var} = \int_0^t \frac{\beta(s)}{\bar{\alpha}_s} ds = \frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}$$

因此 $x_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, 1 - \bar{\alpha}_t)$，可以写成：

 $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$





### A.5.2 反向扩散过程



扩散模型的魔力在于反向过程。Anderson (1982) 证明了，如果前向过程是：



 $$dx = f(x, t)dt + g(t)dB_t$$




那么时间反演的过程（从 $t=T$ 到 $t=0$）满足：



 $$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{B}_t$$




其中 $\bar{B}_t$ 是反向时间的布朗运动，$p_t(x)$ 是 $x_t$ 的概率密度。



 **Score function 的出现！**

$\nabla_x \log p_t(x)$ 称为 score function，它指向概率密度增加最快的方向。学习这个函数是扩散模型的核心任务。




## A.6 鞅论基础



### A.6.1 鞅的定义与性质



> **定义**
> 定义 A.5（鞅）

 随机过程 $\{M_t\}_{t \geq 0}$ 称为关于滤波 $\{\mathcal{F}_t\}$ 的鞅，如果：


 - $M_t$ 是 $\mathcal{F}_t$-可测的
 - $\mathbb{E}[|M_t|]  **定义**
> 定理 A.3（鞅表示定理）

 设 $M_t$ 是关于布朗运动 $B_t$ 生成的滤波的平方可积鞅。则存在适应过程 $\phi_t$ 使得：
 $$M_t = M_0 + \int_0^t \phi_s dB_s$$



这个定理告诉我们，所有的鞅都可以表示为关于布朗运动的随机积分，这在金融数学和扩散模型理论中非常重要。



### A.6.3 Doob-Meyer 分解



任何连续半鞅 $X_t$ 都可以唯一分解为：



 $$X_t = X_0 + M_t + A_t$$



其中 $M_t$ 是局部鞅，$A_t$ 是有界变差过程。




练习 A.4：验证鞅性质

证明过程 $M_t = \int_0^t s dB_s$ 是鞅，并计算其二次变差 $\langle M \rangle_t$。

**解答：**


1. 验证鞅性质：对 $s 




## A.7 Girsanov 定理与测度变换



### A.7.1 Radon-Nikodym 导数



> **定义**
> 定义 A.6（Radon-Nikodym 导数）

 设 $\mathbb{P}$ 和 $\mathbb{Q}$ 是概率测度，如果 $\mathbb{Q} \ll \mathbb{P}$（$\mathbb{Q}$ 关于 $\mathbb{P}$ 绝对连续），则存在 $\mathcal{F}$-可测的非负随机变量 $Z$，使得：
 $$\mathbb{Q}(A) = \mathbb{E}^{\mathbb{P}}[Z \mathbf{1}_A]$$
 $Z$ 称为 Radon-Nikodym 导数，记为 $\frac{d\mathbb{Q}}{d\mathbb{P}}$。



### A.7.2 Girsanov 定理



> **定义**
> 定理 A.4（Girsanov 定理）

 设 $\theta_t$ 是适应过程，满足 Novikov 条件。定义：
 $$Z_t = \exp\left(-\int_0^t \theta_s dB_s - \frac{1}{2}\int_0^t \theta_s^2 ds\right)$$
 在新测度 $\mathbb{Q}$ 下（$\frac{d\mathbb{Q}}{d\mathbb{P}}\big|_{\mathcal{F}_t} = Z_t$），过程：
 $$\tilde{B}_t = B_t + \int_0^t \theta_s ds$$
 是标准布朗运动。



Girsanov 定理允许我们改变概率测度，从而改变漂移项。这在扩散模型的理论分析中非常重要。



## A.8 Fokker-Planck 方程与 Kolmogorov 方程



### A.8.1 Fokker-Planck 方程



对于 SDE $dX_t = b(X_t, t)dt + \sigma(X_t, t)dB_t$，其概率密度 $p(x, t)$ 满足 Fokker-Planck 方程（也称为前向 Kolmogorov 方程）：




 $$\frac{\partial p}{\partial t} = -\nabla \cdot (b p) + \frac{1}{2}\nabla^2 : (\sigma \sigma^T p)$$




在一维情况下：



 $$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}(b(x,t) p) + \frac{1}{2}\frac{\partial^2}{\partial x^2}(\sigma^2(x,t) p)$$




### A.8.2 后向 Kolmogorov 方程



对于函数 $u(x, t) = \mathbb{E}[f(X_T) | X_t = x]$，它满足后向 Kolmogorov 方程：




 $$\frac{\partial u}{\partial t} + b(x,t)\frac{\partial u}{\partial x} + \frac{1}{2}\sigma^2(x,t)\frac{\partial^2 u}{\partial x^2} = 0$$




边界条件：$u(x, T) = f(x)$。




练习 A.5：Ornstein-Uhlenbeck 过程的稳态分布

使用 Fokker-Planck 方程求解 OU 过程 $dX_t = -\theta X_t dt + \sigma dB_t$ 的稳态分布。

**解答：**


稳态时 $\frac{\partial p}{\partial t} = 0$，Fokker-Planck 方程变为：

 $$0 = \frac{\partial}{\partial x}(\theta x p) + \frac{\sigma^2}{2}\frac{\partial^2 p}{\partial x^2}$$

令概率流 $J = -\theta x p - \frac{\sigma^2}{2}\frac{\partial p}{\partial x} = 0$（稳态无净流）


解得：$p(x) \propto \exp\left(-\frac{\theta x^2}{\sigma^2}\right)$


归一化后得到稳态分布：$p(x) = \sqrt{\frac{\theta}{\pi \sigma^2}} \exp\left(-\frac{\theta x^2}{\sigma^2}\right)$


即 $X_\infty \sim \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$。





## A.9 随机分析的高级主题



### A.9.1 局部时与 Tanaka 公式



布朗运动在 0 点的局部时 $L_t^0$ 测量布朗运动在 0 附近花费的时间：



 $$L_t^0 = \lim_{\epsilon \to 0} \frac{1}{2\epsilon} \int_0^t \mathbf{1}_{|B_s|  **定义**
> 定理 A.5（Tanaka 公式）

 对于布朗运动 $B_t$：
 $$|B_t| = \int_0^t \text{sgn}(B_s) dB_s + L_t^0$$



### A.9.2 反射布朗运动



反射布朗运动 $|B_t|$ 在 0 处被反射。它的生成元是：



 $$\mathcal{L} = \frac{1}{2}\frac{d^2}{dx^2}, \quad x > 0$$



边界条件：$\frac{\partial u}{\partial x}(0) = 0$（Neumann 边界条件）。



### A.9.3 Bessel 过程



$n$ 维 Bessel 过程定义为 $n$ 维布朗运动的范数：$R_t = |B_t^{(n)}|$。它满足：



 $$dR_t = \frac{n-1}{2R_t}dt + dW_t$$



其中 $W_t$ 是一维布朗运动。



## A.10 数值方法



### A.10.1 Euler-Maruyama 方法



对于 SDE $dX_t = b(X_t)dt + \sigma(X_t)dB_t$，Euler-Maruyama 离散化为：



 $$X_{n+1} = X_n + b(X_n)\Delta t + \sigma(X_n)\sqrt{\Delta t} Z_n$$



其中 $Z_n \sim \mathcal{N}(0, 1)$ 独立同分布。



### A.10.2 Milstein 方法



更高阶的 Milstein 方法包含伊藤修正项：



 $$X_{n+1} = X_n + b(X_n)\Delta t + \sigma(X_n)\sqrt{\Delta t} Z_n + \frac{1}{2}\sigma(X_n)\sigma'(X_n)\Delta t(Z_n^2 - 1)$$







`import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama(x0, drift, diffusion, T, N):
 """Euler-Maruyama方法"""
 dt = T / N
 X = np.zeros(N + 1)
 X[0] = x0

 for i in range(N):
 dW = np.sqrt(dt) * np.random.randn()
 X[i+1] = X[i] + drift(X[i]) * dt + diffusion(X[i]) * dW

 return X

def milstein(x0, drift, diffusion, diffusion_prime, T, N):
 """Milstein方法"""
 dt = T / N
 X = np.zeros(N + 1)
 X[0] = x0

 for i in range(N):
 Z = np.random.randn()
 dW = np.sqrt(dt) * Z
 X[i+1] = X[i] + drift(X[i]) * dt + diffusion(X[i]) * dW + \
 0.5 * diffusion(X[i]) * diffusion_prime(X[i]) * dt * (Z**2 - 1)

 return X

# 测试：几何布朗运动
# dX_t = μX_t dt + σX_t dB_t
μ, σ = 0.1, 0.3
drift = lambda x: μ * x
diffusion = lambda x: σ * x
diffusion_prime = lambda x: σ

# 解析解
def analytical_solution(x0, t, W):
 return x0 * np.exp((μ - 0.5 * σ**2) * t + σ * W)

# 模拟参数
x0, T, N = 1.0, 1.0, 1000
t = np.linspace(0, T, N + 1)

# 多次模拟比较误差
n_simulations = 100
euler_errors = []
milstein_errors = []

for _ in range(n_simulations):
 # 生成布朗运动路径
 dW = np.sqrt(T/N) * np.random.randn(N)
 W = np.concatenate([[0], np.cumsum(dW)])

 # 真实解
 X_true = analytical_solution(x0, t, W)

 # Euler-Maruyama
 X_euler = euler_maruyama(x0, drift, diffusion, T, N)
 euler_errors.append(np.abs(X_euler[-1] - X_true[-1]))

 # Milstein
 X_milstein = milstein(x0, drift, diffusion, diffusion_prime, T, N)
 milstein_errors.append(np.abs(X_milstein[-1] - X_true[-1]))

print(f"Euler-Maruyama 平均误差: {np.mean(euler_errors):.6f}")
print(f"Milstein 平均误差: {np.mean(milstein_errors):.6f}")

# 绘制一条样本路径
plt.figure(figsize=(10, 6))
plt.plot(t, X_true, 'k-', label='解析解', linewidth=2)
plt.plot(t, X_euler, 'b--', label='Euler-Maruyama', alpha=0.7)
plt.plot(t, X_milstein, 'r:', label='Milstein', alpha=0.7)
plt.xlabel('时间 t')
plt.ylabel('X(t)')
plt.title('SDE数值方法比较')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`




## A.11 与扩散模型的深层联系



### A.11.1 Score函数与漂移项


 在扩散模型中，反向SDE的漂移项包含score函数 $\nabla_x \log p_t(x)$：



 $$dx = \left[f(x, t) - g^2(t)\nabla_x \log p_t(x)\right]dt + g(t)d\bar{B}_t$$




这里的score函数满足：



 $$\nabla_x \log p_t(x) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\mathbb{E}[\epsilon | x_t = x]$$




### A.11.2 时间反演与伴随过程



对于前向SDE，其时间反演过程（从 $t=T$ 到 $t=0$）由Anderson定理给出。这正是扩散模型反向过程的理论基础。




综合练习：推导扩散模型的反向SDE

从前向SDE $dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dB_t$ 出发，使用时间反演理论推导反向SDE。

**提示：**



 - 写出前向过程的Fokker-Planck方程
 - 使用Anderson定理，反向漂移项为：$b_{rev} = -b_{for} + \sigma^2 \nabla \log p_t$
 - 计算score函数 $\nabla \log p_t(x)$
 - 代入得到反向SDE



关键洞察：score函数编码了数据分布的信息，学习score函数等价于学习如何去噪。





## A.12 进一步学习资源



 **推荐阅读：**


 - Øksendal, B. "Stochastic Differential Equations" - SDE 的经典教材
 - Karatzas, I. & Shreve, S. "Brownian Motion and Stochastic Calculus" - 更深入的理论
 - Evans, L.C. "An Introduction to Stochastic Differential Equations" - 适合初学者






## 本章小结


在本附录中，我们快速回顾了理解连续时间扩散模型所需的数学基础：



 - 测度论提供了处理连续随机变量的严格框架
 - 布朗运动是扩散过程的基本构建块
 - 伊藤公式是随机微积分的链式法则，包含额外的二阶项
 - SDE 描述了随机系统的演化，扩散模型的前向过程就是一个 SDE



这些工具将在第5章中用于理解扩散模型的连续时间公式化，特别是 Score-based SDE 框架。