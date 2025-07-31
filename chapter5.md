[← 上一章](chapter4.md) | 第5章 / 共14章 | [下一章 →](chapter6.md)

# 第5章：连续时间扩散模型 (PDE/SDE)

到目前为止，我们学习的扩散模型都是在离散时间步上定义的。但如果我们让时间步数趋于无穷，会发生什么？答案是：我们得到了一个更强大、更灵活的数学框架——随机微分方程（Stochastic Differential Equations, SDEs）。Song等人在2021年的工作《Score-Based Generative Modeling through Stochastic Differential Equations》中，将DDPM和NCSN等模型统一在SDE的视角下，开启了连续时间生成建模的新纪元。本章将深入探讨SDE框架，理解其与离散模型的联系，并介绍其对应的反向SDE、概率流ODE和Fokker-Planck方程等核心概念。

## 5.1 从离散到连续：SDE的极限之美

### 5.1.1 离散过程的极限

想象你在拍摄一个物体从清晰逐渐模糊的过程。如果你每秒拍一张照片，得到的是一个离散的序列；但如果拍摄速度越来越快，最终你会得到一个连续的视频。扩散模型从离散到连续的转变正是这样一个过程。

让我们回顾DDPM的离散前向过程：
$x_k = \sqrt{1-\beta_k} x_{k-1} + \sqrt{\beta_k} z_{k-1}, \quad z_{k-1} \sim \mathcal{N}(0, I)$

这个过程有一个美妙的物理类比：想象一滴墨水在水中扩散。每一个时间步，墨水分子都会：
1. **保持一部分原位置**：这对应 $\sqrt{1-\beta_k} x_{k-1}$ 项，表示墨水的"惯性"
2. **加入随机扰动**：这对应 $\sqrt{\beta_k} z_{k-1}$ 项，表示分子的布朗运动

#### 系数的深层含义

为什么选择 $\sqrt{1-\beta_k}$ 和 $\sqrt{\beta_k}$ 这样的系数？这不是随意的，而是基于深刻的数学考虑：

**方差守恒原理**：假设 $x_{k-1}$ 的方差为 $\sigma^2$，而 $z_{k-1}$ 是标准正态分布（方差为1）。那么 $x_k$ 的方差为：
$$\text{Var}(x_k) = (1-\beta_k)\text{Var}(x_{k-1}) + \beta_k \cdot 1 = (1-\beta_k)\sigma^2 + \beta_k$$

当 $\sigma^2 = 1$ 时，我们得到 $\text{Var}(x_k) = 1$，方差保持不变！这种设计避免了数值不稳定：如果方差不断增长，最终会导致数值溢出；如果方差不断衰减，信号会消失在数值精度中。

**信噪比的渐进衰减**：定义信噪比（Signal-to-Noise Ratio, SNR）为：
$$\text{SNR}_k = \frac{\text{信号强度}}{\text{噪声强度}} = \frac{\bar{\alpha}_k}{1-\bar{\alpha}_k}$$

其中 $\bar{\alpha}_k = \prod_{i=1}^k (1-\beta_i)$。随着 $k$ 增加，SNR单调递减，最终趋近于0，这意味着数据信号逐渐被噪声淹没。

#### 泰勒展开与连续化

当 $\beta_k$ 很小时，过程变化缓慢，我们可以用泰勒展开来近似：
$\sqrt{1 - \beta_k} \approx 1 - \frac{\beta_k}{2} - \frac{\beta_k^2}{8} + O(\beta_k^3)$

保留一阶项，更新步骤变为：
$x_k - x_{k-1} \approx -\frac{\beta_k}{2} x_{k-1} + \sqrt{\beta_k} z_{k-1}$

💡 **直觉理解**：左边是位置的变化量，右边第一项是一个"向原点的拉力"（因为系数为负），第二项是随机扰动。这就像一个被橡皮筋拴在原点的粒子，在随机力的作用下运动。

这个近似的精度如何？让我们分析误差项：
- **二阶误差**：$O(\beta_k^2)$ 项在实际应用中通常很小。例如，如果 $\beta_k = 0.0001$（典型值），则二阶误差约为 $10^{-8}$
- **累积误差**：虽然单步误差很小，但经过 $N$ 步后，累积误差可能达到 $O(N\beta_k^2)$。这解释了为什么需要足够小的 $\beta_k$

现在进行时间的连续化。将时间区间 $[0, T]$ 分成 $N$ 份，令 $\Delta t = T/N$，并设 $\beta_k = b(t_k)\Delta t$，其中 $b(t)$ 是噪声调度函数。代入后：
$\frac{x(t_k) - x(t_{k-1})}{\Delta t} \approx -\frac{b(t_{k-1})}{2} x(t_{k-1}) + \sqrt{b(t_{k-1})} \frac{z_{k-1}}{\sqrt{\Delta t}}$

#### 白噪声的涌现

这里的关键洞察是：当 $\Delta t \to 0$ 时，
- 左边收敛到导数 $\frac{dx}{dt}$
- 右边第一项保持不变
- 右边第二项 $\frac{z_{k-1}}{\sqrt{\Delta t}}$ 看起来会爆炸！

但奇妙的是，这个"爆炸"的项正是白噪声的正确缩放。让我们深入理解这一点：

**布朗运动的构造**：考虑随机游走 $S_n = \sum_{i=1}^n X_i$，其中 $X_i$ 是独立同分布的随机变量，满足 $\mathbb{E}[X_i] = 0$，$\text{Var}(X_i) = 1$。根据中心极限定理：
$$\frac{S_n}{\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

在我们的设定中，$z_i \sim \mathcal{N}(0, 1)$ 是独立的，时间步长为 $\Delta t = T/N$。定义：
$$W(t) = \sum_{i=1}^{\lfloor t/\Delta t \rfloor} \sqrt{\Delta t} \cdot z_i$$

当 $\Delta t \to 0$ 时，这个过程收敛到标准布朗运动！

**白噪声的数学含义**：形式上，白噪声 $\xi(t) = dW_t/dt$ 是布朗运动的"导数"。虽然布朗运动几乎处处不可导，但我们可以在分布意义下理解这个导数：
- $\mathbb{E}[\xi(t)] = 0$（零均值）
- $\mathbb{E}[\xi(t)\xi(s)] = \delta(t-s)$（瞬时相关）
- 在任意有限时间区间上的积分是高斯分布

因此，$z_k / \sqrt{\Delta t}$ 的极限正是白噪声 $\xi(t)$！

#### SDE的诞生

最终，我们得到了随机微分方程（SDE）：
$dx_t = -\frac{b(t)}{2} x_t dt + \sqrt{b(t)} dW_t$

这就是DDPM在连续时间下的极限形式，被称为方差保持（Variance Preserving, VP）SDE。

🎯 **为什么叫"方差保持"？** 让我们计算方差的演化。使用Itô公式，对于 $V(t) = \mathbb{E}[||x_t||^2]$：
$$\frac{dV}{dt} = -b(t)V(t) + d \cdot b(t)$$

其中 $d$ 是数据维度。当 $V(0) = d$（标准化数据）时，稳态解为 $V(\infty) = d$，即方差保持不变！这避免了数值不稳定，是VP-SDE的一大优势。

**噪声调度函数 $b(t)$ 的选择**：
- **线性调度**：$b(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$
- **余弦调度**：$b(t) = \pi \sin(\pi t/T) / T$（更平滑的过渡）
- **对数调度**：针对高分辨率图像设计，在早期阶段噪声增长更慢

每种调度都对应着不同的扩散速度和生成质量权衡。

<details>
<summary><strong>深入探索：从离散到连续的数学严格性</strong></summary>

上述推导虽然直观，但数学上需要更严格的处理：

1. **收敛性**：需要证明离散过程 $\{x_k\}$ 在某种意义下（如弱收敛）收敛到连续过程 $\{x_t\}$
2. **唯一性**：需要证明极限SDE有唯一解
3. **正则性**：需要保证系数函数 $b(t)$ 满足某些条件（如Lipschitz连续性）

相关定理包括：
- **Donsker不变原理**：随机游走收敛到布朗运动
- **Stroock-Varadhan定理**：离散马尔可夫链收敛到扩散过程
- **Wong-Zakai逼近**：光滑随机过程逼近白噪声驱动的SDE

研究方向：
- 探索非标准缩放下的极限行为（如重尾噪声）
- 研究时间非均匀离散化的极限
- 分析数值误差的传播和累积

</details>

### 5.1.2 SDE的统一框架

SDE为我们提供了一个统一的语言来描述各种扩散模型。一个通用的前向SDE可以写成：
$dx_t = f(x_t, t) dt + g(t) dW_t$

这个方程包含两个关键组件：
- **漂移系数 $f(x_t, t)$**：描述了数据演化的确定性趋势，像是一个"力场"在引导数据的运动
- **扩散系数 $g(t)$**：控制着随机噪声的强度，决定了过程的随机性程度

#### 三大SDE家族的深入理解

> **定义：SDE家族**
> 
> **1. VP-SDE (Variance Preserving) - 方差保持型**
> 
> 对应DDPM，其形式为：
> $dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} dW_t$
> 
> 其中 $\beta(t)$ 是噪声调度函数。VP-SDE的精妙之处在于：
> - **物理直觉**：像是一个弹簧振子在粘性介质中的运动，既有回复力（$-\frac{1}{2}\beta(t)x_t$），又有随机扰动
> - **方差特性**：如果 $\mathbb{E}[x_0^Tx_0] = d$（$d$ 是数据维度），那么对于适当的 $\beta(t)$，有 $\mathbb{E}[x_t^Tx_t] \approx d$
> - **数值稳定性**：避免了数值爆炸或消失，特别适合深度网络训练
> 
> **2. VE-SDE (Variance Exploding) - 方差爆炸型**
> 
> 对应NCSN，其形式为：
> $dx_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dW_t$
> 
> 注意没有漂移项！这意味着：
> - **物理直觉**：纯粹的扩散过程，像是热传导或分子扩散
> - **方差演化**：$\mathbb{E}[||x_t||^2] = \mathbb{E}[||x_0||^2] + d\sigma^2(t)$，方差单调增长
> - **多尺度特性**：通过选择 $\sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$，可以覆盖多个噪声尺度
> 
> **3. sub-VP-SDE - 次方差保持型**
> 
> 这是VP-SDE的改进版本：
> $dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})} dW_t$
> 
> - **理论优势**：保证了精确的方差守恒，而不是近似
> - **实践意义**：在长时间演化中更稳定

#### SDE选择的艺术

选择哪种SDE并非随意，而是要考虑数据特性和计算效率：

1. **数据分布的考虑**：
   - 如果数据天然具有单位方差（如标准化后的图像），VP-SDE是自然选择
   - 如果数据分布在多个尺度上（如自然图像的多分辨率结构），VE-SDE可能更合适

2. **训练稳定性**：
   - VP-SDE通常更稳定，因为方差有界
   - VE-SDE需要仔细设计 $\sigma(t)$ 的增长速度

3. **采样效率**：
   - VP-SDE的轨迹更"直"，可能需要更少的采样步数
   - VE-SDE的轨迹更"曲"，但可能探索空间更充分

**实践经验分享**：

在实际应用中，选择SDE类型往往需要实验验证。以下是一些经验法则：

- **图像生成**：VP-SDE在大多数情况下表现良好，特别是配合余弦噪声调度
- **音频生成**：由于音频信号的动态范围大，VE-SDE可能更合适
- **3D点云**：数据分布不均匀，可以考虑自适应的SDE设计
- **分子生成**：需要保持物理约束，可能需要特殊设计的SDE

**SDE的数值求解**：

在实践中，我们需要离散化SDE来进行数值求解。最简单的是Euler-Maruyama方法：
$$x_{t+\Delta t} = x_t + f(x_t, t)\Delta t + g(t)\sqrt{\Delta t} \cdot z_t$$

其中 $z_t \sim \mathcal{N}(0, I)$。PyTorch中的实现通常使用：
- `torch.randn_like()` 生成噪声
- `torch.sqrt()` 计算平方根
- 自适应步长控制提高精度

<details>
<summary><strong>高级话题：设计新的SDE</strong></summary>

SDE的设计空间远不止VP和VE。一些前沿研究方向包括：

1. **数据适应型SDE**：
   - 根据数据的局部几何结构调整 $f$ 和 $g$
   - 例如：$f(x,t) = -\nabla U(x,t)$，其中 $U$ 是学习到的势能函数

2. **流形上的SDE**：
   - 当数据位于低维流形上时，标准SDE可能效率低下
   - 可以设计保持在流形上的SDE：$dx_t = P_x f(x,t)dt + P_x g(t)dW_t$
   - 其中 $P_x$ 是投影到切空间的算子

3. **各向异性SDE**：
   - 让扩散系数依赖于方向：$g(t) \to G(x,t)$（矩阵值函数）
   - 可以更好地适应数据的协方差结构

4. **时间反演对称性**：
   - 设计满足某种对称性的SDE，使得前向和反向过程更相似
   - 可能导致更高效的采样

PyTorch中相关的工具：
- `torch.nn.functional.normalize` - 用于方差归一化
- `torch.autograd` - 计算分数函数
- `torchdiffeq` - 求解SDE/ODE

</details>

💡 **开放问题**：
1. **最优SDE设计**：给定数据分布，是否存在某种意义下"最优"的SDE？优化目标可能包括采样效率、训练稳定性、生成质量等。
2. **SDE的组合**：能否在不同时间段使用不同的SDE（如开始用VE探索，后期用VP精调）？
3. **离散数据的SDE**：如何为文本、图等离散数据设计合适的"连续化"SDE？

<details>
<summary><strong>练习 5.1：理解SDE的极限过程</strong></summary>

1. **验证方差保持性质**：
   对于VP-SDE $dx_t = -\frac{1}{2}b(t)x_t dt + \sqrt{b(t)}dW_t$，证明当 $x_0$ 满足 $\mathbb{E}[||x_0||^2] = d$ 时，存在合适的 $b(t)$ 使得 $\mathbb{E}[||x_t||^2] \approx d$ 对所有 $t$ 成立。
   
   提示：使用Itô公式计算 $d\mathbb{E}[||x_t||^2]$。

2. **比较不同的噪声调度**：
   实现并比较三种噪声调度函数：
   - 线性：$b(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$
   - 余弦：$b(t) = \pi \sin(\pi t/T) / T$
   - 指数：$b(t) = \beta_{\min} e^{t \log(\beta_{\max}/\beta_{\min})/T}$
   
   分析它们对应的信噪比 $\text{SNR}(t)$ 的衰减曲线。

3. **探索极限行为**：
   考虑离散过程 $x_{k+1} = \sqrt{1-\beta}x_k + \sqrt{\beta}z_k$，其中 $\beta = b\Delta t$。
   - 当 $\Delta t \to 0$ 时，证明这个过程收敛到VP-SDE
   - 数值实验：对于不同的 $\Delta t$，比较离散过程和连续SDE的轨迹
   - 分析收敛速度：误差如何随 $\Delta t$ 变化？

4. **研究扩展**：
   - 如果噪声不是高斯的（如Lévy噪声），极限过程会是什么？
   - 对于非马尔可夫过程（有记忆），如何推导连续时间极限？
   - 探索分数布朗运动（fractional Brownian motion）在扩散模型中的应用

</details>

## 5.2 反向时间SDE：学习去噪

如果前向SDE描述了数据如何被噪声破坏，那么我们如何构建一个反向的过程来从噪声中恢复数据呢？这个问题的答案揭示了扩散模型的深刻数学结构。

### 5.2.1 时间反演的魔法

想象你在看一段视频：墨水在清水中扩散，从一个集中的墨滴逐渐弥漫开来。现在，如果你倒放这段视频，会看到什么？分散的墨水神奇地聚集回原点！这正是反向SDE要实现的：时间的反演。

但这里有一个关键问题：在物理世界中，扩散是不可逆的（热力学第二定律）。那么数学上如何实现这种"反熵"过程呢？答案是：我们需要额外的信息——分数函数。

> **定理：Anderson反向时间SDE (1982)**
> 
> 对于前向SDE：
> $dx = f(x, t)dt + g(t)dW_t, \quad t \in [0, T]$
> 
> 其对应的反向时间SDE（从时间 $T$ 到 $0$）为：
> $dx_t = [f(x_t, t) - g(t)^2 \nabla_{x_t} \log p_t(x_t)] dt + g(t) d\bar{W}_t$
> 
> 其中：
> - $d\bar{W}_t$ 是反向时间的布朗运动（独立于前向的 $dW_t$）
> - $\nabla_{x_t} \log p_t(x_t)$ 是时刻 $t$ 的分数函数
> - $p_t(x)$ 是前向过程在时刻 $t$ 的边缘概率密度

### 5.2.2 直觉理解：为什么需要分数？

反向SDE的漂移项可以分解为两部分：
$$\underbrace{f(x_t, t)}_{\text{原始漂移}} - \underbrace{g(t)^2 \nabla_{x_t} \log p_t(x_t)}_{\text{分数修正项}}$$

1. **原始漂移项**：如果只有这一项，时间反演后的过程仍会向同一方向演化（想象一个向下流的河流，倒放视频它还是向下流）

2. **分数修正项**：这是使过程真正反向的关键！
   - 分数 $\nabla \log p_t$ 指向概率密度增加最快的方向
   - 系数 $g(t)^2$ 确保修正强度与噪声强度匹配
   - 负号使得过程向高概率区域移动

🎯 **物理类比**：想象粒子在一个势能场中运动：
- 前向过程：粒子从势能低处（数据）滚向高处（噪声），同时受到随机扰动
- 反向过程：粒子需要知道"哪里是下坡"（分数函数），才能滚回原处

### 5.2.3 分数函数的核心地位

Anderson定理揭示了一个深刻的事实：**扩散模型的本质是学习分数函数**。这统一了看似不同的两种方法：

1. **DDPM视角**：训练网络预测噪声 $\epsilon_\theta(x_t, t)$
2. **Score Matching视角**：训练网络预测分数 $s_\theta(x_t, t) \approx \nabla \log p_t(x_t)$

它们之间的关系是：
$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

#### 推导这个关键关系

这个关系的推导基于一个关键观察：在VP-SDE下，$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，因此：

**步骤1：条件分布的分数**
给定 $x_0$，$x_t$ 的条件分布是高斯分布：
$$p(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

其对数和分数为：
$$\log p(x_t|x_0) = -\frac{||x_t - \sqrt{\bar{\alpha}_t} x_0||^2}{2(1-\bar{\alpha}_t)} + \text{const}$$
$$\nabla_{x_t} \log p(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

**步骤2：边缘分数与条件分数的关系**
利用贝叶斯定理和分数的性质：
$$\nabla_{x_t} \log p(x_t) = \nabla_{x_t} \log \int p(x_t|x_0) p(x_0) dx_0$$

使用重参数化技巧和 denoising score matching 的结果，可以证明：
$$\nabla_{x_t} \log p(x_t) = \mathbb{E}_{x_0 \sim p(x_0|x_t)}[\nabla_{x_t} \log p(x_t|x_0)]$$

当神经网络 $\epsilon_\theta$ 能够准确预测噪声时，上述期望可以近似为单点估计，从而得到了DDPM和Score Matching的等价性。

#### 分数函数的几何意义

分数函数 $\nabla \log p(x)$ 有着深刻的几何含义：

1. **梯度场视角**：分数定义了一个向量场，每一点的向量指向概率密度增加最快的方向
2. **能量视角**：如果定义能量 $E(x) = -\log p(x)$，则分数 $\nabla \log p(x) = -\nabla E(x)$ 是负能量梯度
3. **最优传输视角**：分数场定义了将任意分布传输到数据分布的最优路径

**可视化理解**：
```
低概率区域 ←←←← 分数场 →→→→ 高概率区域
    噪声                        数据
```

#### 分数函数的性质与挑战

**理论性质**：
1. **Stein恒等式**：对于光滑函数 $\phi$ 满足一定衰减条件，有
   $$\mathbb{E}_{x \sim p}[\nabla \cdot \phi(x) + \phi(x) \cdot \nabla \log p(x)] = 0$$
   这是score matching的理论基础。

2. **分数的奇异性**：在数据分布的支撑集边界，分数可能不连续甚至无穷大。这解释了为什么需要添加噪声来"平滑"分布。

3. **维度诅咒**：在高维空间中，分数函数的估计变得极其困难。扩散模型通过多尺度噪声巧妙地缓解了这个问题。

**实践挑战**：
1. **数值稳定性**：当 $t \to 0$ 时，$1-\bar{\alpha}_t \to 0$，分数可能爆炸
2. **边界效应**：真实数据往往位于低维流形上，在流形外分数定义不明确
3. **多模态分布**：分数在模态之间的低密度区域可能指向错误方向

### 5.2.4 实现细节与挑战

⚡ **实现挑战**：

1. **分数函数的参数化**：
   - 直接参数化：$s_\theta(x_t, t)$ 直接输出分数
   - 噪声参数化：$\epsilon_\theta(x_t, t)$ 预测噪声，然后转换为分数
   - 实践中噪声参数化通常更稳定

2. **时间编码**：
   - 网络需要知道当前时间 $t$ 以给出正确的分数
   - 常用方法：正弦编码、可学习的嵌入、FiLM层

3. **数值稳定性**：
   - 在 $t \approx 0$ 时，分数可能很大（概率集中）
   - 在 $t \approx T$ 时，分数接近零（接近标准正态）
   - 需要合适的归一化和数值技巧

<details>
<summary><strong>深入探索：反向SDE的推导思路</strong></summary>

Anderson定理的证明涉及高深的随机分析，但核心思想可以这样理解：

1. **Girsanov定理**：描述了如何通过改变漂移项来改变概率测度
2. **时间反演公式**：对于马尔可夫过程，存在时间反演的一般理论
3. **Doob's h-transform**：通过乘以一个正函数来构造新的马尔可夫过程

关键步骤：
- 定义反向时间过程 $\hat{x}_s = x_{T-s}$
- 使用Bayes定理计算反向转移概率
- 应用Girsanov定理得到反向SDE的形式

这个推导的美妙之处在于，它将看似不可能的任务（时间反演）转化为一个可学习的问题（估计分数函数）。

</details>

<details>
<summary><strong>练习 5.2：推导反向SDE</strong></summary>

1. **VP-SDE的反向过程**：
   给定前向VP-SDE：$dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dW_t$
   
   应用Anderson定理，写出反向SDE：
   $dx_t = [-\frac{1}{2} \beta(t) x_t - \beta(t) \nabla_{x_t} \log p_t(x_t)] dt + \sqrt{\beta(t)} d\bar{W}_t$
   
   简化为：$dx_t = \frac{1}{2} \beta(t) [x_t + 2\nabla_{x_t} \log p_t(x_t)] dt + \sqrt{\beta(t)} d\bar{W}_t$

2. **与DDPM的联系**：
   在DDPM框架下，$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。
   
   利用这个重参数化，证明：
   - $\nabla_{x_t} \log p_t(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$
   - 因此，如果 $\epsilon_\theta(x_t, t) \approx \epsilon$，则 $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$

3. **研究思路**：
   - **光滑性要求**：Anderson定理要求 $p_t$ 足够光滑。探索什么条件保证这一点
   - **各向异性扩散**：当 $g(t) = G(x,t)$ 是矩阵时，反向SDE变为：
     $dx_t = [f(x_t,t) - \nabla \cdot (G(x_t,t)G(x_t,t)^T) - G(x_t,t)G(x_t,t)^T \nabla \log p_t(x_t)]dt + G(x_t,t)d\bar{W}_t$
   - **非马尔可夫情况**：如果前向过程有记忆，反向过程会如何变化？

</details>

<details>
<summary><strong>练习 5.3：分数函数的性质探索</strong></summary>

1. **验证Stein恒等式**：
   对于标准正态分布 $p(x) = \mathcal{N}(0, I)$，验证Stein恒等式：
   $$\mathbb{E}_{x \sim p}[x \cdot \phi(x) - \nabla \cdot \phi(x)] = 0$$
   
   其中 $\phi$ 是任意光滑且增长不太快的向量场。

2. **分数函数的估计误差**：
   假设我们有分数的近似 $s_\theta(x) \approx \nabla \log p(x)$，定义Fisher散度：
   $$D_F(p_\theta || p) = \mathbb{E}_{x \sim p}[||s_\theta(x) - \nabla \log p(x)||^2]$$
   
   证明：当使用这个近似分数进行Langevin采样时，稳态分布与真实分布的KL散度受Fisher散度控制。

3. **多尺度分数匹配**：
   考虑不同噪声水平 $\{\sigma_i\}_{i=1}^L$ 下的加噪数据分布 $p_{\sigma_i}(x) = \int p(y) \mathcal{N}(x; y, \sigma_i^2 I) dy$。
   
   - 推导 $\nabla \log p_{\sigma_i}(x)$ 与原始分布的关系
   - 解释为什么需要多个噪声尺度
   - 设计一个加权方案来组合不同尺度的分数

4. **流形上的分数**：
   如果数据位于 $d$ 维空间中的 $k$ 维流形 $\mathcal{M}$ 上（$k < d$），分析：
   - 分数函数在流形上和流形外的行为
   - 如何修改分数匹配目标以适应流形结构
   - 探索"投影分数"：$P_{T_x\mathcal{M}} \nabla \log p(x)$ 的性质

</details>

## 5.3 概率流ODE：确定性的生成路径

SDE的采样过程是随机的，意味着从同一个噪声 $x_T$ 出发，每次得到的 $x_0$ 都会略有不同。这种随机性有时是优点（增加多样性），有时是缺点（难以复现、调试困难）。是否存在一种确定性的路径，也能将噪声映射到数据呢？

### 5.3.1 从随机到确定：概率流的发现

想象一条河流中的叶子。每片叶子的轨迹都是随机的（受到涡流影响），但整体的流动模式是确定的。概率流ODE捕捉的正是这种"平均流动"。

> **定义：概率流ODE**
> 
> 对于SDE：$dx_t = f(x_t, t) dt + g(t) dW_t$
> 
> 存在唯一的ODE，使得其解的分布与SDE相同：
> $dx_t = \left[f(x_t, t) - \frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)\right] dt$
> 
> 这个ODE被称为概率流（Probability Flow）ODE。注意：
> - 没有随机项 $dW_t$，完全确定性
> - 漂移项 = 原始漂移 - (1/2) × 扩散强度 × 分数
> - 边缘分布 $p_t(x)$ 与原SDE完全相同

### 5.3.2 为什么概率流ODE有效？

这个结果初看令人惊讶：随机过程和确定性过程怎么会有相同的分布演化？关键在于理解两种不同的视角：

1. **粒子视角（SDE）**：
   - 跟踪单个粒子的随机轨迹
   - 每个粒子受到随机力的影响
   - 多次运行得到不同结果

2. **流体视角（ODE）**：
   - 跟踪概率密度的演化
   - 描述"概率流体"的速度场
   - 确定性的演化规律

数学上，这两种视角通过Fokker-Planck方程联系起来。SDE和其对应的概率流ODE都满足同一个Fokker-Planck方程，因此具有相同的密度演化。

🎨 **可视化理解**：
```
SDE轨迹（多条随机路径）：        概率流ODE（确定性流线）：
    噪声                              噪声
     ↓ ～～～                          ↓
     ↓   ～～                          ↓
     ↓ ～～～                          ↓
     ↓   ～～                          ↓
    数据                             数据
```

### 5.3.3 概率流ODE的推导直觉

概率流ODE中的修正项 $-\frac{1}{2}g(t)^2\nabla\log p_t$ 从何而来？这里有一个优美的解释：

#### 从Fokker-Planck方程出发

考虑前向SDE：$dx_t = f(x_t, t)dt + g(t)dW_t$

其对应的Fokker-Planck方程描述了概率密度的演化：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \Delta p_t$$

现在，我们寻找一个ODE $dx_t = v(x_t, t)dt$，使得其密度演化也满足同样的Fokker-Planck方程。对于ODE，密度演化由连续性方程描述：
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (v p_t) = 0$$

比较两个方程，我们需要：
$$-\nabla \cdot (v p_t) = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \Delta p_t$$

使用恒等式 $\Delta p_t = \nabla \cdot (\nabla p_t) = \nabla \cdot (p_t \nabla \log p_t)$，得到：
$$v p_t = f p_t - \frac{1}{2}g(t)^2 p_t \nabla \log p_t$$

因此：
$$v(x_t, t) = f(x_t, t) - \frac{1}{2}g(t)^2 \nabla \log p_t(x_t)$$

这就是概率流ODE的速度场！

#### 物理直觉：扩散引起的漂移

在SDE中，随机项 $g(t)dW_t$ 造成了两种效应：

1. **扩散效应**：使分布变宽
2. **漂移效应**：由于扩散的不均匀性产生的净流动

让我们通过一个具体例子理解漂移效应：

**例子：一维高斯分布**
考虑密度 $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-x^2/(2\sigma^2)}$，其分数为：
$$\nabla \log p(x) = -\frac{x}{\sigma^2}$$

在 $x > 0$ 区域：
- 左边（靠近原点）的密度更高
- 扩散使粒子向两边随机移动
- 但由于左边密度高，更多粒子从左边扩散过来
- 净效果：粒子向右漂移

这个净漂移正是由 $-\frac{1}{2}g^2 \nabla \log p$ 项描述的！当 $x > 0$ 时，$\nabla \log p < 0$，所以漂移方向为正，与直觉一致。

#### 概率流ODE的深层含义

概率流ODE揭示了一个深刻的事实：**随机性可以被确定性的向量场完全捕捉**。这个向量场不仅包含了原始的漂移，还包含了由于随机扩散的不均匀性产生的"有效漂移"。

**数学视角**：概率流ODE定义了一个保测度的流（measure-preserving flow），将初始分布 $p_0$ 传输到最终分布 $p_T$。这与最优传输理论有深刻联系。

**计算视角**：概率流ODE提供了一种确定性的采样方法，这带来了许多优势：

1. **确定性采样**：从一个 $x_T$ 出发，总能得到完全相同的 $x_0$。这对于需要可复现生成的任务非常有用。

2. **更快的采样**：作为ODE，我们可以使用各种现成的高阶数值求解器：
   - **Euler法**：一阶精度，最简单
   - **Heun法**：二阶精度，需要两次函数评估
   - **RK4**：四阶精度，经典选择
   - **自适应求解器**：如Dormand-Prince，自动调整步长
   
   使用高阶求解器，可以用比SDE求解器少得多的步数（例如20-50步 vs 1000步）得到高质量的样本。

3. **精确的似然计算**：通过瞬时变量变换公式（instantaneous change of variables），可以精确计算对数似然：
   $$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla \cdot v(x_t, t) dt$$
   
   其中轨迹 $\{x_t\}$ 由概率流ODE生成。

#### 概率流ODE与神经ODE的联系

概率流ODE可以看作是一种特殊的神经ODE（Neural ODE），其中：
- 速度场 $v(x,t)$ 由神经网络参数化
- 网络通过预测分数函数来隐式定义速度场
- 整个系统可以端到端训练

这建立了扩散模型与连续归一化流（Continuous Normalizing Flows）的桥梁，开启了许多研究方向。

🌟 **理论空白**：SDE和ODE提供了两种不同的采样路径。SDE路径是随机的、高维的，而ODE路径是确定性的、低维的。这两种路径的几何性质有何不同？它们在数据流形上是如何移动的？理解这一点可能有助于设计出更优的采样算法。

<details>
<summary><strong>练习 5.4：概率流ODE的性质与应用</strong></summary>

1. **验证概率流ODE保持边缘分布**：
   对于VP-SDE和其对应的概率流ODE，验证它们在任意时刻 $t$ 的边缘分布 $p_t(x)$ 相同。
   
   提示：证明两者满足相同的Fokker-Planck方程。

2. **似然计算的实现**：
   使用概率流ODE的瞬时变量变换公式：
   $$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla \cdot v(x_t, t) dt$$
   
   其中 $v(x,t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla \log p_t(x)$。
   
   - 推导 $\nabla \cdot v(x,t)$ 的表达式
   - 说明如何使用神经网络计算这个散度
   - 讨论计算复杂度和数值稳定性

3. **比较不同的ODE求解器**：
   实现并比较以下求解器在概率流ODE上的表现：
   - Euler法（一阶）
   - Heun法（二阶）
   - RK4（四阶）
   
   分析：
   - 不同步数下的生成质量
   - 计算时间 vs 质量的权衡
   - 数值误差的累积

4. **SDE与ODE路径的几何分析**：
   对于简单的2D数据分布（如双模态高斯混合），可视化并分析：
   - SDE的多条随机轨迹
   - ODE的确定性流线
   - 两者在穿越低密度区域时的行为差异
   - 轨迹的曲率和长度统计

5. **研究扩展：最优传输视角**：
   概率流ODE定义了一个传输映射 $T: x_T \mapsto x_0$。探索：
   - 这个映射是否是某种意义下的"最优"？
   - 与Monge-Kantorovich最优传输问题的联系
   - 如何设计具有最优传输性质的新型ODE？

</details>

## 5.4 Fokker-Planck方程：从粒子到密度的演化

SDE和ODE描述了单个数据点（粒子）的轨迹。如果我们想从宏观上描述整个概率密度 $p_t(x)$ 的演化，就需要偏微分方程（Partial Differential Equation, PDE）的语言，这就是Fokker-Planck方程。

### 5.4.1 Fokker-Planck方程的物理直觉

想象一大群粒子在流体中运动。每个粒子既受到确定性的流动（漂移），又受到随机的分子碰撞（扩散）。Fokker-Planck方程描述的正是这群粒子的密度如何随时间演化。

> **定义：Fokker-Planck方程**
> 对于一个SDE $dx = f(x,t)dt + g(t)dW_t$，其概率密度 $p_t(x)$ 的演化遵循Fokker-Planck方程：
> $$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (f(x,t)p_t(x)) + \frac{1}{2} g(t)^2 \Delta p_t(x)$$
> 其中 $\nabla \cdot$ 是散度算子，$\Delta$ 是拉普拉斯算子。

让我们深入理解这个方程的每一项：

#### 第一项：漂移输运 $-\nabla \cdot (f(x,t)p_t(x))$

这项描述了由确定性漂移 $f(x,t)$ 引起的概率流动：
- $f(x,t)p_t(x)$ 是概率流密度（probability flux）
- $\nabla \cdot$ 计算流的散度，即净流出量
- 负号表示：流出导致密度减少

**物理类比**：想象河流中的染料。水流（$f$）携带染料（$p$）移动，某处的染料浓度变化取决于流入和流出的差额。

#### 第二项：扩散平滑 $\frac{1}{2}g(t)^2 \Delta p_t(x)$

这项描述了随机扩散对密度的影响：
- $\Delta p_t = \sum_i \frac{\partial^2 p_t}{\partial x_i^2}$ 衡量密度的"曲率"
- 在密度峰值处，$\Delta p < 0$，密度减少
- 在密度谷底处，$\Delta p > 0$，密度增加
- 总效果：密度被"抹平"

**物理类比**：墨水在静水中扩散，从高浓度区域向低浓度区域自发流动，最终趋于均匀。

### 5.4.2 Fokker-Planck方程的推导

从SDE到Fokker-Planck方程的推导基于一个关键思想：**粒子守恒**。

考虑任意测试函数 $\phi(x)$（光滑且紧支撑），其期望值的演化：
$$\frac{d}{dt}\mathbb{E}[\phi(x_t)] = \mathbb{E}\left[\frac{d\phi(x_t)}{dt}\right]$$

使用Itô公式：
$$d\phi(x_t) = \nabla\phi \cdot dx_t + \frac{1}{2}\text{tr}(gg^T \nabla^2\phi) dt$$
$$= \nabla\phi \cdot f dt + \nabla\phi \cdot g dW_t + \frac{1}{2}g^2 \Delta\phi dt$$

取期望（注意 $\mathbb{E}[dW_t] = 0$）：
$$\frac{d}{dt}\mathbb{E}[\phi(x_t)] = \mathbb{E}[f \cdot \nabla\phi + \frac{1}{2}g^2 \Delta\phi]$$

另一方面，用密度表示期望：
$$\frac{d}{dt}\int \phi(x) p_t(x) dx = \int \phi(x) \frac{\partial p_t}{\partial t} dx$$

因此：
$$\int \phi(x) \frac{\partial p_t}{\partial t} dx = \int [f \cdot \nabla\phi + \frac{1}{2}g^2 \Delta\phi] p_t dx$$

使用分部积分（将导数从 $\phi$ 转移到 $p_t$）：
$$\int \phi \frac{\partial p_t}{\partial t} dx = \int \phi [-\nabla \cdot (fp_t) + \frac{1}{2}g^2 \Delta p_t] dx$$

由于 $\phi$ 是任意的，我们得到Fokker-Planck方程！

### 5.4.3 特殊情况与解析解

#### 例1：纯扩散（Ornstein-Uhlenbeck过程）
考虑VP-SDE：$dx = -\frac{1}{2}\beta x dt + \sqrt{\beta} dW$

Fokker-Planck方程变为：
$$\frac{\partial p}{\partial t} = \frac{\beta}{2}\nabla \cdot (xp) + \frac{\beta}{2}\Delta p$$

对于初始条件 $p_0(x) = \delta(x-x_0)$，解为：
$$p_t(x) = \mathcal{N}(x; x_0 e^{-\beta t/2}, \frac{1-e^{-\beta t}}{1}I)$$

这正是DDPM中的前向过程！

#### 例2：稳态分布
当 $\frac{\partial p}{\partial t} = 0$ 时，得到稳态Fokker-Planck方程：
$$\nabla \cdot (fp_{\infty}) = \frac{1}{2}g^2 \Delta p_{\infty}$$

对于VP-SDE，稳态分布是标准正态分布 $p_{\infty}(x) = \mathcal{N}(0, I)$。

### 5.4.4 Fokker-Planck方程的数值方法

虽然我们通常不直接求解Fokker-Planck方程来训练扩散模型，但理解其数值方法有助于深入理解模型行为：

1. **有限差分法**：将空间离散化为网格，用差分近似导数
2. **有限元法**：使用基函数展开密度，转化为ODE系统
3. **粒子方法**：用大量粒子的经验分布近似连续密度
4. **谱方法**：在频域求解，利用快速傅里叶变换

每种方法都有其优缺点，选择取决于问题的维度、边界条件和精度要求。

### 5.4.5 PDE视角的深刻洞察

Fokker-Planck方程揭示了扩散模型的几个深刻性质：

1. **最大熵原理**：扩散过程增加系统的熵，最终达到最大熵分布（高斯分布）

2. **可逆性**：知道分数函数 $\nabla \log p_t$ 后，可以反向求解Fokker-Planck方程，实现时间反演

3. **变分原理**：Fokker-Planck方程可以看作某个自由能泛函的梯度流：
   $$\frac{\partial p}{\partial t} = \nabla \cdot \left(p \nabla \frac{\delta \mathcal{F}[p]}{\delta p}\right)$$
   
   其中 $\mathcal{F}[p]$ 是自由能，包含熵和势能项。

4. **与量子力学的联系**：通过Wick旋转，Fokker-Planck方程与薛定谔方程相关联，开启了量子-经典对应的研究

🔬 **研究线索**：Fokker-Planck方程与最优传输理论中的Wasserstein梯度流有深刻联系。扩散模型可以被看作是在概率分布空间中，沿着某种能量泛函的梯度方向进行演化。探索这种几何观点是当前理论研究的一大热点。

<details>
<summary><strong>练习 5.5：Fokker-Planck方程的理解与应用</strong></summary>

1. **验证解析解**：
   对于Ornstein-Uhlenbeck过程 $dx = -\gamma x dt + \sigma dW$：
   - 写出对应的Fokker-Planck方程
   - 验证 $p_t(x) = \mathcal{N}(x; x_0 e^{-\gamma t}, \frac{\sigma^2}{2\gamma}(1-e^{-2\gamma t}))$ 是其解
   - 讨论 $t \to \infty$ 时的稳态分布

2. **数值求解Fokker-Planck方程**：
   实现一维Fokker-Planck方程的有限差分求解器：
   - 使用中心差分近似空间导数
   - 使用显式或隐式时间步进
   - 与粒子模拟（求解对应SDE）的结果比较
   
   考虑的测试案例：
   - 双井势能：$f(x) = -\nabla U(x)$，其中 $U(x) = (x^2-1)^2$
   - 验证稳态分布 $p_\infty(x) \propto e^{-2U(x)/g^2}$

3. **熵的演化**：
   定义Shannon熵 $H[p] = -\int p \log p dx$。
   - 证明：对于Fokker-Planck方程，$\frac{dH}{dt} \geq 0$（熵增原理）
   - 什么时候等号成立？
   - 计算VP-SDE过程中熵的演化曲线

4. **反向Fokker-Planck方程**：
   推导反向时间SDE对应的Fokker-Planck方程：
   - 从反向SDE出发，应用标准推导
   - 验证它与前向Fokker-Planck方程的关系
   - 解释为什么需要知道分数函数

5. **研究扩展：Wasserstein梯度流**：
   Fokker-Planck方程可以写成Wasserstein梯度流的形式：
   $$\frac{\partial p}{\partial t} = \nabla \cdot \left(p \nabla \frac{\delta \mathcal{F}[p]}{\delta p}\right)$$
   
   探索：
   - 对于不同的自由能 $\mathcal{F}[p]$，得到什么样的演化方程？
   - 扩散模型对应的自由能是什么？
   - 如何设计新的自由能来得到更好的生成模型？

</details>

<details>
<summary><strong>练习 5.6：SDE, ODE, PDE的统一视角</strong></summary>

综合本章所学，分析三种数学框架的联系：

1. **统一框架**：
   给定前向VP-SDE：$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dW$
   - 写出对应的概率流ODE
   - 写出对应的Fokker-Planck方程
   - 验证三者描述同一个概率演化过程

2. **计算复杂度分析**：
   比较三种方法在以下任务上的计算复杂度：
   - 生成单个样本
   - 计算似然 $p(x)$
   - 训练模型参数
   
   考虑维度 $d$、时间步数 $N$、样本数 $M$ 的影响。

3. **选择指南**：
   为以下应用场景选择最合适的框架（SDE/ODE/PDE）：
   - 需要快速生成大量样本
   - 需要精确计算似然进行模型选择
   - 需要理论分析收敛性
   - 需要可解释的生成过程
   - 需要编辑已有样本

4. **创新思考**：
   - 能否设计一个在不同阶段使用不同框架的混合算法？
   - 如何利用PDE的理论洞察改进SDE/ODE的数值算法？
   - 是否存在其他数学框架可以描述扩散过程？

</details>

## 本章小结

本章将我们对扩散模型的理解从离散时间步提升到了连续时间的SDE/PDE框架，这是一个更深刻、更统一的视角。

- **从离散到连续**：我们展示了当时间步数趋于无穷时，离散的DDPM和NCSN过程如何自然地收敛到连续的SDE。
- **反向时间SDE**：我们学习了Anderson定理，它揭示了反向去噪过程的核心是学习分数函数 $\nabla_{x_t} \log p_t(x)$ ，从而将DDPM和分数匹配统一起来。
- **概率流ODE**：我们发现每个SDE都对应一个确定性的ODE，它不仅能实现更快的采样，还能进行精确的似然计算。
- **Fokker-Planck方程**：我们引入了描述概率密度演化的PDE，为宏观理论分析提供了工具。

这个连续时间框架不仅统一了现有的模型，更为未来的创新（如设计新的SDE、开发更快的求解器）提供了无限可能。下一章，我们将探讨另一个优雅的连续时间框架——流匹配（Flow Matching），它从最优传输的视角为生成建模提供了新的思路。
