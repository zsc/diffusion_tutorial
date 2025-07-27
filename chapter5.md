[← 上一章](chapter4.md) | 第5章 / 共14章 | [下一章 →](chapter6.md)

# 第5章：连续时间扩散模型 (PDE/SDE)

到目前为止，我们学习的扩散模型都是在离散时间步上定义的。但如果我们让时间步数趋于无穷，会发生什么？答案是：我们得到了一个更强大、更灵活的数学框架——随机微分方程（Stochastic Differential Equations, SDEs）。Song等人在2021年的工作《Score-Based Generative Modeling through Stochastic Differential Equations》中，将DDPM和NCSN等模型统一在SDE的视角下，开启了连续时间生成建模的新纪元。本章将深入探讨SDE框架，理解其与离散模型的联系，并介绍其对应的反向SDE、概率流ODE和Fokker-Planck方程等核心概念。

## 5.1 从离散到连续：SDE的极限之美

### 5.1.1 离散过程的极限

让我们回顾DDPM的离散前向过程：
$x_k = \sqrt{1-\beta_k} x_{k-1} + \sqrt{\beta_k} z_{k-1}, \quad z_{k-1} \sim \mathcal{N}(0, I)$
其中 $\beta_k$ 是一个小的正数。当 $\beta_k$ 趋近于0时，我们可以用泰勒展开近似 $\sqrt{1 - \beta_k} \approx 1 - \beta_k / 2$。于是，更新步骤可以写成：
$x_k - x_{k-1} \approx -\frac{\beta_k}{2} x_{k-1} + \sqrt{\beta_k} z_{k-1}$
这看起来非常像一个微分方程的离散化形式。现在，我们将时间区间 $[0, T]$ 分成 $N$ 份，令 $\Delta t = T/N$，并设 $\beta_k = b(t_k)\Delta t$，其中 $b(t)$ 是一个连续的函数。那么上式变为：
$\frac{x(t_k) - x(t_{k-1})}{\Delta t} \approx -\frac{b(t_{k-1})}{2} x(t_{k-1}) + \sqrt{b(t_{k-1})} \frac{z_{k-1}}{\sqrt{\Delta t}}$
当 $N \to \infty$（即 $\Delta t \to 0$）时，左边变成了导数 $dx/dt$。而右边的随机项 $z_k / \sqrt{\Delta t}$，根据中心极限定理的推广，其累积效应收敛到一个称为“白噪声”的随机过程，其积分就是布朗运动（Wiener Process）$W_t$。最终，我们得到了一个随机微分方程（SDE）：
$dx_t = -\frac{b(t)}{2} x_t dt + \sqrt{b(t)} dW_t$
这就是DDPM在连续时间下的极限形式，被称为一种方差保持（Variance Preserving, VP）的SDE。

### 5.1.2 SDE的统一框架

SDE为我们提供了一个统一的语言来描述各种扩散模型。一个通用的前向SDE可以写成：
$dx_t = f(x_t, t) dt + g(t) dW_t$
其中 $f(x, t)$ 称为漂移系数（drift coefficient），描述了数据演化的确定性趋势；$g(t)$ 称为扩散系数（diffusion coefficient），控制着随机噪声的强度。

> **定义：SDE家族**
> - **VP-SDE (Variance Preserving)**：对应DDPM。$f(x,t) = -1/2 \beta(t) x$， $g(t) = \sqrt{\beta(t)}$。其特点是在演化过程中，如果初始数据方差为1，那么在任何时刻 $t$，数据方差都近似为1。
> - **VE-SDE (Variance Exploding)**：对应NCSN。$f(x,t) = 0$， $g(t) = \sqrt{d[\sigma^2(t)]/dt}$。其特点是数据均值不变，但方差随时间爆炸式增长。
> - **sub-VP-SDE**：是VP-SDE的一个变体，具有更好的数值稳定性和理论性质。

💡 **开放问题**：VP和VE只是两种最常见的选择。是否存在其他形式的SDE，能在特定类型的数据（如具有周期性的时间序列、位于特殊流形上的数据）上实现更高效的生成？设计数据依赖的SDE是一个前沿研究方向。

## 5.2 反向时间SDE：学习去噪

如果前向SDE描述了数据如何被噪声破坏，那么我们如何构建一个反向的过程来从噪声中恢复数据呢？物理学家和数学家早在几十年前就解决了这个问题。

> **定理：Anderson反向时间SDE**
> 对于一个前向SDE $dx = f(x, t)dt + g(t)dW_t$，其对应的反向时间SDE（从时间 $T$ 到 $0$）存在且唯一，形式如下：
> $dx_t = [f(x_t, t) - g(t)^2 \nabla_{x_t} \log p_t(x_t)] dt + g(t) d\bar{W}_t$
> 其中 $d\bar{W}_t$ 是反向时间的布朗运动，而 $\nabla_{x_t} \log p_t(x_t)$ 就是我们在第四章学习过的**分数函数**！

这个定理是整个生成模型理论的基石。它告诉我们：
1.  **反向过程是存在的**：从噪声恢复数据的过程在数学上是良定义的。
2.  **反向过程的核心是分数**：反向的漂移项 = 前向漂移项 - 噪声强度 * 分数。为了反转过程，我们必须在每个时刻 $t$ 沿着能使概率密度 $p_t(x_t)$ 增加最快的方向（即分数函数的方向）进行修正。
3.  **生成建模的核心任务**：因此，训练一个扩散模型的核心任务，就是学习所有时刻 $t$ 的分数函数 $\nabla_{x_t} \log p_t(x_t)$。这完美地将DDPM的去噪目标和NCSN的分数匹配目标统一了起来。

⚡ **实现挑战**：分数函数 $\nabla_{x_t} \log p_t(x_t)$ 是未知的，且依赖于整个数据集在时刻 $t$ 的分布。我们需要用一个神经网络 $s_\theta(x_t, t)$ 来近似它。这个网络被称为分数模型（score model）。

<details>
<summary><strong>练习 5.1：推导反向SDE</strong></summary>

1.  **VP-SDE的反向过程**：给定前向VP-SDE $dx = -1/2 \beta(t) x dt + \sqrt{\beta(t)} dW_t$，写出其对应的反向SDE表达式。
2.  **与DDPM的联系**：在DDPM中，我们训练模型 $\epsilon_\theta(x_t, t)$ 来预测噪声。而在SDE框架下，我们训练模型 $s_\theta(x_t, t)$ 来预测分数。证明这两个模型之间存在一个简单的线性关系：$s_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$。
3.  **研究思路**：
    *   Anderson定理的原始证明涉及随机微积分和测度论。可以查阅相关文献，理解其对概率密度函数 $p_t$ 的光滑性要求。
    *   探索当 $g(t)$ 是矩阵（各向异性噪声）时，反向SDE的形式如何变化。

</details>

## 5.3 概率流ODE：确定性的生成路径

SDE的采样过程是随机的，意味着从同一个噪声 $x_T$ 出发，每次得到的 $x_0$ 都会略有不同。是否存在一种确定性的路径，也能将噪声映射到数据呢？答案是肯定的，这就是概率流（Probability Flow）ODE。

> **定义：概率流ODE**
> 每一个SDE都存在一个与之对应的常微分方程（Ordinary Differential Equation, ODE），它具有与SDE完全相同的边缘概率密度 $p_t(x)$。这个ODE被称为概率流ODE：
> $dx_t = [f(x_t, t) - \frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)] dt$
> 注意，这个ODE中没有了随机的 $dW_t$ 项！

概率流ODE的直观理解是，它描述了概率密度 $p_t(x)$ 中“概率质量”的平均速度场。沿着这个速度场流动，样本的分布演化将和随机的SDE完全一样。

**概率流ODE的优势**：
1.  **确定性采样**：从一个 $x_T$ 出发，总能得到完全相同的 $x_0$。这对于需要可复现生成的任务非常有用。
2.  **更快的采样**：作为ODE，我们可以使用各种现成的高阶数值求解器（如Runge-Kutta法），可以用比SDE求解器少得多的步数（例如50-100步 vs 1000步）得到高质量的样本。这是DDIM等快速采样算法的理论基础。
3.  **精确的似然计算**：通过ODE的瞬时变量变换公式，可以精确计算出数据点 $x_0$ 的对数似然，而SDE只能计算一个下界。

🌟 **理论空白**：SDE和ODE提供了两种不同的采样路径。SDE路径是随机的、高维的，而ODE路径是确定性的、低维的。这两种路径的几何性质有何不同？它们在数据流形上是如何移动的？理解这一点可能有助于设计出更优的采样算法。

## 5.4 Fokker-Planck方程：从粒子到密度的演化

SDE和ODE描述了单个数据点（粒子）的轨迹。如果我们想从宏观上描述整个概率密度 $p_t(x)$ 的演化，就需要偏微分方程（Partial Differential Equation, PDE）的语言，这就是Fokker-Planck方程。

> **定义：Fokker-Planck方程**
> 对于一个SDE $dx = f(x,t)dt + g(t)dW_t$，其概率密度 $p_t(x)$ 的演化遵循Fokker-Planck方程：
> $\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (f(x,t)p_t(x)) + \frac{1}{2} g(t)^2 \Delta p_t(x)$
> 其中 $\nabla \cdot$ 是散度算子，$\Delta$ 是拉普拉斯算子。

这个方程的直观含义是：概率密度的变化率 = 由漂移项引起的输运（advection）+ 由扩散项引起的平滑（diffusion）。它将微观的粒子随机运动与宏观的概率密度演化联系在了一起。虽然在实践中我们通常不直接求解这个PDE，但它为理解扩散模型提供了强大的理论工具，并与物理学中的热力学、流体力学等领域建立了深刻的联系。

🔬 **研究线索**：Fokker-Planck方程与最优传输理论中的Wasserstein梯度流有深刻联系。扩散模型可以被看作是在概率分布空间中，沿着某种能量泛函的梯度方向进行演化。探索这种几何观点是当前理论研究的一大热点。

<details>
<summary><strong>练习 5.2：SDE, ODE, PDE的联系与区别</strong></summary>

1.  **写出ODE**：给定前向VP-SDE，写出其对应的概率流ODE的表达式。
2.  **写出PDE**：给定前向VP-SDE，写出其对应的Fokker-Planck方程。
3.  **比较与分析**：总结SDE、ODE、PDE这三种数学工具在描述扩散模型时的角色和优缺点。
    *   SDE的优点是什么？（提示：与分数匹配的联系）
    *   ODE的优点是什么？（提示：采样速度和似然计算）
    *   PDE的优点是什么？（提示：宏观理论分析）
4.  **开放探索**：概率流ODE提供了一条确定性的从噪声到数据的路径。这本质上定义了一个可逆的函数 $g: x_T \to x_0$。这与连续归一化流（Continuous Normalizing Flows, CNF）有何联系？分析两种模型的异同。

</details>

## 本章小结

本章将我们对扩散模型的理解从离散时间步提升到了连续时间的SDE/PDE框架，这是一个更深刻、更统一的视角。

- **从离散到连续**：我们展示了当时间步数趋于无穷时，离散的DDPM和NCSN过程如何自然地收敛到连续的SDE。
- **反向时间SDE**：我们学习了Anderson定理，它揭示了反向去噪过程的核心是学习分数函数 $\nabla_{x_t} \log p_t(x)$，从而将DDPM和分数匹配统一起来。
- **概率流ODE**：我们发现每个SDE都对应一个确定性的ODE，它不仅能实现更快的采样，还能进行精确的似然计算。
- **Fokker-Planck方程**：我们引入了描述概率密度演化的PDE，为宏观理论分析提供了工具。

这个连续时间框架不仅统一了现有的模型，更为未来的创新（如设计新的SDE、开发更快的求解器）提供了无限可能。下一章，我们将探讨另一个优雅的连续时间框架——流匹配（Flow Matching），它从最优传输的视角为生成建模提供了新的思路。
