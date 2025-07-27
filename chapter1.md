[← 返回目录](index.md)
 第1章 / 共14章
 [下一章 →](chapter2.md)



# 第1章：扩散模型导论



## 1.1 什么是扩散模型？



扩散模型（Diffusion Models）是一类强大的生成模型，它通过学习数据的逐步去噪过程来生成高质量的样本。这个过程可以类比为物理学中的扩散现象：就像墨水在水中逐渐扩散直至均匀分布，扩散模型将数据逐步添加噪声直至变成纯噪声，然后学习如何反转这个过程。



> **定义**
> 定义 1.1（扩散模型）

 扩散模型是一类概率生成模型，它定义了两个过程：


 - **前向过程（Forward Process）**：将数据逐步添加噪声，最终变成纯高斯噪声
 - **反向过程（Reverse Process）**：从纯噪声开始，逐步去噪恢复出数据





## 1.2 扩散模型的数学基础



### 1.2.1 前向扩散过程



给定数据点 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$，前向过程通过 $T$ 步逐渐添加高斯噪声：




 $$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$




其中 $\beta_t$ 是第 $t$ 步的噪声调度（noise schedule），控制每一步添加噪声的量。通过重参数化技巧，我们可以直接从 $\mathbf{x}_0$ 采样任意时刻 $t$ 的 $\mathbf{x}_t$：




 $$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$




其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$。




练习 1.1：理解噪声调度

假设我们有一个线性噪声调度：$\beta_t = \frac{t}{T} \cdot 0.02$，其中 $T=1000$。



 - 计算 $t=100$ 时的 $\bar{\alpha}_{100}$
 - 当 $t \to T$ 时，$\mathbf{x}_t$ 的分布趋向于什么？

**解答：**



 - 首先计算 $\beta_{100} = \frac{100}{1000} \times 0.02 = 0.002$
 - 因此 $\alpha_{100} = 1 - 0.002 = 0.998$
 - $\bar{\alpha}_{100} = \prod_{s=1}^{100}\alpha_s \approx 0.998^{100} \approx 0.819$
 - 当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，所以 $q(\mathbf{x}_t | \mathbf{x}_0) \to \mathcal{N}(0, \mathbf{I})$





### 1.2.2 反向去噪过程



反向过程的目标是学习条件分布 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$，使其能够逐步去除噪声。我们通常将其参数化为：




 $$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$$




## 1.3 扩散模型的优势




 - **生成质量高**：能够生成极其逼真的图像，在许多指标上超越GAN
 - **训练稳定**：不存在GAN的模式崩塌问题，训练过程稳定可靠
 - **理论基础扎实**：有严格的概率论基础，目标函数有明确的意义
 - **灵活性强**：易于扩展到条件生成、图像编辑等任务




## 1.4 实践：可视化扩散过程



让我们通过一个简单的例子来直观理解扩散过程。下面的代码展示了如何对一个2D高斯分布进行前向扩散：




import numpy as np
import matplotlib.pyplot as plt

# 生成初始数据：2D高斯分布
np.random.seed(42)
x0 = np.random.randn(1000, 2) * 0.5 + np.array([2, 2])

# 定义噪声调度
T = 100
betas = np.linspace(0.001, 0.02, T)
alphas = 1 - betas
alphas_bar = np.cumprod(alphas)

# 可视化不同时间步的数据分布
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
time_steps = [0, 25, 50, 75, 99]

for i, t in enumerate(time_steps):
 if t == 0:
 xt = x0
 else:
 # 前向扩散：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
 epsilon = np.random.randn(*x0.shape)
 xt = np.sqrt(alphas_bar[t-1]) * x0 + np.sqrt(1 - alphas_bar[t-1]) * epsilon

 axes[i].scatter(xt[:, 0], xt[:, 1], alpha=0.5, s=10)
 axes[i].set_xlim(-4, 4)
 axes[i].set_ylim(-4, 4)
 axes[i].set_title(f't = {t}')
 axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




练习 1.2：实现简单的1D扩散
 实现一个1D扩散过程的完整前向和反向过程。给定初始数据为单点 $x_0 = 5$：



 - 实现前向扩散过程，记录每一步的值
 - 假设你知道真实的反向过程，实现去噪过程
 - 绘制前向和反向过程的轨迹

**参考代码：**



import numpy as np
import matplotlib.pyplot as plt

# 初始化
x0 = 5.0
T = 50
betas = np.linspace(0.01, 0.2, T)

# 前向过程
x_forward = [x0]
for t in range(T):
 noise = np.random.randn()
 x_next = np.sqrt(1 - betas[t]) * x_forward[-1] + np.sqrt(betas[t]) * noise
 x_forward.append(x_next)

# 反向过程（假设知道真实的去噪方向）
x_reverse = [x_forward[-1]]
for t in range(T-1, -1, -1):
 # 这里简化处理，实际需要学习
 predicted_x0 = x0 # 假设知道目标
 direction = (predicted_x0 - x_reverse[-1]) / (t + 1)
 x_reverse.append(x_reverse[-1] + direction * 0.1 + np.random.randn() * 0.01)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(x_forward, 'b-', label='Forward Process', alpha=0.7)
plt.plot(x_reverse, 'r-', label='Reverse Process', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()





## 1.5 历史发展与里程碑


 扩散模型的发展经历了几个重要阶段：




 - **2015年**：Sohl-Dickstein等人提出了基于非平衡热力学的深度无监督学习方法
 - **2020年**：Ho等人提出DDPM（Denoising Diffusion Probabilistic Models），简化了训练过程
 - **2021年**：Song等人提出DDIM，大幅加速了采样过程
 - **2022年**：Stable Diffusion的发布，使文本到图像生成达到了新高度




## 1.6 本章小结



在本章中，我们学习了：



 - 扩散模型的基本概念和直观理解
 - 前向扩散过程的数学表述
 - 反向去噪过程的基本思想
 - 扩散模型相比其他生成模型的优势
 - 通过代码实现加深对扩散过程的理解




下一章我们将深入学习DDPM的具体实现细节，包括训练算法、损失函数推导和实际代码实现。




综合练习：比较不同噪声调度

实现并比较三种不同的噪声调度策略：



 - 线性调度：$\beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}})$
 - 余弦调度：基于余弦函数的平滑调度
 - 二次调度：$\beta_t = \beta_{\text{start}} + \left(\frac{t}{T}\right)^2(\beta_{\text{end}} - \beta_{\text{start}})$



绘制 $\bar{\alpha}_t$ 随时间的变化曲线，并讨论它们的优缺点。

**提示：**



 - 线性调度简单直观，但可能在早期步骤添加噪声过快
 - 余弦调度在中间阶段更平滑，有助于保留更多信息
 - 二次调度在早期添加噪声较慢，后期加速
 - 选择哪种调度取决于具体任务和数据特性