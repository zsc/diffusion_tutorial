[← 上一章](chapter2.md)
 第3章 / 共14章
 [下一章 →](chapter4.md)



# 第3章：去噪扩散概率模型 (DDPM)



 2020年，Ho等人的论文"Denoising Diffusion Probabilistic Models"让扩散模型真正进入了实用阶段。DDPM不仅简化了训练过程，还达到了与GAN相媲美的生成质量。本章将深入剖析DDPM的数学原理、训练算法和实现细节。通过本章学习，你将掌握如何从零实现一个完整的DDPM，并理解其背后的概率论基础。



## 3.1 DDPM的核心思想：简化与统一



在DDPM之前，扩散模型虽然理论优雅，但实践困难。2015年Sohl-Dickstein等人的开创性工作需要估计整个反向过程的熵，训练极其复杂。DDPM的革命性贡献在于：**将复杂的变分推断简化为简单的去噪任务**。



> **定义**
> DDPM的三个关键简化



 - **固定方差调度**：前向过程使用预定义的 $\beta_t$ 序列，无需学习
 - **简化反向过程**：假设反向过程也是高斯分布，只需学习均值（实际上是学习噪声）
 - **重参数化目标**：将预测均值转换为预测噪声，大幅提升训练稳定性





### 3.1.1 从复杂到简单：DDPM的洞察



让我们通过一个类比来理解DDPM的核心思想：




#### 墨水扩散的类比


想象一滴墨水在水中扩散：



 - **前向过程**：墨水逐渐扩散，最终均匀分布（物理过程，确定的）
 - **反向过程**：如何让扩散的墨水重新聚集？（需要学习的）



DDPM的关键洞察：**在每个时间步，我们只需要知道"墨水应该向哪个方向聚集"**，而这个方向恰好与添加的噪声方向相反！




### 3.1.2 数学框架概览



DDPM定义了两个过程：




 **前向过程（固定）**：

 $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$


 **反向过程（学习）**：

 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$




关键创新在于如何参数化 $\boldsymbol{\mu}_\theta$：




# 早期方法：直接预测均值（不稳定）
mean = model(x_t, t)

# DDPM创新：预测噪声（稳定且有效）
noise_pred = model(x_t, t)
mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_t)



### 3.1.3 为什么预测噪声更好？


 这个看似简单的改变带来了巨大的好处：



> **定义**
> 预测噪声的优势

 
 
 方面
 预测均值
 预测噪声
 
 
 输出范围
 需要匹配数据分布
 标准高斯（已归一化）
 
 
 训练信号
 随t变化剧烈
 各时间步相对一致
 
 
 梯度流
 可能梯度消失
 梯度传播良好
 
 
 物理意义
 预测去噪后的图像
 预测添加的噪声
 
 



### 3.1.4 DDPM vs 早期扩散模型



让我们对比DDPM与2015年的原始扩散模型：




# 2015年的扩散模型（复杂）
# 需要估计：
# 1. 前向过程的熵
# 2. 反向过程的完整分布
# 3. 变分参数的优化
# 训练极其不稳定，生成质量差

# DDPM（2020年）的训练（极简）
for x_0, _ in dataloader:
 t = torch.randint(0, num_timesteps, (batch_size,))
 noise = torch.randn_like(x_0)
 x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * noise

 noise_pred = model(x_t, t)
 loss = F.mse_loss(noise_pred, noise)
 loss.backward()


 这种简化不是以牺牲性能为代价的——相反，DDPM首次让扩散模型在生成质量上与GAN竞争，同时保持了训练的稳定性。




思考题 3.1：直觉理解

为什么在高噪声情况下（大的t），预测噪声比预测原始图像更容易？提示：考虑信噪比。

**答案：**


当t很大时，$\mathbf{x}_t \approx \mathcal{N}(0, \mathbf{I})$，几乎是纯噪声。此时：



 - 原始图像 $\mathbf{x}_0$ 的信息几乎完全丢失，预测它需要"凭空想象"
 - 但添加的噪声 $\boldsymbol{\epsilon}$ 是已知的，且占主导地位
 - 网络只需要识别噪声模式，而不是重建复杂的图像结构



类比：在雪花噪声的电视屏幕上，识别噪声模式比重建原始节目容易得多。





## 3.2 前向过程：数学推导与性质



前向过程是扩散模型的基础，它定义了如何将数据逐步转换为噪声。虽然这个过程在训练和推理时都不需要实际执行完整的马尔可夫链，但理解其数学性质对掌握DDPM至关重要。



### 3.2.1 马尔可夫链的构建



前向过程定义为一个马尔可夫链：




 $$\mathbf{x}_0 \to \mathbf{x}_1 \to \mathbf{x}_2 \to \cdots \to \mathbf{x}_T$$




其中每一步的转移概率为：




 $$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$




> **定义**
> 关键性质1：方差调度的约束


为什么是 $\sqrt{1-\beta_t}$ 而不是其他系数？这是为了保持信号的期望能量：



 $$\mathbb{E}[\|\mathbf{x}_t\|^2 | \mathbf{x}_{t-1}] = (1-\beta_t)\|\mathbf{x}_{t-1}\|^2 + \beta_t \cdot d$$



其中 $d$ 是数据维度。当 $\beta_t$ 很小时，信号能量近似保持不变。




让我们验证这个性质：




import torch
import matplotlib.pyplot as plt

# 验证能量保持性质
x_0 = torch.randn(1000, 3, 32, 32) # 1000个32x32的RGB图像
beta = 0.02 # 典型的beta值

# 一步前向过程
noise = torch.randn_like(x_0)
x_1 = torch.sqrt(1 - beta) * x_0 + torch.sqrt(beta) * noise

print(f"原始信号能量: {x_0.pow(2).mean():.4f}")
print(f"扩散后信号能量: {x_1.pow(2).mean():.4f}")
print(f"理论预期: {(1-beta)*x_0.pow(2).mean() + beta*3*32*32:.4f}")



### 3.2.2 重参数化技巧


 DDPM的一个关键技巧是：我们可以直接从 $\mathbf{x}_0$ 采样任意时刻的 $\mathbf{x}_t$，而不需要逐步模拟整个马尔可夫链。



> **定义**
> 定理：闭式采样公式


定义 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$，则：



 $$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$





**证明**（这个证明很重要，值得仔细理解）：





我们用归纳法证明。


**基础情况**（$t=1$）：显然成立，因为 $\bar{\alpha}_1 = \alpha_1 = 1 - \beta_1$。



**归纳步骤**：假设对 $t-1$ 成立，即：

 $$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}$$


其中 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$。根据前向过程定义：

 $$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t$$


代入 $\mathbf{x}_{t-1}$ 的表达式：

 $$\mathbf{x}_t = \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t$$

 $$= \sqrt{\alpha_t\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\boldsymbol{\epsilon}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t$$


注意到 $\alpha_t\bar{\alpha}_{t-1} = \bar{\alpha}_t$，且两个独立高斯噪声的线性组合仍是高斯噪声：

 $$\text{Var}[\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\boldsymbol{\epsilon}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t] = \alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = 1-\bar{\alpha}_t$$


因此：

 $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$


其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。证毕。





### 3.2.3 噪声调度的设计



噪声调度 $\{\beta_t\}_{t=1}^T$ 的选择对模型性能有重要影响。DDPM原文使用线性调度，但后续研究发现其他调度可能更优。




import numpy as np
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
 """DDPM原始的线性调度"""
 return np.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
 """Improved DDPM的余弦调度"""
 steps = timesteps + 1
 t = np.linspace(0, timesteps, steps)
 alphas_cumprod = np.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
 alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
 betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
 return np.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
 """二次调度（较少使用）"""
 t = np.linspace(0, 1, timesteps)
 return beta_start + (beta_end - beta_start) * t ** 2

# 可视化不同调度
timesteps = 1000
linear_betas = linear_beta_schedule(timesteps)
cosine_betas = cosine_beta_schedule(timesteps)
quadratic_betas = quadratic_beta_schedule(timesteps)

# 计算信噪比（更直观的指标）
# 比较不同噪声调度
linear_betas = linear_beta_schedule(1000)
cosine_betas = cosine_beta_schedule(1000)
quadratic_betas = quadratic_beta_schedule(1000)

# 计算信噪比
def compute_snr(betas):
 alphas = 1 - betas
 alphas_cumprod = np.cumprod(alphas)
 return alphas_cumprod / (1 - alphas_cumprod)

# 展示不同调度下的关键统计数据
alphas_cumprod_linear = np.cumprod(1 - linear_betas)
alphas_cumprod_cosine = np.cumprod(1 - cosine_betas)

t_vis = [0, 250, 500, 750, 999]
print("Signal preservation (√ᾱ_t) at key timesteps:")
print("Timestep | Linear | Cosine")
for t in t_vis:
 print(f"{t:8d} | {np.sqrt(alphas_cumprod_linear[t]):.4f} | {np.sqrt(alphas_cumprod_cosine[t]):.4f}")

print("\nSNR at key timesteps:")
snr_linear = compute_snr(linear_betas)
snr_cosine = compute_snr(cosine_betas)
for t in t_vis:
 print(f"{t:8d} | {snr_linear[t]:.4f} | {snr_cosine[t]:.4f}")



> **定义**
> 调度策略对比

 
 
 调度类型
 特点
 优势
 劣势
 
 
 线性 (Linear)
 β线性增长
 简单直观
 前期破坏过快
 
 
 余弦 (Cosine)
 基于SNR设计
 更好的感知质量
 末期可能过慢
 
 
 二次 (Quadratic)
 β二次增长
 前期保留更多信息
 后期可能太激进
 
 




练习 3.2：实现自定义噪声调度
 设计一个"S形"噪声调度，使得：



 - 前期（t 800）：再次放缓，确保收敛到纯噪声



实现这个调度并与标准调度对比SNR曲线。

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
 """S形噪声调度"""
 t = np.linspace(-6, 6, timesteps)
 sigmoid = 1 / (1 + np.exp(-t))
 betas = beta_start + (beta_end - beta_start) * sigmoid
 return betas

# 也可以分段设计
def piecewise_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
 """分段噪声调度"""
 betas = np.zeros(timesteps)

 # 前期：缓慢增长
 t1 = int(0.2 * timesteps)
 betas[:t1] = np.linspace(beta_start, beta_start * 5, t1)

 # 中期：快速增长
 t2 = int(0.8 * timesteps)
 betas[t1:t2] = np.linspace(beta_start * 5, beta_end * 0.8, t2 - t1)

 # 后期：缓慢增长到beta_end
 betas[t2:] = np.linspace(beta_end * 0.8, beta_end, timesteps - t2)

 return betas
 **关键洞察**：好的噪声调度应该在保留足够信息和充分探索噪声空间之间取得平衡。余弦调度之所以优于线性调度，正是因为它更好地平衡了这两个需求。





## 3.3 反向过程：从噪声到图像



反向过程是扩散模型的核心——如何从纯噪声逐步恢复出清晰的数据。DDPM的关键贡献之一是推导出了在已知 $\mathbf{x}_0$ 时的反向条件分布的闭式解。



### 3.3.1 反向条件概率的推导



这是DDPM中最重要的数学推导之一。我们想要计算 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$。



> **定义**
> 定理：反向过程的后验分布


给定 $\mathbf{x}_t$ 和 $\mathbf{x}_0$，反向过程的后验分布为：



 $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I})$$



其中：



 $$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t$$

 $$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$





**证明**：使用贝叶斯定理：




 $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}$$




由于前向过程的马尔可夫性质，$q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t|\mathbf{x}_{t-1})$。现在我们知道：




 - $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$
 - $q(\mathbf{x}_{t-1}|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$
 - $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$




将三个高斯分布代入贝叶斯公式，经过繁琐但直接的代数运算（主要是配方），可以得到上述结果。




#### 💡 关键洞察


注意 $\tilde{\boldsymbol{\mu}}_t$ 是 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的**线性组合**！这意味着：



 - 如果我们知道 $\mathbf{x}_0$，反向过程就是确定的（除了小的高斯噪声）
 - 实践中我们不知道 $\mathbf{x}_0$，所以需要神经网络来预测它
 - 这解释了为什么扩散模型本质上是在学习"去噪"





### 3.3.2 参数化选择：预测噪声 vs 预测均值



既然 $\tilde{\boldsymbol{\mu}}_t$ 依赖于未知的 $\mathbf{x}_0$，我们需要用神经网络来近似它。DDPM提供了几种参数化方式：




# 方式1：直接预测均值（最直接但不稳定）
mu_theta = model(x_t, t)

# 方式2：预测x_0（需要clip到合理范围）
x_0_pred = model(x_t, t)
mu_theta = (sqrt_alpha_bar_prev * beta_t * x_0_pred +
 sqrt_alpha_t * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)

# 方式3：预测噪声（DDPM的选择，最稳定）
epsilon_pred = model(x_t, t)
x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
mu_theta = (sqrt_alpha_bar_prev * beta_t * x_0_pred +
 sqrt_alpha_t * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)


 为什么预测噪声更好？让我们通过重参数化来理解：





由于 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$，我们可以表示：

 $$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$


代入 $\tilde{\boldsymbol{\mu}}_t$ 的表达式，经过化简可得：

 $$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$$




这个表达式揭示了一个优雅的事实：**反向过程的均值只需要知道添加的噪声 $\boldsymbol{\epsilon}$！**



> **定义**
> 三种参数化的对比

 
 
 参数化
 优点
 缺点
 使用场景
 
 
 预测 $\boldsymbol{\mu}_\theta$
 直接，无需转换
 不同t的输出尺度差异大
 几乎不用
 
 
 预测 $\mathbf{x}_0$
 语义清晰
 高噪声时预测困难
 某些条件生成任务
 
 
 预测 $\boldsymbol{\epsilon}$
 输出标准化，训练稳定
 间接，需要转换
 标准选择
 
 



### 3.3.3 方差的处理：固定 vs 可学习



DDPM的另一个简化是使用固定的方差 $\tilde{\beta}_t$。但这是最优的吗？




# DDPM：固定方差（两种选择）
# 选择1：使用后验方差
variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

# 选择2：使用β_t（DDPM论文的选择）
variance = beta_t

# 改进的DDPM：学习方差
# 网络同时预测噪声和方差
epsilon_pred, v_pred = model(x_t, t).chunk(2, dim=1)

# 参数化方差（在对数空间插值）
min_log = torch.log(beta_t)
max_log = torch.log((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
log_variance = v_pred * max_log + (1 - v_pred) * min_log
variance = torch.exp(log_variance)




#### ⚠️ 实践经验

 尽管学习方差理论上更优（可以获得更好的似然），但在实践中：



 - 固定方差的DDPM已经能生成高质量图像
 - 学习方差增加了训练的复杂度
 - 对于大多数应用，固定方差是足够的
 - 如果追求最优似然（如压缩任务），才考虑学习方差






练习 3.3：验证不同参数化的等价性

实现三种参数化方式，验证它们在数学上是等价的：



 - 给定相同的 $\mathbf{x}_t$、$\mathbf{x}_0$ 和 $t$
 - 计算真实的噪声 $\boldsymbol{\epsilon}$
 - 用三种方式计算 $\tilde{\boldsymbol{\mu}}_t$
 - 验证结果相同（在数值精度内）

import torch

# 设置
batch_size = 4
channels = 3
size = 32
t = 500
T = 1000

# 初始化
x_0 = torch.randn(batch_size, channels, size, size)
epsilon = torch.randn_like(x_0)

# 计算alpha相关值
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alpha_t = alphas[t]
alpha_bar_t = alphas_bar[t]
alpha_bar_prev = alphas_bar[t-1]
beta_t = betas[t]

# 前向过程
x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

# 方式1：直接计算真实的后验均值
mu_true = (torch.sqrt(alpha_bar_prev) * beta_t * x_0 +
 torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)

# 方式2：通过预测x_0
x_0_pred = x_0 # 假设完美预测
mu_x0 = (torch.sqrt(alpha_bar_prev) * beta_t * x_0_pred +
 torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)

# 方式3：通过预测噪声
epsilon_pred = epsilon # 假设完美预测
x_0_from_eps = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
mu_eps = (torch.sqrt(alpha_bar_prev) * beta_t * x_0_from_eps +
 torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)

# 或者直接用简化公式
mu_eps_direct = (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_t)

# 验证
print(f"方式1和方式2的差异: {(mu_true - mu_x0).abs().max():.6f}")
print(f"方式1和方式3的差异: {(mu_true - mu_eps).abs().max():.6f}")
print(f"方式1和方式3(直接)的差异: {(mu_true - mu_eps_direct).abs().max():.6f}")

# 输出应该都接近0（在浮点精度范围内）
 **关键洞察**：三种参数化在数学上等价，但训练动态不同。预测噪声之所以更稳定，是因为噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 始终是标准化的，而 $\mathbf{x}_0$ 的分布可能很复杂。





## 3.4 训练目标：变分下界的简化



DDPM的另一个重要贡献是将复杂的变分下界（ELBO）简化为一个简单的去噪目标。这一节我们将详细推导这个过程。



### 3.4.1 完整的变分下界



我们的目标是最大化数据的对数似然 $\log p_\theta(\mathbf{x}_0)$。由于直接计算困难，我们优化其变分下界：




 $$\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = -L_{\text{VLB}}$$




其中 $L_{\text{VLB}}$ 是变分下界损失。经过展开（使用马尔可夫性质），可以得到：




 $$L_{\text{VLB}} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0$$




其中各项定义为：



> **定义**
> 变分下界的三个组成部分



 $$L_T = D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))$$
 $$L_{t-1} = \mathbb{E}_q\left[D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]$$
 $$L_0 = \mathbb{E}_q\left[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]$$




 - $L_T$：先验匹配项，通常很小可以忽略（因为 $q(\mathbf{x}_T|\mathbf{x}_0) \approx \mathcal{N}(0, \mathbf{I})$）
 - $L_{t-1}$：去噪匹配项，这是主要的优化目标
 - $L_0$：重建项，决定最终输出质量





关键在于如何处理 $L_{t-1}$ 项。由于我们知道 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的闭式解（见3.3.1节），且假设 $p_\theta$ 也是高斯分布，KL散度可以简化为：




 $$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2\right] + C$$




其中 $C$ 是与 $\theta$ 无关的常数。



### 3.4.2 简化的去噪目标



DDPM的关键洞察是：通过选择噪声预测参数化，可以将上述目标进一步简化。回忆3.3.2节的结果：




 $$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$$




如果我们参数化 $\boldsymbol{\mu}_\theta$ 为：




 $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$




那么 $L_{t-1}$ 可以简化为：




 $$L_{t-1} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$




其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。




#### 🎯 DDPM的简化训练目标


Ho等人发现，忽略权重系数并对所有时间步求和，得到的简化目标效果更好：



 $$L_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$



这就是著名的"简单损失"——只需要预测噪声！




### 3.4.3 损失函数的加权策略



虽然简单损失效果很好，但不同时间步的重要性确实不同。后续研究提出了各种加权策略：




import torch
import matplotlib.pyplot as plt

# 不同的损失加权策略
def get_loss_weight(t, strategy='simple', snr_gamma=5.0):
 """
 计算时间步t的损失权重

 策略:
 - simple: 所有时间步权重相同（DDPM原始）
 - snr: 基于信噪比的加权
 - truncated_snr: 截断的SNR加权（防止极端值）
 - importance: 基于重要性采样
 """
 if strategy == 'simple':
 return 1.0

 elif strategy == 'snr':
 # 权重与信噪比成反比
 snr = alpha_bar[t] / (1 - alpha_bar[t])
 return 1.0 / (1.0 + snr)

 elif strategy == 'truncated_snr':
 # Min-SNR-γ 加权（Hang et al., 2023）
 snr = alpha_bar[t] / (1 - alpha_bar[t])
 return torch.minimum(snr, torch.tensor(snr_gamma)) / snr

 elif strategy == 'importance':
 # 基于L_t系数的重要性加权
 return beta[t]**2 / (2 * sigma[t]**2 * alpha[t] * (1 - alpha_bar[t]))

# 分析不同加权策略
T = 1000
t = torch.arange(T)
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sigma = beta # DDPM的选择

# 计算不同策略的权重
strategies = ['simple', 'snr', 'truncated_snr', 'importance']
weight_stats = {}

for strategy in strategies:
 weights = torch.tensor([get_loss_weight(i, strategy) for i in range(T)])
 weight_stats[strategy] = {
 'min': weights.min().item(),
 'max': weights.max().item(),
 'mean': weights.mean().item(),
 'std': weights.std().item()
 }

# 打印权重统计
print("Loss weight statistics for different strategies:")
for strategy, stats in weight_stats.items():
 print(f"\n{strategy}:")
 print(f" Min: {stats['min']:.6f}")
 print(f" Max: {stats['max']:.6f}")
 print(f" Mean: {stats['mean']:.6f}")
 print(f" Std: {stats['std']:.6f}")



> **定义**
> 加权策略对比

 
 
 策略
 动机
 效果
 计算开销
 
 
 简单 (Simple)
 简化训练
 基准，效果已经不错
 最低
 
 
 SNR加权
 平衡不同噪声水平
 改善高噪声区域
 低
 
 
 Min-SNR-γ
 避免极端权重
 目前最优
 低
 
 
 重要性采样
 理论最优
 实践中不稳定
 中等
 
 



### 3.4.4 训练算法总结


 综合以上推导，DDPM的训练算法极其简洁：




def train_ddpm(model, dataloader, num_epochs, T=1000):
 """DDPM训练循环"""
 optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

 # 预计算噪声调度相关值
 betas = linear_beta_schedule(T)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)
 sqrt_alphas_bar = torch.sqrt(alphas_bar)
 sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

 for epoch in range(num_epochs):
 for batch_idx, (x_0, _) in enumerate(dataloader):
 batch_size = x_0.shape[0]

 # 随机采样时间步
 t = torch.randint(0, T, (batch_size,), device=x_0.device)

 # 采样噪声
 epsilon = torch.randn_like(x_0)

 # 前向扩散：计算x_t
 x_t = (sqrt_alphas_bar[t, None, None, None] * x_0 +
 sqrt_one_minus_alphas_bar[t, None, None, None] * epsilon)

 # 预测噪声
 epsilon_pred = model(x_t, t)

 # 计算损失
 loss = F.mse_loss(epsilon_pred, epsilon)

 # 反向传播
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 if batch_idx % 100 == 0:
 print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')




练习 3.4：实现加权损失
 修改上述训练代码，实现Min-SNR-γ加权策略：



 - 计算每个时间步的SNR
 - 应用Min-SNR-γ加权（建议γ=5）
 - 比较加权前后的训练曲线

def train_ddpm_weighted(model, dataloader, num_epochs, T=1000, snr_gamma=5.0):
 """带Min-SNR加权的DDPM训练"""
 optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

 # 预计算
 betas = linear_beta_schedule(T)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)
 sqrt_alphas_bar = torch.sqrt(alphas_bar)
 sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

 # 预计算SNR和权重
 snr = alphas_bar / (1 - alphas_bar)
 snr_clipped = torch.minimum(snr, torch.tensor(snr_gamma))
 loss_weights = snr_clipped / snr

 for epoch in range(num_epochs):
 for batch_idx, (x_0, _) in enumerate(dataloader):
 batch_size = x_0.shape[0]

 # 采样时间步
 t = torch.randint(0, T, (batch_size,), device=x_0.device)

 # 前向扩散
 epsilon = torch.randn_like(x_0)
 x_t = (sqrt_alphas_bar[t, None, None, None] * x_0 +
 sqrt_one_minus_alphas_bar[t, None, None, None] * epsilon)

 # 预测噪声
 epsilon_pred = model(x_t, t)

 # 计算加权损失
 mse_loss = (epsilon_pred - epsilon).pow(2).mean(dim=[1,2,3])
 weights = loss_weights[t]
 loss = (weights * mse_loss).mean()

 # 反向传播
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

# 关键改进：
# 1. 高SNR（低噪声）区域的权重被降低，避免过拟合细节
# 2. 低SNR（高噪声）区域保持较高权重，确保结构学习
# 3. γ参数控制截断程度，通常5-10效果较好
 **实践建议**：Min-SNR-γ加权在高分辨率图像生成中特别有效，可以显著改善生成质量。但对于低分辨率或简单数据集，简单损失可能已经足够。





## 3.5 采样算法：从理论到实践



训练好DDPM后，如何生成新的样本？这一节我们将详细介绍DDPM的采样算法，从标准的1000步采样到各种实用技巧。



### 3.5.1 标准DDPM采样



DDPM的采样过程是从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，逐步去噪直到得到清晰的图像 $\mathbf{x}_0$。



> **定义**
> DDPM采样算法


对于每一步 $t = T, T-1, ..., 1$：



 $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$$



其中：



 - $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 是训练好的噪声预测网络
 - $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 是采样噪声（当 $t > 1$ 时）
 - $\sigma_t$ 是方差，DDPM使用 $\sigma_t = \beta_t$





完整的实现代码：




@torch.no_grad()
def ddpm_sample(model, shape, num_timesteps=1000, device='cuda'):
 """
 DDPM标准采样算法

 Args:
 model: 训练好的噪声预测模型
 shape: 生成图像的形状，如 (batch_size, 3, 32, 32)
 num_timesteps: 总时间步数
 device: 计算设备

 Returns:
 生成的图像 x_0
 """
 # 预计算噪声调度
 betas = linear_beta_schedule(num_timesteps).to(device)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)
 sqrt_alphas = torch.sqrt(alphas)
 sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

 # 从纯噪声开始
 x_t = torch.randn(shape, device=device)

 # 逐步去噪
 for t in reversed(range(num_timesteps)):
 # 创建时间步张量
 t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

 # 预测噪声
 epsilon_pred = model(x_t, t_tensor)

 # 计算均值
 mean = (x_t - betas[t] / sqrt_one_minus_alphas_bar[t] * epsilon_pred) / sqrt_alphas[t]

 # 添加噪声（除了最后一步）
 if t > 0:
 noise = torch.randn_like(x_t)
 std = torch.sqrt(betas[t]) # DDPM使用β_t作为方差
 x_t = mean + std * noise
 else:
 x_t = mean

 return x_t



#### 采样过程的可视化


 为了更好地理解采样过程，让我们可视化不同时间步的中间结果：




def get_sampling_trajectory(model, num_steps_to_show=10):
 """获取DDPM采样过程的中间结果"""
 # 采样并保存中间结果
 shape = (1, 3, 32, 32)
 T = 1000

 # 选择要展示的时间步
 steps_to_show = torch.linspace(T-1, 0, num_steps_to_show, dtype=torch.long)
 intermediate_results = []

 # 初始化
 x_t = torch.randn(shape, device='cuda')
 betas = linear_beta_schedule(T).to('cuda')
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)

 # 采样过程
 for t in reversed(range(T)):
 t_tensor = torch.full((1,), t, device='cuda', dtype=torch.long)

 # 预测并更新
 epsilon_pred = model(x_t, t_tensor)
 # ... (采样步骤同上)

 # 保存中间结果
 if t in steps_to_show:
 # 将x_t映射到[0, 1]范围（用于可视化）
 img = (x_t.clamp(-1, 1) + 1) / 2
 intermediate_results.append({
 'timestep': t,
 'image': img.cpu()
 })

 return intermediate_results

# 使用示例
trajectory = get_sampling_trajectory(model, num_steps_to_show=10)
print(f"Saved {len(trajectory)} intermediate results")
for i, result in enumerate(trajectory):
 print(f"Step {i}: t={result['timestep']}, shape={result['image'].shape}")




#### 采样过程的特点



 - **前期（t ≈ 1000）**：主要恢复全局结构和大致形状
 - **中期（t ≈ 500）**：细化对象轮廓和主要特征
 - **后期（t ≈ 0）**：添加纹理细节和高频信息


 这个过程类似于艺术家作画：先勾勒轮廓，再填充颜色，最后添加细节。




#### 计算效率分析



标准DDPM采样的主要问题是速度慢。让我们分析一下计算成本：




def analyze_sampling_cost(model, batch_size=16, image_size=256):
 """分析DDPM采样的计算成本"""
 import time

 shape = (batch_size, 3, image_size, image_size)
 T = 1000

 # 测量单次前向传播时间
 x = torch.randn(shape, device='cuda')
 t = torch.randint(0, T, (batch_size,), device='cuda')

 # 预热GPU
 for _ in range(10):
 _ = model(x, t)
 torch.cuda.synchronize()

 # 计时
 start = time.time()
 num_runs = 50
 for _ in range(num_runs):
 _ = model(x, t)
 torch.cuda.synchronize()
 end = time.time()

 time_per_forward = (end - start) / num_runs
 total_time = time_per_forward * T

 print(f"图像尺寸: {image_size}×{image_size}")
 print(f"批次大小: {batch_size}")
 print(f"单次前向传播: {time_per_forward*1000:.2f} ms")
 print(f"完整采样 (1000步): {total_time:.2f} 秒")
 print(f"每秒生成图像数: {batch_size/total_time:.3f}")

 # 内存使用估计
 model_params = sum(p.numel() for p in model.parameters()) * 4 / 1024**3 # GB
 activation_memory = batch_size * 3 * image_size**2 * 4 * 50 / 1024**3 # 粗略估计
 print(f"\n内存使用:")
 print(f"模型参数: {model_params:.2f} GB")
 print(f"激活值 (估计): {activation_memory:.2f} GB")



> **定义**
> 典型性能数据

 
 
 配置
 单步时间
 总采样时间
 吞吐量
 
 
 32×32, batch=64
 ~5ms
 5秒
 12.8 图像/秒
 
 
 256×256, batch=8
 ~50ms
 50秒
 0.16 图像/秒
 
 
 512×512, batch=4
 ~200ms
 200秒
 0.02 图像/秒
 
 
 *基于RTX 3090，实际性能因模型架构而异





练习 3.5.1：实现采样进度条

修改DDPM采样函数，添加：



 - tqdm进度条显示采样进度
 - 可选的中间结果保存
 - EMA（指数移动平均）模型支持

from tqdm import tqdm

@torch.no_grad()
def ddpm_sample_with_progress(
 model,
 shape,
 num_timesteps=1000,
 device='cuda',
 use_ema=True,
 ema_model=None,
 save_intermediate=False,
 save_steps=None
):
 """增强版DDPM采样"""
 # 选择模型
 if use_ema and ema_model is not None:
 sample_model = ema_model
 else:
 sample_model = model

 sample_model.eval()

 # 预计算
 betas = linear_beta_schedule(num_timesteps).to(device)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)
 sqrt_alphas = torch.sqrt(alphas)
 sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

 # 初始化
 x_t = torch.randn(shape, device=device)
 intermediates = []

 # 采样循环
 for t in tqdm(reversed(range(num_timesteps)), desc='Sampling', total=num_timesteps):
 t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

 # 预测噪声
 epsilon_pred = sample_model(x_t, t_tensor)

 # 更新x_t
 mean = (x_t - betas[t] / sqrt_one_minus_alphas_bar[t] * epsilon_pred) / sqrt_alphas[t]

 if t > 0:
 noise = torch.randn_like(x_t)
 std = torch.sqrt(betas[t])
 x_t = mean + std * noise
 else:
 x_t = mean

 # 保存中间结果
 if save_intermediate and save_steps is not None and t in save_steps:
 intermediates.append({
 't': t,
 'x_t': x_t.cpu().clone(),
 'pred_x_0': self._predict_x0_from_eps(x_t, t, epsilon_pred)
 })

 if save_intermediate:
 return x_t, intermediates
 else:
 return x_t

def _predict_x0_from_eps(x_t, t, epsilon_pred):
 """从噪声预测恢复x_0（用于可视化）"""
 return (x_t - sqrt_one_minus_alphas_bar[t] * epsilon_pred) / sqrt_alphas_bar[t]
 **使用技巧**：



 - EMA模型通常生成质量更好，训练时应同时维护
 - 保存中间结果有助于调试和理解模型行为
 - 对于批量生成，考虑使用DataLoader风格的生成器以节省内存





### 3.5.2 采样的随机性控制



DDPM采样过程中的随机性来源于两个地方：初始噪声 $\mathbf{x}_T$ 和每步添加的噪声 $\mathbf{z}_t$。通过控制这些随机性，我们可以影响生成结果的多样性和质量。



#### 温度参数的引入



类似于其他生成模型，我们可以引入温度参数来控制采样的随机性：




def ddpm_sample_with_temperature(
 model,
 shape,
 temperature=1.0,
 noise_temperature=1.0,
 num_timesteps=1000,
 device='cuda'
):
 """
 带温度控制的DDPM采样

 Args:
 temperature: 控制初始噪声的温度
 noise_temperature: 控制每步噪声的温度
 """
 # 预计算（同前）
 betas = linear_beta_schedule(num_timesteps).to(device)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)

 # 温度调整的初始噪声
 x_t = torch.randn(shape, device=device) * temperature

 for t in reversed(range(num_timesteps)):
 t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

 # 预测噪声
 epsilon_pred = model(x_t, t_tensor)

 # 计算均值
 mean = (x_t - betas[t] / torch.sqrt(1 - alphas_bar[t]) * epsilon_pred) / torch.sqrt(alphas[t])

 if t > 0:
 # 温度调整的步进噪声
 noise = torch.randn_like(x_t) * noise_temperature
 std = torch.sqrt(betas[t])
 x_t = mean + std * noise
 else:
 x_t = mean

 return x_t



> **定义**
> 温度参数的效果



 - **temperature  1.0**：增加初始随机性，生成更多样但可能质量较低的样本
 - **noise_temperature  1.0**：增加去噪随机性，可能产生更多细节但也可能引入伪影





#### 确定性采样：DDIM预览


 一个有趣的观察是：如果我们完全去除步进噪声（设置 $\sigma_t = 0$），采样过程变成确定性的。这就是DDIM的核心思想：




def ddpm_deterministic_sample(model, shape, num_timesteps=1000, eta=0.0):
 """
 确定性或部分确定性采样
 eta=0: 完全确定性（DDIM）
 eta=1: 标准DDPM（完全随机）
 """
 x_t = torch.randn(shape, device='cuda')

 for t in reversed(range(num_timesteps)):
 # 预测噪声
 epsilon_pred = model(x_t, t)

 # 预测x_0
 x_0_pred = (x_t - torch.sqrt(1 - alphas_bar[t]) * epsilon_pred) / torch.sqrt(alphas_bar[t])

 if t > 0:
 # 计算方向指向x_{t-1}
 direction = torch.sqrt(1 - alphas_bar[t-1]) * epsilon_pred

 # 确定性部分
 x_t = torch.sqrt(alphas_bar[t-1]) * x_0_pred + direction

 # 随机部分（由eta控制）
 if eta > 0:
 noise = torch.randn_like(x_t)
 variance = eta * betas[t] * (1 - alphas_bar[t-1]) / (1 - alphas_bar[t])
 x_t = x_t + torch.sqrt(variance) * noise
 else:
 x_t = x_0_pred

 return x_t



#### 采样种子与可重复性


 对于需要可重复结果的应用，控制随机种子至关重要：




class SeededSampler:
 """可重复的采样器"""
 def __init__(self, model, device='cuda'):
 self.model = model
 self.device = device

 def sample_with_seed(self, seed, shape, **kwargs):
 """使用指定种子采样"""
 # 保存当前随机状态
 cpu_state = torch.get_rng_state()
 cuda_state = torch.cuda.get_rng_state(self.device)

 # 设置种子
 torch.manual_seed(seed)
 torch.cuda.manual_seed(seed)

 # 采样
 result = ddpm_sample(self.model, shape, device=self.device, **kwargs)

 # 恢复随机状态
 torch.set_rng_state(cpu_state)
 torch.cuda.set_rng_state(cuda_state, self.device)

 return result

 def sample_variations(self, base_seed, num_variations, shape, temperature_range=(0.8, 1.2)):
 """生成同一种子的多个变体"""
 variations = []

 for i in range(num_variations):
 # 使用相同的基础种子但不同的温度
 temp = np.linspace(temperature_range[0], temperature_range[1], num_variations)[i]

 torch.manual_seed(base_seed)
 torch.cuda.manual_seed(base_seed)

 sample = ddpm_sample_with_temperature(
 self.model, shape,
 temperature=temp,
 device=self.device
 )
 variations.append(sample)

 return torch.stack(variations)



#### 高级技巧：引导采样（Guided Sampling）


 我们可以在采样过程中加入额外的引导信号，这是条件生成的基础：




def guided_sample(model, shape, guidance_fn=None, guidance_scale=1.0):
 """
 带引导的采样
 guidance_fn: 计算引导梯度的函数
 guidance_scale: 引导强度
 """
 x_t = torch.randn(shape, device='cuda')
 x_t.requires_grad = True

 for t in reversed(range(num_timesteps)):
 # 标准DDPM更新
 with torch.no_grad():
 epsilon_pred = model(x_t, t)
 mean = compute_mean(x_t, epsilon_pred, t)
 std = torch.sqrt(betas[t])

 # 计算引导梯度
 if guidance_fn is not None and t > 0:
 # 计算引导损失
 guidance_loss = guidance_fn(x_t, t)

 # 计算梯度
 grad = torch.autograd.grad(guidance_loss, x_t)[0]

 # 应用引导（注意符号：我们要最小化损失）
 mean = mean - guidance_scale * std**2 * grad

 # 更新x_t
 if t > 0:
 noise = torch.randn_like(x_t)
 x_t = mean + std * noise
 else:
 x_t = mean

 x_t = x_t.detach().requires_grad_(True)

 return x_t.detach()

# 示例：类别引导
def classifier_guidance(x_t, t, classifier, target_class):
 """使用分类器引导生成特定类别"""
 logits = classifier(x_t, t)
 log_prob = F.log_softmax(logits, dim=1)
 return -log_prob[:, target_class].sum() # 负对数概率作为损失




练习 3.5.2：探索温度参数的影响
 实现一个实验，系统地探索不同温度参数对生成结果的影响：



 - 固定种子，改变temperature（0.5, 0.7, 1.0, 1.3, 1.5）
 - 固定种子，改变noise_temperature（0, 0.5, 1.0, 1.5）
 - 可视化结果并计算多样性指标（如平均像素方差）

def temperature_ablation_study(model, seed=42):
 """温度参数消融实验"""
 shape = (1, 3, 32, 32)

 # 实验1：初始温度的影响
 init_temps = [0.5, 0.7, 1.0, 1.3, 1.5]
 init_results = []

 for temp in init_temps:
 torch.manual_seed(seed)
 torch.cuda.manual_seed(seed)

 sample = ddpm_sample_with_temperature(
 model, shape,
 temperature=temp,
 noise_temperature=1.0
 )
 init_results.append(sample)

 # 实验2：噪声温度的影响
 noise_temps = [0.0, 0.5, 1.0, 1.5]
 noise_results = []

 for noise_temp in noise_temps:
 torch.manual_seed(seed)
 torch.cuda.manual_seed(seed)

 sample = ddpm_sample_with_temperature(
 model, shape,
 temperature=1.0,
 noise_temperature=noise_temp
 )
 noise_results.append(sample)

 # 分析结果
 results = {
 'init_temperature': {},
 'noise_temperature': {}
 }

 # 计算初始温度的影响
 print("初始温度对图像统计特性的影响:")
 for temp, img in zip(init_temps, init_results):
 stats = {
 'mean': img.mean().item(),
 'std': img.std().item(),
 'min': img.min().item(),
 'max': img.max().item()
 }
 results['init_temperature'][temp] = stats
 print(f" T_init={temp}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

 # 计算噪声温度的影响
 print("\n噪声温度对图像统计特性的影响:")
 for temp, img in zip(noise_temps, noise_results):
 stats = {
 'mean': img.mean().item(),
 'std': img.std().item(),
 'min': img.min().item(),
 'max': img.max().item()
 }
 results['noise_temperature'][temp] = stats
 print(f" T_noise={temp}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

 return results

# 额外分析：多次采样的多样性
def diversity_analysis(model, num_samples=100):
 """分析不同温度设置下的样本多样性"""
 shape = (num_samples, 3, 32, 32)

 # 标准采样
 samples_standard = ddpm_sample(model, shape)

 # 低温采样
 samples_low_temp = ddpm_sample_with_temperature(
 model, shape, temperature=0.7, noise_temperature=0.7
 )

 # 计算成对距离
 def pairwise_l2_distance(samples):
 # 展平样本
 flat = samples.view(num_samples, -1)
 # 计算成对L2距离
 distances = torch.cdist(flat, flat, p=2)
 # 取上三角部分（避免重复）
 mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
 return distances[mask].mean().item()

 div_standard = pairwise_l2_distance(samples_standard)
 div_low_temp = pairwise_l2_distance(samples_low_temp)

 print(f"标准采样的平均成对距离: {div_standard:.4f}")
 print(f"低温采样的平均成对距离: {div_low_temp:.4f}")
 print(f"多样性降低比例: {(1 - div_low_temp/div_standard)*100:.1f}%")
 **关键发现**：



 - 降低初始温度会使生成结果更接近"平均"图像，减少极端情况
 - noise_temperature=0 会产生过度平滑的结果，丢失纹理细节
 - 适度降低温度（0.7-0.9）通常能提高感知质量，但会牺牲多样性
 - 对于特定应用，需要在质量和多样性之间找到平衡





### 3.5.3 常见问题与调试技巧



DDPM采样过程中可能遇到各种问题。本节总结常见问题及其解决方案，帮助你快速定位和修复问题。



#### 问题1：生成结果全是噪声



> **定义**
> 症状与原因



 - **症状**：采样结果看起来像随机噪声，没有任何结构
 - **可能原因**：


 模型未正确加载或权重损坏
 - 噪声调度计算错误
 - 时间步编码错误
 - 输入归一化不匹配


 






# 调试步骤1：验证模型预测
def debug_model_predictions(model, device='cuda'):
 """检查模型在不同时间步的预测"""
 # 创建测试输入
 x = torch.randn(1, 3, 32, 32, device=device)

 # 测试几个关键时间步
 test_timesteps = [0, 250, 500, 750, 999]

 for t in test_timesteps:
 t_tensor = torch.tensor([t], device=device)
 with torch.no_grad():
 pred = model(x, t_tensor)

 print(f"t={t}:")
 print(f" Input stats: mean={x.mean():.4f}, std={x.std():.4f}")
 print(f" Pred stats: mean={pred.mean():.4f}, std={pred.std():.4f}")

 # 预测应该接近标准正态分布
 if abs(pred.mean()) > 0.5 or abs(pred.std() - 1.0) > 0.5:
 print(" ⚠️ 警告：预测统计量异常！")

# 调试步骤2：验证噪声调度
def debug_noise_schedule(num_timesteps=1000):
 """检查噪声调度的合理性"""
 betas = linear_beta_schedule(num_timesteps)
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)

 print("噪声调度检查:")
 print(f"β_0 = {betas[0]:.6f}, β_T = {betas[-1]:.6f}")
 print(f"ᾱ_0 = {alphas_bar[0]:.6f}, ᾱ_T = {alphas_bar[-1]:.6f}")

 # 检查关键属性
 if alphas_bar[-1] > 0.01:
 print("⚠️ 警告：ᾱ_T 太大，最终噪声水平不够")
 if betas[0] > 0.01:
 print("⚠️ 警告：β_0 太大，初始破坏太严重")

 # 检查单调性
 if not torch.all(alphas_bar[1:] 



#### 问题2：生成结果模糊或过度平滑




##### 常见原因及解决方案



 - **方差设置过小**：检查是否使用了过小的 $\sigma_t$
 - **提前停止采样**：确保完成所有1000步（或设定的步数）
 - **模型过拟合到均值**：可能需要调整训练时的噪声调度
 - **数值精度问题**：使用FP16时某些操作可能损失精度






```python
# 诊断过度平滑问题
def diagnose_smoothness(model, num_samples=10):
 """诊断生成结果的平滑度问题"""
 samples = []

 # 生成多个样本
 for _ in range(num_samples):
 sample = ddpm_sample(model, (1, 3, 32, 32))
 samples.append(sample)

 samples = torch.cat(samples, dim=0)

 # 计算高频信息
 def compute_high_freq_energy(images):
 # 使用Sobel滤波器检测边缘
 sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
 dtype=torch.float32).view(1, 1, 3, 3)
 sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
 dtype=torch.float32).view(1, 1, 3, 3)

 # 转换为灰度
 gray = images.mean(dim=1, keepdim=True)

 # 计算梯度
 edges_x = F.conv2d(gray, sobel_x, padding=1)
 edges_y = F.conv2d(gray, sobel_y, padding=1)
 edges = torch.sqrt(edges_x**2 + edges_y**2)

 return edges.mean().item()

 # 与真实数据对比
 real_data = next(iter(train_loader))[0][:num_samples]

 gen_hf = compute_high_freq_energy(samples)
 real_hf = compute_high_freq_energy(real_data)

 print(f"生成样本的高频能量: {gen_hf:.4f}")
 print(f"真实数据的高频能量: {real_hf:.4f}")
 print(f"比率: {gen_hf/real_hf:.2f}")

 if gen_hf/real_hf DDPM的主要性能瓶颈在于需要1000步迭代，每步都需要通过U-Net进行一次前向传播。典型的性能特征：



 - **预计算阶段**：约0.1秒，包括噪声调度的计算
 - **模型推理**：每步15-50ms（取决于模型大小和GPU），总计15-50秒
 - **更新计算**：每步1-2ms，相对可忽略




**优化建议**：



 - 使用更小的模型架构（减少通道数或层数）
 - 启用混合精度推理（torch.cuda.amp）
 - 使用torch.compile()进行图优化（PyTorch 2.0+）
 - 采用DDIM等快速采样方法（可减少到50步以下）
 - 批量生成以提高GPU利用率





#### 问题4：内存溢出（OOM）




# 内存友好的批量采样
def memory_efficient_batch_sampling(model, total_samples, batch_size=16,
 image_shape=(3, 32, 32)):
 """内存高效的批量采样"""
 all_samples = []

 # 分批生成
 num_batches = (total_samples + batch_size - 1) // batch_size

 for i in tqdm(range(num_batches), desc="Batch sampling"):
 current_batch_size = min(batch_size, total_samples - i * batch_size)
 shape = (current_batch_size,) + image_shape

 # 生成当前批次
 with torch.cuda.amp.autocast(): # 使用混合精度节省内存
 samples = ddpm_sample(model, shape)

 # 立即移到CPU以释放GPU内存
 all_samples.append(samples.cpu())

 # 清理GPU缓存
 if i % 10 == 0:
 torch.cuda.empty_cache()

 return torch.cat(all_samples, dim=0)

# 诊断内存使用
def diagnose_memory_usage(model, batch_sizes=[1, 2, 4, 8, 16]):
 """诊断不同批次大小的内存使用"""
 import gc

 for bs in batch_sizes:
 torch.cuda.empty_cache()
 gc.collect()

 try:
 # 记录初始内存
 init_mem = torch.cuda.memory_allocated() / 1024**3

 # 尝试采样
 shape = (bs, 3, 256, 256) # 使用较大尺寸测试
 _ = ddpm_sample(model, shape, num_timesteps=50) # 只测试50步

 # 记录峰值内存
 peak_mem = torch.cuda.max_memory_allocated() / 1024**3

 print(f"Batch size {bs}: 峰值内存 {peak_mem:.2f}GB "
 f"(增加 {peak_mem - init_mem:.2f}GB)")

 except torch.cuda.OutOfMemoryError:
 print(f"Batch size {bs}: OOM!")
 break
 finally:
 torch.cuda.empty_cache()



#### 可视化调试工具




```python
def analyze_sampling_debug(model):
 """分析采样过程用于调试"""
 # 设置
 shape = (1, 3, 32, 32)
 checkpoints = [999, 800, 600, 400, 200, 100, 50, 20, 10, 0]

 # 收集数据
 x_t = torch.randn(shape, device='cuda')
 debug_data = {
 'x_t_history': [x_t.cpu()],
 'pred_x0_history': [],
 'noise_pred_history': [],
 'noise_stats': []
 }

 # 采样并记录
 betas = linear_beta_schedule(1000).cuda()
 alphas = 1 - betas
 alphas_bar = torch.cumprod(alphas, dim=0)

 for t in reversed(range(1000)):
 t_tensor = torch.tensor([t], device='cuda')

 # 预测
 epsilon_pred = model(x_t, t_tensor)

 # 预测的x_0
 pred_x0 = (x_t - torch.sqrt(1 - alphas_bar[t]) * epsilon_pred) / torch.sqrt(alphas_bar[t])

 # 更新
 mean = (x_t - betas[t] / torch.sqrt(1 - alphas_bar[t]) * epsilon_pred) / torch.sqrt(alphas[t])
 if t > 0:
 noise = torch.randn_like(x_t)
 x_t = mean + torch.sqrt(betas[t]) * noise
 else:
 x_t = mean

 # 记录检查点
 if t in checkpoints:
 debug_data['x_t_history'].append(x_t.cpu())
 debug_data['pred_x0_history'].append(pred_x0.cpu())
 debug_data['noise_pred_history'].append(epsilon_pred.cpu())
 debug_data['noise_stats'].append({
 't': t,
 'mean': epsilon_pred.mean().item(),
 'std': epsilon_pred.std().item(),
 'min': epsilon_pred.min().item(),
 'max': epsilon_pred.max().item()
 })

 # 打印分析结果
 print("采样过程调试分析:")
 print("==================")
 print("\n噪声预测统计:")
 print("Timestep | Mean | Std | Min | Max")
 print("-" * 55)
 for stats in debug_data['noise_stats']:
 print(f"{stats['t']:8d} | {stats['mean']:9.6f} | {stats['std']:9.6f} | {stats['min']:9.6f} | {stats['max']:9.6f}")

 # 检查x_0预测的稳定性
 print("\nx_0预测稳定性分析:")
 for i, (t, x0) in enumerate(zip(checkpoints[:-1], debug_data['pred_x0_history'])):
 x0_range = x0.max().item() - x0.min().item()
 x0_clipped = (x0  1).sum().item()
 total_pixels = x0.numel()
 clip_ratio = x0_clipped / total_pixels
 print(f"t={t:3d}: range={x0_range:.3f}, clipped pixels={clip_ratio:.1%}")

 return debug_data
```





练习 3.5.3：实现采样质量诊断工具
 创建一个综合诊断工具，能够：



 - 自动检测常见的采样问题
 - 生成诊断报告
 - 提供具体的修复建议

class DDPMSamplingDiagnostics:
 """DDPM采样综合诊断工具"""

 def __init__(self, model, device='cuda'):
 self.model = model
 self.device = device
 self.diagnostics = {}

 def run_full_diagnostics(self, num_samples=5):
 """运行完整诊断"""
 print("=== DDPM采样诊断开始 ===\n")

 # 1. 模型基础检查
 self._check_model_basics()

 # 2. 噪声调度检查
 self._check_noise_schedule()

 # 3. 采样质量检查
 self._check_sampling_quality(num_samples)

 # 4. 性能检查
 self._check_performance()

 # 5. 生成报告
 self._generate_report()

 def _check_model_basics(self):
 """检查模型基础设置"""
 print("1. 检查模型基础设置...")

 # 检查模型是否在eval模式
 if self.model.training:
 self.diagnostics['model_mode'] = 'WARNING: 模型在训练模式'
 else:
 self.diagnostics['model_mode'] = 'OK: 模型在评估模式'

 # 检查参数统计
 params = []
 for p in self.model.parameters():
 params.append(p.data.flatten())
 params = torch.cat(params)

 param_mean = params.mean().item()
 param_std = params.std().item()

 if abs(param_mean) > 1.0 or param_std > 10.0:
 self.diagnostics['param_stats'] = f'WARNING: 参数统计异常 (mean={param_mean:.3f}, std={param_std:.3f})'
 else:
 self.diagnostics['param_stats'] = 'OK: 参数统计正常'

 def _check_noise_schedule(self):
 """检查噪声调度"""
 print("2. 检查噪声调度...")

 betas = linear_beta_schedule(1000)
 alphas_bar = torch.cumprod(1 - betas, dim=0)

 # 检查端点
 if alphas_bar[0] 0.01:
 self.diagnostics['schedule_end'] = f'WARNING: α̅_T={alphas_bar[-1]:.4f} 太大'
 else:
 self.diagnostics['schedule_end'] = 'OK: 终点正常'

 def _check_sampling_quality(self, num_samples):
 """检查采样质量"""
 print(f"3. 检查采样质量 (生成{num_samples}个样本)...")

 samples = []
 for _ in range(num_samples):
 sample = ddpm_sample(self.model, (1, 3, 32, 32), device=self.device)
 samples.append(sample)
 samples = torch.cat(samples)

 # 检查输出范围
 sample_min = samples.min().item()
 sample_max = samples.max().item()

 if sample_min 3:
 self.diagnostics['output_range'] = f'WARNING: 输出范围异常 [{sample_min:.2f}, {sample_max:.2f}]'
 else:
 self.diagnostics['output_range'] = 'OK: 输出范围正常'

 # 检查多样性
 if num_samples > 1:
 diversity = samples.std(dim=0).mean().item()
 if diversity 60:
 self.diagnostics['performance'] = f'WARNING: 预计采样时间过长 ({total_time:.1f}秒)'
 else:
 self.diagnostics['performance'] = f'OK: 预计采样时间 {total_time:.1f}秒'

 def _generate_report(self):
 """生成诊断报告"""
 print("\n=== 诊断报告 ===")

 warnings = 0
 for key, value in self.diagnostics.items():
 if value.startswith('WARNING'):
 print(f"❌ {value}")
 warnings += 1
 else:
 print(f"✅ {value}")

 print(f"\n总结: {len(self.diagnostics)}项检查, {warnings}个警告")

 if warnings > 0:
 print("\n建议的修复步骤:")
 if 'model_mode' in self.diagnostics and 'WARNING' in self.diagnostics['model_mode']:
 print("- 调用 model.eval() 切换到评估模式")
 if 'schedule_end' in self.diagnostics and 'WARNING' in self.diagnostics['schedule_end']:
 print("- 增加总时间步数或调整beta_end")
 if 'diversity' in self.diagnostics and 'WARNING' in self.diagnostics['diversity']:
 print("- 检查模型是否过拟合或模式崩塌")
 if 'performance' in self.diagnostics and 'WARNING' in self.diagnostics['performance']:
 print("- 考虑使用DDIM或其他快速采样方法")

# 使用示例
diagnostics = DDPMSamplingDiagnostics(model)
diagnostics.run_full_diagnostics()
 **诊断工具的扩展**：可以添加更多检查项，如：



 - 检查是否使用了EMA模型
 - 验证条件生成的正确性
 - 检测特定的视觉伪影（棋盘效应、色彩偏移等）
 - 与真实数据分布的统计对比





## 3.6 完整实现：构建你的第一个DDPM


本节将把前面学到的所有概念整合成一个完整的DDPM实现。我们将构建一个可以在MNIST数据集上训练的完整系统。



### 3.6.1 模型架构


首先，让我们实现一个适合DDPM的U-Net架构。这个架构需要：



 - 接受带噪声的图像 $x_t$ 作为输入
 - 接受时间步 $t$ 作为条件信息
 - 输出预测的噪声 $\epsilon_\theta(x_t, t)$




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
 """正弦位置编码，用于时间步嵌入"""
 def __init__(self, dim):
 super().__init__()
 self.dim = dim

 def forward(self, time):
 device = time.device
 half_dim = self.dim // 2
 embeddings = math.log(10000) / (half_dim - 1)
 embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
 embeddings = time[:, None] * embeddings[None, :]
 embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
 return embeddings

class ResidualBlock(nn.Module):
 """带时间嵌入的残差块"""
 def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
 super().__init__()
 self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
 self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
 self.time_emb = nn.Linear(time_emb_dim, out_channels)
 self.dropout = nn.Dropout(dropout)
 self.norm1 = nn.GroupNorm(8, out_channels)
 self.norm2 = nn.GroupNorm(8, out_channels)
 self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

 def forward(self, x, t):
 h = self.conv1(x)
 h = self.norm1(h)
 h = F.silu(h)

 # 添加时间嵌入
 h = h + self.time_emb(F.silu(t))[:, :, None, None]

 h = self.conv2(h)
 h = self.norm2(h)
 h = F.silu(h)
 h = self.dropout(h)

 return h + self.shortcut(x)

class AttentionBlock(nn.Module):
 """自注意力块"""
 def __init__(self, channels, num_heads=4):
 super().__init__()
 self.num_heads = num_heads
 self.norm = nn.GroupNorm(8, channels)
 self.qkv = nn.Conv2d(channels, channels * 3, 1)
 self.proj = nn.Conv2d(channels, channels, 1)

 def forward(self, x):
 B, C, H, W = x.shape
 h = self.norm(x)
 qkv = self.qkv(h)
 q, k, v = qkv.chunk(3, dim=1)

 # 重塑为多头格式
 q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
 k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
 v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

 # 计算注意力
 scale = (C // self.num_heads) ** -0.5
 attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
 out = torch.matmul(attn, v)

 # 重塑回原始格式
 out = out.transpose(2, 3).contiguous().view(B, C, H, W)
 return x + self.proj(out)



 架构设计要点


 - **时间嵌入**：使用正弦位置编码将离散时间步转换为连续表示
 - **残差连接**：每个块都包含残差连接，有助于梯度流动
 - **注意力机制**：在低分辨率特征图上使用自注意力，捕获长程依赖
 - **GroupNorm**：使用组归一化而非批归一化，更适合小批量训练





#### 轻量级DDPM U-Net

 对于简单任务（如MNIST），可以使用更轻量的架构：



class SimpleDDPMUNet(nn.Module):
 """轻量级DDPM U-Net，适用于MNIST等简单数据集"""
 def __init__(self, image_channels=1, n_channels=32, ch_mults=(1, 2, 2, 4),
 n_blocks=2):
 super().__init__()

 # 时间嵌入
 self.time_emb = nn.Sequential(
 SinusoidalPositionalEmbedding(n_channels),
 nn.Linear(n_channels, n_channels * 4),
 nn.GELU(),
 nn.Linear(n_channels * 4, n_channels * 4)
 )

 # 输入层
 self.conv_in = nn.Conv2d(image_channels, n_channels, 3, padding=1)

 # 下采样
 self.downs = nn.ModuleList()
 chs = [n_channels]
 now_ch = n_channels

 for i, mult in enumerate(ch_mults):
 out_ch = n_channels * mult
 for _ in range(n_blocks):
 self.downs.append(ResidualBlock(now_ch, out_ch, n_channels * 4))
 now_ch = out_ch
 chs.append(now_ch)

 if i  0:
 self.ups.append(nn.ConvTranspose2d(now_ch, now_ch, 4, stride=2, padding=1))

 # 输出层
 self.conv_out = nn.Sequential(
 nn.GroupNorm(8, now_ch),
 nn.SiLU(),
 nn.Conv2d(now_ch, image_channels, 3, padding=1)
 )

 def forward(self, x, t):
 # 获取时间嵌入
 t = self.time_emb(t)

 # 初始卷积
 h = self.conv_in(x)

 # 下采样
 hs = [h]
 for layer in self.downs:
 if isinstance(layer, ResidualBlock):
 h = layer(h, t)
 else:
 h = layer(h)
 hs.append(h)

 # 中间层
 for layer in self.middle:
 h = layer(h, t)

 # 上采样
 for layer in self.ups:
 if isinstance(layer, ResidualBlock):
 h = layer(torch.cat([h, hs.pop()], dim=1), t)
 else:
 h = layer(h)

 # 输出
 return self.conv_out(h)




练习 3.6.1：模型参数计算
 实现一个函数来计算U-Net模型的参数量，并比较不同配置的模型大小。

def count_parameters(model):
 """计算模型参数量"""
 return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_model_sizes():
 """比较不同模型配置的参数量"""
 configs = [
 {"name": "Tiny", "n_channels": 16, "ch_mults": (1, 2, 2)},
 {"name": "Small", "n_channels": 32, "ch_mults": (1, 2, 2, 4)},
 {"name": "Base", "n_channels": 64, "ch_mults": (1, 2, 4, 8)},
 {"name": "Large", "n_channels": 128, "ch_mults": (1, 2, 4, 8)}
 ]

 for config in configs:
 model = SimpleDDPMUNet(
 n_channels=config["n_channels"],
 ch_mults=config["ch_mults"]
 )
 params = count_parameters(model)
 print(f"{config['name']}: {params:,} parameters ({params/1e6:.2f}M)")

# 输出示例：
# Tiny: 461,729 parameters (0.46M)
# Small: 3,652,481 parameters (3.65M)
# Base: 35,742,785 parameters (35.74M)
# Large: 142,836,097 parameters (142.84M)





### 3.6.2 训练循环

 现在让我们实现完整的DDPM训练循环。这个实现包含了前面章节介绍的所有关键组件。



import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class DDPMTrainer:
 """DDPM训练器"""
 def __init__(self, model, device='cuda', num_timesteps=1000,
 beta_start=1e-4, beta_end=0.02, loss_type='l2'):
 self.model = model.to(device)
 self.device = device
 self.num_timesteps = num_timesteps
 self.loss_type = loss_type

 # 设置噪声调度
 self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
 self.alphas = 1 - self.betas
 self.alphas_bar = torch.cumprod(self.alphas, dim=0)
 self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
 self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

 # 用于采样的预计算值
 self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
 self.sqrt_alphas_bar_prev = torch.sqrt(
 torch.cat([torch.tensor([1.0]).to(device), self.alphas_bar[:-1]])
 )
 self.sqrt_one_minus_alphas_bar_prev = torch.sqrt(
 1 - torch.cat([torch.tensor([1.0]).to(device), self.alphas_bar[:-1]])
 )
 self.posterior_variance = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)

 def forward_diffusion(self, x_0, t, noise=None):
 """前向扩散过程"""
 if noise is None:
 noise = torch.randn_like(x_0)

 sqrt_alphas_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
 sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)

 x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise
 return x_t, noise

 def compute_loss(self, x_0, t):
 """计算训练损失"""
 noise = torch.randn_like(x_0)
 x_t, _ = self.forward_diffusion(x_0, t, noise)
 noise_pred = self.model(x_t, t)

 if self.loss_type == 'l2':
 loss = torch.nn.functional.mse_loss(noise_pred, noise)
 elif self.loss_type == 'l1':
 loss = torch.nn.functional.l1_loss(noise_pred, noise)
 else:
 raise ValueError(f"Unknown loss type: {self.loss_type}")

 return loss

 def train_step(self, batch, optimizer):
 """单步训练"""
 x_0 = batch[0].to(self.device)
 batch_size = x_0.shape[0]

 # 随机采样时间步
 t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

 # 计算损失
 loss = self.compute_loss(x_0, t)

 # 反向传播
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 return loss.item()

 @torch.no_grad()
 def sample(self, num_samples, image_size=(1, 28, 28), return_trajectory=False):
 """DDPM采样"""
 self.model.eval()

 # 从纯噪声开始
 x_t = torch.randn(num_samples, *image_size, device=self.device)

 trajectory = [x_t.cpu()] if return_trajectory else None

 # 逐步去噪
 for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
 t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

 # 预测噪声
 noise_pred = self.model(x_t, t_batch)

 # 计算均值
 beta_t = self.betas[t]
 sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
 sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

 mean = sqrt_recip_alpha_t * (
 x_t - beta_t / sqrt_one_minus_alpha_bar_t * noise_pred
 )

 # 添加噪声（除了最后一步）
 if t > 0:
 noise = torch.randn_like(x_t)
 posterior_variance_t = self.posterior_variance[t]
 x_t = mean + torch.sqrt(posterior_variance_t) * noise
 else:
 x_t = mean

 if return_trajectory and t % 100 == 0:
 trajectory.append(x_t.cpu())

 self.model.train()

 if return_trajectory:
 return x_t, trajectory
 return x_t

def train_ddpm(model, train_loader, num_epochs=100, lr=2e-4,
 device='cuda', save_interval=10):
 """完整的DDPM训练流程"""
 trainer = DDPMTrainer(model, device=device)
 optimizer = optim.Adam(model.parameters(), lr=lr)

 # 训练历史
 losses = []

 for epoch in range(num_epochs):
 epoch_losses = []
 pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

 for batch in pbar:
 loss = trainer.train_step(batch, optimizer)
 epoch_losses.append(loss)
 pbar.set_postfix({'loss': f"{loss:.4f}"})

 avg_loss = np.mean(epoch_losses)
 losses.append(avg_loss)
 print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

 # 定期生成样本
 if (epoch + 1) % save_interval == 0:
 samples = trainer.sample(16)
 save_samples(samples, epoch + 1)

 # 保存检查点
 torch.save({
 'epoch': epoch,
 'model_state_dict': model.state_dict(),
 'optimizer_state_dict': optimizer.state_dict(),
 'loss': avg_loss,
 }, f'ddpm_checkpoint_epoch_{epoch+1}.pt')

 return trainer, losses

def save_samples(samples, epoch, save_dir='./samples'):
 """保存生成的样本"""
 import os
 os.makedirs(save_dir, exist_ok=True)

 # 保存为PyTorch张量格式
 torch.save(samples, os.path.join(save_dir, f'samples_epoch_{epoch}.pt'))

 # 可选：保存为单独的图像文件
 if samples.shape[1] == 1: # 单通道图像
 from torchvision.utils import save_image
 # 将值域从[-1, 1]映射到[0, 1]
 samples_normalized = (samples + 1) / 2
 save_image(samples_normalized,
 os.path.join(save_dir, f'grid_epoch_{epoch}.png'),
 nrow=4, normalize=False)

 print(f"已保存 {len(samples)} 个样本到 {save_dir}")



#### 使用示例：在MNIST上训练DDPM



```python
# 准备数据集
transform = transforms.Compose([
 transforms.ToTensor(),
 transforms.Normalize((0.5,), (0.5,)) # 归一化到[-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True,
 download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128,
 shuffle=True, num_workers=4)

# 创建模型
model = SimpleDDPMUNet(
 image_channels=1,
 n_channels=32,
 ch_mults=(1, 2, 2, 4),
 n_blocks=2
)

# 训练模型
trainer, losses = train_ddpm(
 model=model,
 train_loader=train_loader,
 num_epochs=50,
 lr=2e-4,
 device='cuda' if torch.cuda.is_available() else 'cpu',
 save_interval=10
)

# 生成新样本
new_samples = trainer.sample(64, image_size=(1, 28, 28))

# 分析训练损失
print("训练损失统计:")
print(f" 初始损失: {losses[0]:.4f}")
print(f" 最终损失: {losses[-1]:.4f}")
print(f" 最低损失: {min(losses):.4f} (Epoch {losses.index(min(losses)) + 1})")
print(f" 损失下降: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
```




 训练技巧


 - **学习率调度**：使用余弦退火或线性衰减可以提升训练稳定性
 - **EMA**：使用指数移动平均（EMA）可以获得更稳定的生成质量
 - **梯度裁剪**：防止梯度爆炸，特别是在训练初期
 - **混合精度训练**：使用FP16可以加速训练并减少显存占用





#### 高级训练技术



```python
class EMA:
 """指数移动平均"""
 def __init__(self, model, decay=0.995):
 self.model = model
 self.decay = decay
 self.shadow = {}
 self.backup = {}
 self.register()

 def register(self):
 for name, param in self.model.named_parameters():
 if param.requires_grad:
 self.shadow[name] = param.data.clone()

 def update(self):
 for name, param in self.model.named_parameters():
 if param.requires_grad:
 self.shadow[name] = self.decay * self.shadow[name] + \
 (1 - self.decay) * param.data

 def apply_shadow(self):
 for name, param in self.model.named_parameters():
 if param.requires_grad:
 self.backup[name] = param.data
 param.data = self.shadow[name]

 def restore(self):
 for name, param in self.model.named_parameters():
 if param.requires_grad:
 param.data = self.backup[name]
 self.backup = {}

def train_ddpm_with_ema(model, train_loader, num_epochs=100):
 """带EMA的DDPM训练"""
 trainer = DDPMTrainer(model)
 optimizer = optim.Adam(model.parameters(), lr=2e-4)
 scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
 ema = EMA(model)

 scaler = torch.cuda.amp.GradScaler() # 混合精度训练

 for epoch in range(num_epochs):
 for batch in train_loader:
 x_0 = batch[0].to(trainer.device)
 batch_size = x_0.shape[0]
 t = torch.randint(0, trainer.num_timesteps, (batch_size,),
 device=trainer.device)

 # 混合精度训练
 with torch.cuda.amp.autocast():
 loss = trainer.compute_loss(x_0, t)

 optimizer.zero_grad()
 scaler.scale(loss).backward()

 # 梯度裁剪
 scaler.unscale_(optimizer)
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

 scaler.step(optimizer)
 scaler.update()

 # 更新EMA
 ema.update()

 scheduler.step()

 # 使用EMA权重生成样本
 if (epoch + 1) % 10 == 0:
 ema.apply_shadow()
 samples = trainer.sample(16)
 save_samples(samples, epoch + 1)
 ema.restore()

 return trainer, ema
```





练习 3.6.2：实现学习率预热
 修改训练代码，添加学习率预热（warmup）功能，在训练初期逐渐增加学习率。

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
 """带预热的余弦退火调度器"""
 def __init__(self, optimizer, warmup_epochs, total_epochs,
 warmup_lr=1e-5, base_lr=2e-4, min_lr=1e-6):
 self.warmup_epochs = warmup_epochs
 self.total_epochs = total_epochs
 self.warmup_lr = warmup_lr
 self.base_lr = base_lr
 self.min_lr = min_lr
 super().__init__(optimizer)

 def get_lr(self):
 if self.last_epoch





### 3.6.3 评估与可视化

 评估生成模型的质量是一个重要但富有挑战性的任务。本节介绍常用的评估指标和可视化方法。


#### 常用评估指标


 生成模型评估指标


 - **FID (Fréchet Inception Distance)**：衡量生成分布与真实分布的距离
 - **IS (Inception Score)**：评估生成样本的质量和多样性
 - **LPIPS**：感知相似度，更符合人类视觉感知
 - **Precision/Recall**：分别衡量质量和覆盖度





#### FID计算实现



import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

class FIDCalculator:
 """FID (Fréchet Inception Distance) 计算器"""
 def __init__(self, device='cuda'):
 self.device = device
 self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
 self.inception.eval()
 # 移除最后的全连接层
 self.inception.fc = torch.nn.Identity()

 @torch.no_grad()
 def extract_features(self, images):
 """提取Inception特征"""
 # 确保图像大小至少为299x299（Inception-v3要求）
 if images.shape[2] 



#### Inception Score实现



```python
class InceptionScore:
 """Inception Score计算器"""
 def __init__(self, device='cuda'):
 self.device = device
 self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
 self.inception.eval()

 @torch.no_grad()
 def compute_is(self, images, batch_size=32, splits=10):
 """计算Inception Score

 Args:
 images: 生成的图像张量
 batch_size: 批处理大小
 splits: 用于计算IS的分割数

 Returns:
 is_mean: IS均值
 is_std: IS标准差
 """
 # 获取预测
 preds = []
 for i in range(0, len(images), batch_size):
 batch = images[i:i+batch_size].to(self.device)

 # 调整大小和通道
 if batch.shape[2] = num_samples:
 break
 real_samples = torch.cat(real_samples, dim=0)[:num_samples]

 # 计算FID
 print("计算FID...")
 fid_score = self.fid_calculator.compute_fid_from_samples(
 real_samples, generated_samples, batch_size=batch_size
 )

 # 计算IS
 print("计算Inception Score...")
 is_mean, is_std = self.is_calculator.compute_is(
 generated_samples, batch_size=batch_size
 )

 # 计算样本多样性
 diversity = self.compute_diversity(generated_samples)

 results = {
 'fid': fid_score,
 'is_mean': is_mean,
 'is_std': is_std,
 'diversity': diversity
 }

 return results, generated_samples

 def compute_diversity(self, samples):
 """计算样本多样性（使用LPIPS或简单的L2距离）"""
 # 简化版：使用L2距离
 n_samples = min(1000, len(samples))
 indices = torch.randperm(len(samples))[:n_samples]
 subset = samples[indices]

 # 计算两两之间的L2距离
 distances = []
 for i in range(n_samples):
 for j in range(i+1, n_samples):
 dist = torch.norm(subset[i] - subset[j], p=2)
 distances.append(dist.item())

 return np.mean(distances)

 def save_results(self, results, samples, save_path='evaluation_results.pt'):
 """保存评估结果"""
 # 保存样本和评估指标
 torch.save({
 'samples': samples,
 'metrics': results,
 'timestamp': np.datetime64('now')
 }, save_path)

 # 打印评估报告
 print("\n" + "="*50)
 print("评估结果报告")
 print("="*50)
 print(f"FID Score: {results['fid']:.2f} (越低越好)")
 print(f"Inception Score: {results['is_mean']:.2f} ± {results['is_std']:.2f} (越高越好)")
 print(f"Diversity Score: {results['diversity']:.4f} (越高越好)")
 print(f"\n结果已保存到: {save_path}")

 return results
```




#### 使用示例



```python
# 创建评估器
evaluator = DDPMEvaluator(trainer, test_loader)

# 运行完整评估
results, generated_samples = evaluator.evaluate(num_samples=5000)

# 打印结果
print(f"FID Score: {results['fid']:.2f}")
print(f"Inception Score: {results['is_mean']:.2f} ± {results['is_std']:.2f}")
print(f"Diversity Score: {results['diversity']:.4f}")

# 保存结果
evaluator.save_results(results, generated_samples)

# 分析采样轨迹
def analyze_sampling_trajectory(trainer, num_steps_show=10):
 """分析采样轨迹"""
 # 生成带轨迹的样本
 samples, trajectory = trainer.sample(4, return_trajectory=True)

 # 选择要显示的步骤
 total_steps = len(trajectory)
 step_indices = np.linspace(0, total_steps-1, num_steps_show, dtype=int)

 print("\n采样轨迹分析:")
 print("="*50)
 print(f"总步数: {trainer.num_timesteps}")
 print(f"轨迹采样点: {len(step_indices)}")

 # 分析每个阶段的统计特性
 for i, step_idx in enumerate(step_indices):
 t = trainer.num_timesteps - step_idx * 100 if step_idx > 0 else trainer.num_timesteps
 img_batch = trajectory[step_idx]

 stats = {
 'mean': img_batch.mean().item(),
 'std': img_batch.std().item(),
 'min': img_batch.min().item(),
 'max': img_batch.max().item()
 }

 print(f"\n步骤 {i+1}/{num_steps_show} (t={t}):")
 print(f" 均值: {stats['mean']:6.3f}, 标准差: {stats['std']:6.3f}")
 print(f" 范围: [{stats['min']:6.3f}, {stats['max']:6.3f}]")

 return samples, trajectory

# 分析采样轨迹
final_samples, full_trajectory = analyze_sampling_trajectory(trainer)
```





练习 3.6.3：实现Precision和Recall指标
 实现改进的Precision和Recall指标，分别衡量生成质量和模式覆盖度。

def compute_precision_recall(real_features, gen_features, k=3):
 """计算改进的Precision和Recall

 基于k-最近邻的方法：
 - Precision: 生成样本中有多少落在真实数据的支撑集内
 - Recall: 真实数据的支撑集有多少被生成样本覆盖
 """
 from sklearn.neighbors import NearestNeighbors

 # 构建k-NN模型
 nbrs_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(real_features)
 nbrs_gen = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(gen_features)

 # 计算真实样本的k-NN距离
 distances_real, _ = nbrs_real.kneighbors(real_features)
 distances_real = distances_real[:, -1] # 第k个最近邻的距离

 # 计算生成样本的k-NN距离
 distances_gen, _ = nbrs_gen.kneighbors(gen_features)
 distances_gen = distances_gen[:, -1]

 # 计算Precision：生成样本到真实流形的距离
 distances_gen_to_real, _ = nbrs_real.kneighbors(gen_features, n_neighbors=1)
 distances_gen_to_real = distances_gen_to_real[:, 0]
 precision = np.mean(distances_gen_to_real





## 3.7 DDPM的局限性与改进方向

 虽然DDPM在生成质量上取得了重大突破，但它仍存在一些重要的局限性。理解这些局限性有助于我们理解后续的改进方法。


### 3.7.1 主要局限性



 DDPM的核心问题


 - **采样速度慢**


 需要1000步迭代才能生成一张图像
 - 相比GAN的单次前向传播，效率差距巨大
 - 限制了实时应用的可能性


 
 - **固定的噪声调度**


 线性β调度并非最优
 - 不同数据集可能需要不同的调度策略
 - 训练和采样必须使用相同的调度


 
 - **固定的后验方差**


 DDPM使用固定的后验方差 $\sigma_t^2 = \beta_t$
 - 这可能不是最优选择
 - 限制了模型的表达能力


 
 - **计算资源需求高**


 训练需要大量GPU时间
 - 推理时的内存占用较大
 - 难以在边缘设备上部署


 





### 3.7.2 性能分析




def analyze_ddpm_performance(trainer, num_samples=100):
 """分析DDPM的性能瓶颈"""
 import time

 results = {
 'sampling_times': [],
 'memory_usage': [],
 'step_times': []
 }

 # 测试不同步数的采样时间
 for num_steps in [10, 50, 100, 500, 1000]:
 # 修改采样步数
 original_steps = trainer.num_timesteps
 trainer.num_timesteps = num_steps

 # 计时
 start_time = time.time()
 samples = trainer.sample(num_samples, image_size=(1, 28, 28))
 end_time = time.time()

 sampling_time = end_time - start_time
 results['sampling_times'].append({
 'steps': num_steps,
 'total_time': sampling_time,
 'time_per_sample': sampling_time / num_samples,
 'time_per_step': sampling_time / (num_samples * num_steps)
 })

 trainer.num_timesteps = original_steps

 # 分析每步的时间分布
 with torch.profiler.profile(
 activities=[torch.profiler.ProfilerActivity.CPU,
 torch.profiler.ProfilerActivity.CUDA],
 record_shapes=True
 ) as prof:
 trainer.sample(1, image_size=(1, 28, 28))

 # 打印分析结果
 print("=== DDPM Performance Analysis ===")
 print(f"\nSampling Time vs Steps:")
 for result in results['sampling_times']:
 print(f"Steps: {result['steps']:4d} | "
 f"Total: {result['total_time']:6.2f}s | "
 f"Per Sample: {result['time_per_sample']:6.4f}s | "
 f"Per Step: {result['time_per_step']*1000:6.2f}ms")

 print(f"\nTop operations by time:")
 print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

 return results

# 分析性能结果
def analyze_performance_results(results):
 """分析性能测试结果"""
 steps = [r['steps'] for r in results['sampling_times']]
 times = [r['time_per_sample'] for r in results['sampling_times']]

 print("\n性能分析报告:")
 print("="*60)
 print("步数 | 每样本时间(s) | 相对1000步加速比 | 质量影响")
 print("-"*60)

 baseline_time = times[-1] # 1000步的时间
 for i, (step, time) in enumerate(zip(steps, times)):
 speedup = baseline_time / time
 quality_impact = "高" if step >= 500 else ("中" if step >= 100 else "低")
 print(f"{step:8d} | {time:13.4f} | {speedup:15.1f}x | {quality_impact}")

 print("\n关键发现:")
 print(f"- 从1000步减少到50步可获得 {baseline_time/times[1]:.1f}x 加速")
 print(f"- 每步平均耗时: {times[-1]/1000*1000:.2f}ms")
 print(f"- 主要瓶颈: U-Net前向传播")

 return results



### 3.7.3 改进方向概览



 主要改进方向
 
 
 
 问题
 改进方法
 关键思想
 相关章节
 
 
 
 
 采样速度慢
 DDIM
 确定性采样，跳步
 第8章
 
 
 采样速度慢
 DPM-Solver
 高阶ODE求解器
 第8章
 
 
 固定噪声调度
 Improved DDPM
 余弦调度，学习方差
 第8章
 
 
 理论框架
 Score-based Models
 分数匹配视角
 第4章
 
 
 连续时间
 SDE/ODE
 连续时间框架
 第5章
 
 
 计算效率
 Latent Diffusion
 潜在空间扩散
 第10章
 
 
 一步生成
 Consistency Models
 自一致性映射
 第14章
 
 
 



### 3.7.4 实验：不同改进的效果




```python
class ImprovedDDPMExperiments:
 """实验不同的DDPM改进方法"""

 @staticmethod
 def cosine_beta_schedule(num_timesteps, s=0.008):
 """余弦噪声调度（Improved DDPM）"""
 steps = num_timesteps + 1
 x = torch.linspace(0, num_timesteps, steps)
 alphas_bar = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
 alphas_bar = alphas_bar / alphas_bar[0]
 betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
 return torch.clip(betas, 0.0001, 0.9999)

 @staticmethod
 def learned_variance_output(model_output, num_channels):
 """学习方差的模型输出（Improved DDPM）"""
 # 模型输出两倍通道：前半部分是均值，后半部分是对数方差
 mean, log_variance = torch.split(model_output, num_channels, dim=1)

 # 参数化对数方差在[beta_t, beta_tilde_t]之间
 # log_variance = log(beta_t) + v * log(beta_tilde_t / beta_t)
 # 其中v是模型预测的插值参数
 return mean, log_variance

 @staticmethod
 def ddim_sampling_step(x_t, epsilon_pred, t, t_prev, alphas_bar, eta=0):
 """DDIM采样步骤（可调节随机性）"""
 alpha_bar_t = alphas_bar[t]
 alpha_bar_t_prev = alphas_bar[t_prev] if t_prev >= 0 else 1.0

 # 计算x_0的预测
 x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)

 # 计算方差
 sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
 torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

 # 预测x_{t-1}
 mean = torch.sqrt(alpha_bar_t_prev) * x_0_pred + \
 torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * epsilon_pred

 if t_prev > 0:
 noise = torch.randn_like(x_t)
 x_t_prev = mean + sigma_t * noise
 else:
 x_t_prev = mean

 return x_t_prev

 @staticmethod
 def compare_sampling_methods(model, device='cuda'):
 """比较不同采样方法的效果"""
 results = {}

 # 标准DDPM采样
 print("Testing standard DDPM sampling...")
 start_time = time.time()
 ddpm_samples = standard_ddpm_sample(model, num_samples=16, num_steps=1000)
 ddpm_time = time.time() - start_time
 results['ddpm'] = {'samples': ddpm_samples, 'time': ddpm_time}

 # DDIM采样（50步）
 print("Testing DDIM sampling (50 steps)...")
 start_time = time.time()
 ddim_samples = ddim_sample(model, num_samples=16, num_steps=50, eta=0)
 ddim_time = time.time() - start_time
 results['ddim'] = {'samples': ddim_samples, 'time': ddim_time}

 # 带随机性的DDIM采样
 print("Testing stochastic DDIM (eta=0.5)...")
 start_time = time.time()
 stochastic_samples = ddim_sample(model, num_samples=16, num_steps=50, eta=0.5)
 stochastic_time = time.time() - start_time
 results['stochastic'] = {'samples': stochastic_samples, 'time': stochastic_time}

 return results
```





练习 3.7：实现简化版DDIM
 基于本章学到的DDPM知识，实现一个简化版的DDIM采样器，支持可变步数采样。

@torch.no_grad()
def simplified_ddim_sample(model, shape, num_inference_steps=50,
 num_train_steps=1000, eta=0.0, device='cuda'):
 """简化版DDIM采样实现

 Args:
 model: 训练好的噪声预测模型
 shape: 生成图像的形状
 num_inference_steps: 推理步数（= 0 else 1.0

 # 预测x_0
 x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
 x_0_pred = torch.clamp(x_0_pred, -1, 1) # 数值稳定性

 # 计算方差
 sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
 torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev) if t_prev >= 0 else 0

 # 计算均值
 mean = torch.sqrt(alpha_bar_t_prev) * x_0_pred + \
 torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * epsilon_pred

 # 添加噪声
 if t_prev >= 0:
 noise = torch.randn_like(x_t) if eta > 0 else 0
 x_t = mean + sigma_t * noise
 else:
 x_t = mean

 return x_t

# 测试不同步数和eta值
test_configs = [
 {'steps': 10, 'etas': [0.0, 0.3, 0.7, 1.0]},
 {'steps': 25, 'etas': [0.0, 0.3, 0.7, 1.0]},
 {'steps': 50, 'etas': [0.0, 0.3, 0.7, 1.0]}
]

print("DDIM采样测试结果:")
print("="*60)
print("步数 | η值 | 采样时间(s) | 相对质量评估")
print("-"*60)

for config in test_configs:
 num_steps = config['steps']
 for eta in config['etas']:
 import time
 start = time.time()
 samples = simplified_ddim_sample(
 model, shape=(1, 1, 28, 28),
 num_inference_steps=num_steps,
 eta=eta
 )
 elapsed = time.time() - start

 # 简单的质量评估（基于样本统计）
 quality = "高" if samples.std() > 0.3 else ("中" if samples.std() > 0.2 else "低")

 print(f"{num_steps:4d} | {eta:4.1f} | {elapsed:11.4f} | {quality}")

print("\n关键观察:")
print("- η=0 (确定性采样) 速度最快，质量稳定")
print("- η=1 (完全随机，等同于DDPM) 质量最高但速度慢")
print("- 步数减少显著提升速度，但可能影响质量")






## 本章小结

 在本章中，我们深入学习了DDPM（去噪扩散概率模型）的核心原理和实现细节：



#### 主要收获



 - **理论基础**：理解了前向扩散过程、反向去噪过程和变分下界的推导
 - **实践实现**：构建了完整的DDPM系统，包括U-Net架构、训练循环和采样算法
 - **评估方法**：学习了FID、IS等生成模型评估指标的计算和使用
 - **局限认识**：了解了DDPM的主要问题，为学习后续改进方法打下基础




#### 关键要点



 - DDPM通过逐步添加噪声和学习逆过程来生成数据
 - 训练目标简化为预测每一步添加的噪声
 - 采样过程需要多步迭代，这是主要的效率瓶颈
 - 模型质量高但推理速度慢，这推动了后续的众多改进




#### 展望


在接下来的章节中，我们将探索：



 - **第4章**：从分数匹配的角度重新理解扩散模型
 - **第5章**：连续时间框架下的SDE/ODE表述
 - **第8章**：DDIM等快速采样方法的原理与实现




DDPM奠定了现代扩散模型的基础，理解它的原理对于掌握后续的高级技术至关重要。继续前进，让我们在下一章探索扩散模型的另一种视角——基于分数的生成模型！