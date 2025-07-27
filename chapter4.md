[← 上一章](chapter3.md)
 第4章 / 共14章
 [下一章 →](chapter5.md)



# 第4章：基于分数的生成模型



 基于分数的生成模型（Score-based Generative Models）提供了理解扩散模型的另一个重要视角。通过直接学习数据分布的分数函数（score function，即对数概率密度的梯度），我们可以构建强大的生成模型。本章将深入探讨分数匹配、Langevin动力学以及它们与扩散模型的深层联系。从NCSN到Score SDE，我们将看到分数模型如何与DDPM统一在同一框架下。



## 4.1 分数函数的直觉与重要性



### 4.1.1 什么是分数函数？



分数函数（score function）是概率论和统计学中的一个基本概念，它定义为对数概率密度函数关于数据的梯度：



 $$\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$



这个看似简单的定义蕴含着深刻的意义。让我们通过几个例子来理解它。



 例1：一维高斯分布

对于标准正态分布 $p(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$：



 $$\log p(x) = -\frac{x^2}{2} - \frac{1}{2}\log(2\pi)$$



分数函数为：


 $$\nabla_x \log p(x) = -x$$



注意这个结果的直观性：



 - 当 $x > 0$ 时，分数为负，指向原点（概率更高的方向）
 - 当 $x 



### 4.1.2 为什么分数函数重要？


 分数函数在机器学习和统计学中扮演着核心角色，其重要性体现在多个方面：



#### 1. 无需归一化常数



许多复杂的概率模型（如马尔可夫随机场、能量模型）的归一化常数难以计算：



 $$p(x) = \frac{1}{Z} \exp(-E(x)), \quad Z = \int \exp(-E(x)) dx$$



但分数函数可以直接计算，无需知道 $Z$：



 $$\nabla_x \log p(x) = -\nabla_x E(x)$$



 实例：Ising模型

在统计物理中的Ising模型中，系统能量为：


 $$E(x) = -J \sum_{\langle i,j \rangle} x_i x_j - h \sum_i x_i$$


其中 $x_i \in \{-1, +1\}$。配分函数 $Z$ 的计算是 #P-hard 问题，但能量的梯度（在连续松弛下）却很容易计算。




#### 2. 采样算法的基础



分数函数是许多高效采样算法的核心，特别是基于梯度的马尔可夫链蒙特卡洛（MCMC）方法：



 Langevin动力学

给定目标分布 $p(x)$，以下随机微分方程的平稳分布是 $p(x)$：


 $$dx = \nabla_x \log p(x) dt + \sqrt{2} dW_t$$


这表明只需要分数函数就可以从分布中采样！




#### 3. 与优化的深刻联系



分数函数连接了概率建模和优化理论：




 - **梯度上升**：沿着分数函数的方向移动，相当于在对数概率上做梯度上升
 - **模式寻找**：分数函数为零的点对应分布的局部极值（峰或谷）
 - **能量最小化**：最大化概率等价于最小化能量，分数函数给出下降方向




#### 4. 在生成模型中的应用



分数函数在现代生成模型中起着关键作用：




#### 扩散模型的两种视角



 - **去噪视角（DDPM）**：学习在不同噪声水平下去除噪声
 - **分数匹配视角**：学习不同时刻的分数函数 $\nabla_x \log p_t(x)$



这两种视角在数学上是等价的！去噪函数和分数函数之间存在简单的线性关系。




#### 5. 理论优势



 Fisher信息与分数函数

Fisher信息矩阵定义为：


 $$\mathcal{I} = \mathbb{E}_{x \sim p}[\nabla_x \log p(x) \nabla_x \log p(x)^T]$$


它刻画了分布的"信息几何"，在：



 - 参数估计的Cramér-Rao下界
 - 自然梯度下降
 - 信息几何和流形优化



中都起着核心作用。





# 演示分数函数的各种应用
import torch
import torch.nn as nn

class ScoreFunctionApplications:
 """分数函数的应用演示"""

 @staticmethod
 def langevin_sampling(score_fn, x_init, n_steps=1000, step_size=0.01):
 """使用Langevin动力学采样

 Args:
 score_fn: 分数函数 s(x) = ∇log p(x)
 x_init: 初始点
 n_steps: 采样步数
 step_size: 步长
 """
 x = x_init.clone()
 samples = [x.clone()]

 for _ in range(n_steps):
 noise = torch.randn_like(x)
 x = x + step_size * score_fn(x) + torch.sqrt(2 * step_size) * noise
 samples.append(x.clone())

 return torch.stack(samples)

 @staticmethod
 def mode_finding(score_fn, x_init, n_steps=100, step_size=0.1):
 """使用分数函数寻找模式（局部最大值）

 通过梯度上升找到 ∇log p(x) = 0 的点
 """
 x = x_init.clone()
 x.requires_grad_(True)

 for _ in range(n_steps):
 score = score_fn(x)
 x = x + step_size * score

 # 检查收敛
 if torch.norm(score) 



#### 6. 计算效率


 在高维空间中，分数函数的计算和学习往往比直接学习概率密度更高效：




 - **局部信息**：分数函数只需要局部梯度信息，不需要全局积分
 - **可并行化**：不同数据点的分数可以独立计算
 - **梯度友好**：神经网络天然适合学习梯度形式的函数




### 4.1.3 分数函数的几何意义



分数函数不仅是一个数学工具，它还有深刻的几何直觉。理解这些几何性质有助于我们更好地设计和分析基于分数的算法。



#### 1. 分数函数作为向量场



分数函数 $\nabla_x \log p(x)$ 在每个点 $x$ 定义了一个向量，这些向量共同构成了一个向量场。这个向量场有特殊的性质：




#### 向量场的直观理解



 - **方向**：指向概率密度增长最快的方向
 - **大小**：反映概率密度的变化率
 - **流线**：沿着向量场的积分曲线从低概率区域流向高概率区域





 二维高斯分布的分数场

对于二维高斯分布 $\mathcal{N}(\mu, \Sigma)$，分数函数为：


 $$\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)$$



几何特征：



 - 所有向量都指向均值 $\mu$
 - 距离均值越远，向量越长
 - 等概率线（椭圆）与分数向量正交
 - 沿着主轴方向，收敛速度由特征值决定





#### 2. 分数函数与水平集



概率密度的水平集（等概率面）与分数函数有密切关系：



 水平集的正交性

在任意点 $x$，分数函数 $\nabla_x \log p(x)$ 垂直于过该点的等概率面。



**证明草图**：设 $S_c = \{x : p(x) = c\}$ 是水平集。在 $S_c$ 上的任意切向量 $v$ 满足：


 $$v \cdot \nabla p(x) = 0$$


因此：


 $$v \cdot \nabla \log p(x) = v \cdot \frac{\nabla p(x)}{p(x)} = 0$$




#### 3. 分数函数的散度与拉普拉斯算子



分数函数的散度揭示了分布的局部几何：



 $$\nabla \cdot (\nabla \log p(x)) = \nabla^2 \log p(x) = \frac{\nabla^2 p(x)}{p(x)} - \frac{\|\nabla p(x)\|^2}{p(x)^2}$$




#### 散度的几何含义



 - **正散度**：该点是"源"，概率向外扩散（通常在分布的谷底）
 - **负散度**：该点是"汇"，概率向内聚集（通常在分布的峰值附近）
 - **零散度**：平衡点，可能是鞍点





#### 4. 动力系统视角



将分数函数视为动力系统 $\dot{x} = \nabla_x \log p(x)$ 的向量场，我们可以分析其稳定性：



 平衡点与稳定性


 - **平衡点**：$\nabla_x \log p(x^*) = 0$ 对应概率密度的临界点
 - **稳定性**：由Hessian矩阵 $H = \nabla^2 \log p(x^*)$ 决定


 $H \prec 0$（负定）：稳定吸引子，对应局部最大值
 - $H \succ 0$（正定）：不稳定排斥点，对应局部最小值
 - $H$ 不定：鞍点


 






# 可视化分数函数的几何性质
import torch
import numpy as np

def analyze_score_geometry(score_fn, x):
 """分析给定点的分数函数几何性质"""
 x = x.requires_grad_(True)

 # 计算分数
 score = score_fn(x)

 # 计算散度（Laplacian of log p）
 divergence = 0
 for i in range(len(x)):
 grad_i = torch.autograd.grad(score[i], x,
 retain_graph=True,
 create_graph=True)[0]
 divergence += grad_i[i]

 # 计算Hessian（用于稳定性分析）
 hessian = torch.zeros(len(x), len(x))
 for i in range(len(x)):
 for j in range(len(x)):
 if j >= i: # 利用对称性
 hess_ij = torch.autograd.grad(score[i], x,
 retain_graph=True,
 create_graph=True)[0][j]
 hessian[i, j] = hess_ij
 hessian[j, i] = hess_ij

 # 特征值分析
 eigenvalues, eigenvectors = torch.linalg.eigh(hessian)

 return {
 'score': score.detach(),
 'divergence': divergence.detach(),
 'hessian': hessian.detach(),
 'eigenvalues': eigenvalues.detach(),
 'eigenvectors': eigenvectors.detach()
 }

# 示例：分析二维高斯混合的几何性质
def gmm_score(x, means, weights):
 """高斯混合模型的分数函数"""
 scores = []
 probs = []

 for i, (mean, weight) in enumerate(zip(means, weights)):
 diff = x - mean
 prob = weight * torch.exp(-0.5 * torch.sum(diff**2))
 score = -diff
 scores.append(prob * score)
 probs.append(prob)

 total_prob = sum(probs)
 total_score = sum(scores) / (total_prob + 1e-8)

 return total_score

# 分析不同位置的几何性质
means = [torch.tensor([-2.0, 0.0]), torch.tensor([2.0, 0.0])]
weights = [0.5, 0.5]

# 在峰值处
x_peak = torch.tensor([-2.0, 0.0])
geometry_peak = analyze_score_geometry(
 lambda x: gmm_score(x, means, weights), x_peak
)
print(f"峰值处：散度 = {geometry_peak['divergence']:.3f}")
print(f"特征值：{geometry_peak['eigenvalues']}")

# 在鞍点处（两峰之间）
x_saddle = torch.tensor([0.0, 0.0])
geometry_saddle = analyze_score_geometry(
 lambda x: gmm_score(x, means, weights), x_saddle
)
print(f"\\n鞍点处：散度 = {geometry_saddle['divergence']:.3f}")
print(f"特征值：{geometry_saddle['eigenvalues']}")



#### 5. 分数函数的流形结构


 在高维空间中，数据往往集中在低维流形附近。分数函数能够捕捉这种流形结构：




#### 流形上的分数分解


在数据流形 $\mathcal{M}$ 附近，分数函数可以分解为：


 $$\nabla_x \log p(x) = \nabla_{\mathcal{M}} \log p(x) + \nabla_{\perp} \log p(x)$$



 - **切向分量** $\nabla_{\mathcal{M}} \log p(x)$：沿着流形移动，探索数据分布
 - **法向分量** $\nabla_{\perp} \log p(x)$：将点拉回流形，去除噪声



这解释了为什么分数模型能够有效地进行去噪和生成。




#### 6. 与最优传输的联系



分数函数还与最优传输理论有深刻联系：



 Fokker-Planck方程

考虑概率流 $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$，其中 $v_t$ 是速度场。如果选择：


 $$v_t(x) = \nabla_x \log p_t(x)$$


则得到的是梯度流，它在某种意义下是"最优"的传输方式。




## 4.2 分数匹配（Score Matching）



### 4.2.1 朴素分数匹配的困难



给定数据分布 $p_{data}(x)$ 的样本，我们希望学习一个模型 $s_\theta(x)$ 来逼近真实的分数函数 $\nabla_x \log p_{data}(x)$。最直接的想法是最小化：



 $$\mathcal{L}_{naive}(\theta) = \mathbb{E}_{x \sim p_{data}}\left[\|s_\theta(x) - \nabla_x \log p_{data}(x)\|^2\right]$$



但这个目标函数存在一个致命问题：**我们不知道真实的分数函数** $\nabla_x \log p_{data}(x)$！




#### 朴素方法的困境


计算 $\nabla_x \log p_{data}(x)$ 需要知道 $p_{data}(x)$，但：



 - 如果我们知道 $p_{data}(x)$，就不需要学习了
 - 即使用核密度估计等方法估计 $p_{data}(x)$，在高维空间中也会失效
 - 数值微分在高维空间中不稳定且计算昂贵



这似乎是一个无解的循环依赖！




#### Hyvärinen的突破性发现



2005年，Aapo Hyvärinen提出了一个巧妙的解决方案。他发现可以将上述损失函数改写为不依赖于真实分数的形式：



 分数匹配定理（Hyvärinen, 2005）

在适当的正则性条件下（$p(x)$ 在边界处趋于零），以下等式成立：


 $$\mathbb{E}_{x \sim p_{data}}\left[\|s_\theta(x) - \nabla_x \log p_{data}(x)\|^2\right] = \mathbb{E}_{x \sim p_{data}}\left[\text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right] + C$$


其中 $C$ 是不依赖于 $\theta$ 的常数，$\text{tr}(\nabla_x s_\theta(x))$ 是雅可比矩阵的迹。




#### 证明的关键思想



证明使用了分部积分的技巧。让我们看一个简化的一维情况：



 一维情况的推导

考虑期望：


 $$\mathbb{E}_{x \sim p}[(s_\theta(x) - \nabla_x \log p(x))^2]$$



展开后：


 $$= \mathbb{E}_{x \sim p}[s_\theta(x)^2] - 2\mathbb{E}_{x \sim p}[s_\theta(x) \nabla_x \log p(x)] + \mathbb{E}_{x \sim p}[(\nabla_x \log p(x))^2]$$



关键是处理中间项。注意到 $\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$，所以：


 $$\mathbb{E}_{x \sim p}[s_\theta(x) \nabla_x \log p(x)] = \int s_\theta(x) \frac{\nabla_x p(x)}{p(x)} p(x) dx = \int s_\theta(x) \nabla_x p(x) dx$$



使用分部积分（假设边界项为零）：


 $$\int s_\theta(x) \nabla_x p(x) dx = -\int \nabla_x s_\theta(x) p(x) dx = -\mathbb{E}_{x \sim p}[\nabla_x s_\theta(x)]$$




#### 实际的困难



虽然Hyvärinen的分数匹配理论上解决了问题，但在实践中仍面临挑战：




#### 计算挑战



 - **雅可比矩阵的迹**：计算 $\text{tr}(\nabla_x s_\theta(x))$ 需要 $d$ 次反向传播（$d$ 是数据维度）


 对于图像（$d \sim 10^6$），计算成本过高
 - 虽然有Hutchinson迹估计等技巧，但会引入额外的方差


 

 - **数值稳定性**：


 在数据分布的低密度区域，分数可能非常大
 - 训练不稳定，容易发散


 

 - **边界条件**：


 理论要求 $p(x) \to 0$ 当 $\|x\| \to \infty$
 - 实际数据可能不满足这个条件


 






# 朴素分数匹配的实现（仅用于说明，实践中很少使用）
import torch
import torch.nn as nn

def compute_score_matching_loss(model, x):
 """计算分数匹配损失（朴素版本）

 Args:
 model: 神经网络，输出分数估计 s_θ(x)
 x: 批量数据点 [batch_size, dim]

 Returns:
 loss: 分数匹配损失
 """
 x = x.requires_grad_(True)
 score = model(x)

 # 计算雅可比矩阵的迹
 # 注意：这需要 dim 次反向传播！
 trace_jacobian = 0
 for i in range(x.shape[1]):
 # 计算 ∂s_i/∂x_i
 grad_i = torch.autograd.grad(
 score[:, i].sum(), x,
 create_graph=True, retain_graph=True
 )[0][:, i]
 trace_jacobian = trace_jacobian + grad_i

 # 分数匹配损失
 loss = 0.5 * (score ** 2).sum(dim=1) + trace_jacobian

 return loss.mean()

# 更高效的实现：使用Hutchinson迹估计
def compute_score_matching_loss_hutchinson(model, x, n_hutchinson=1):
 """使用Hutchinson迹估计的分数匹配损失

 Args:
 model: 神经网络
 x: 数据点
 n_hutchinson: Hutchinson估计的采样数
 """
 x = x.requires_grad_(True)
 score = model(x)

 # Hutchinson迹估计
 trace_jacobian = 0
 for _ in range(n_hutchinson):
 # 随机向量 v ~ N(0, I)
 v = torch.randn_like(x)

 # 计算 v^T ∇_x s(x) v
 grad_v = torch.autograd.grad(
 (score * v).sum(), x,
 create_graph=True, retain_graph=True
 )[0]
 trace_jacobian = trace_jacobian + (grad_v * v).sum(dim=1)

 trace_jacobian = trace_jacobian / n_hutchinson

 # 损失
 loss = 0.5 * (score ** 2).sum(dim=1) + trace_jacobian

 return loss.mean()

# 演示为什么朴素方法困难
print("朴素分数匹配的计算复杂度：")
print(f"图像 (1024×1024×3): 需要 {1024*1024*3:,} 次反向传播！")
print(f"即使是 MNIST (28×28): 也需要 {28*28} 次反向传播")
print("\nHutchinson估计引入方差，需要权衡：")
print("- 更多采样 → 更准确但更慢")
print("- 更少采样 → 更快但方差大")



#### 为什么需要新方法？


 朴素分数匹配的这些困难促使研究者寻找更实用的方法：




 - **去噪分数匹配**（Denoising Score Matching）：通过添加噪声避免计算雅可比矩阵
 - **切片分数匹配**（Sliced Score Matching）：将高维问题投影到一维
 - **有限差分分数匹配**（Finite Difference Score Matching）：使用数值微分近似




接下来我们将详细介绍这些更实用的方法。



### 4.2.2 去噪分数匹配（Denoising Score Matching）



去噪分数匹配（DSM）是Vincent (2011)提出的一个优雅的解决方案，它巧妙地避开了计算雅可比矩阵的问题。核心思想是：与其直接学习数据分布的分数，不如学习加噪数据分布的分数。



#### 核心思想



给定干净数据 $x \sim p_{data}(x)$ 和噪声分布 $p_\sigma(\tilde{x}|x)$（通常是高斯噪声），定义加噪数据分布：



 $$p_\sigma(\tilde{x}) = \int p_{data}(x) p_\sigma(\tilde{x}|x) dx$$



去噪分数匹配的关键洞察是：**加噪分布的分数函数可以用条件期望表示**。



 去噪分数匹配定理

对于高斯噪声 $p_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$，加噪分布的分数函数为：


 $$\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = \mathbb{E}_{x \sim p(x|\tilde{x})}\left[\frac{x - \tilde{x}}{\sigma^2}\right]$$


这意味着分数函数指向"去噪"的方向！




#### 为什么这解决了问题？



去噪分数匹配的损失函数为：



 $$\mathcal{L}_{DSM}(\theta) = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)}\left[\left\|s_\theta(\tilde{x}, \sigma) - \frac{x - \tilde{x}}{\sigma^2}\right\|^2\right]$$




#### DSM的优势



 - **无需计算雅可比矩阵**：目标函数中的真实分数 $\frac{x - \tilde{x}}{\sigma^2}$ 是已知的！
 - **数值稳定**：加噪使得分布更平滑，分数函数更稳定
 - **与去噪的联系**：学习分数等价于学习去噪，这有直观的解释
 - **计算高效**：只需要前向传播，没有额外的计算开销





#### 多尺度去噪分数匹配



单一噪声水平的DSM仍有局限：在低噪声时难以覆盖整个数据空间，在高噪声时丢失细节。Song & Ermon (2019)提出使用多个噪声水平：



 噪声调度的设计

选择一系列递增的噪声水平 $\{\sigma_i\}_{i=1}^L$，通常采用几何级数：


 $$\sigma_i = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{i-1}{L-1}}$$



损失函数变为：


 $$\mathcal{L}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{x, \tilde{x}}\left[\left\|s_\theta(\tilde{x}, \sigma_i) - \frac{x - \tilde{x}}{\sigma_i^2}\right\|^2\right]$$



其中 $\lambda(\sigma_i)$ 是权重函数，常见选择包括：



 - 均匀权重：$\lambda(\sigma) = 1$
 - 与噪声成比例：$\lambda(\sigma) = \sigma^2$
 - 平衡权重：$\lambda(\sigma) = \sigma$






# 去噪分数匹配的实现
import torch
import torch.nn as nn
import numpy as np

class DenoisingScoreMatching:
 """去噪分数匹配训练框架"""

 def __init__(self, sigma_min=0.01, sigma_max=50, num_scales=10):
 """
 Args:
 sigma_min: 最小噪声标准差
 sigma_max: 最大噪声标准差
 num_scales: 噪声尺度数量
 """
 self.sigmas = torch.exp(
 torch.linspace(
 np.log(sigma_min),
 np.log(sigma_max),
 num_scales
 )
 )

 def get_noisy_data(self, x, sigma_idx=None):
 """添加高斯噪声

 Args:
 x: 干净数据 [batch_size, ...]
 sigma_idx: 噪声级别索引（None表示随机选择）

 Returns:
 x_noisy: 加噪数据
 sigma: 使用的噪声标准差
 target_score: 真实分数
 """
 batch_size = x.shape[0]

 # 选择噪声级别
 if sigma_idx is None:
 sigma_idx = torch.randint(0, len(self.sigmas), (batch_size,))

 sigma = self.sigmas[sigma_idx].view(batch_size, *([1] * (x.ndim - 1)))
 sigma = sigma.to(x.device)

 # 添加噪声
 noise = torch.randn_like(x)
 x_noisy = x + sigma * noise

 # 真实分数：(x - x_noisy) / sigma^2
 target_score = -(noise / sigma) # 注意这里用noise/sigma而不是(x-x_noisy)/sigma^2

 return x_noisy, sigma.squeeze(), target_score

 def loss_fn(self, model, x):
 """计算去噪分数匹配损失

 Args:
 model: 分数模型 s_θ(x, σ)
 x: 批量干净数据

 Returns:
 loss: DSM损失
 info: 额外信息用于日志
 """
 # 获取加噪数据和目标
 x_noisy, sigma, target_score = self.get_noisy_data(x)

 # 模型预测
 pred_score = model(x_noisy, sigma)

 # 计算损失（可以加权）
 loss = 0.5 * ((pred_score - target_score) ** 2).sum(dim=tuple(range(1, x.ndim)))

 # 按噪声级别加权
 loss_weights = sigma # 或 sigma**2，或 1
 weighted_loss = (loss * loss_weights).mean()

 info = {
 'loss': weighted_loss.item(),
 'mean_score_norm': pred_score.norm(dim=-1).mean().item(),
 'mean_sigma': sigma.mean().item()
 }

 return weighted_loss, info

# 简单的分数网络示例
class ScoreNet(nn.Module):
 """条件分数网络 s_θ(x, σ)"""

 def __init__(self, data_dim, hidden_dim=128, embed_dim=128):
 super().__init__()

 # 时间/噪声嵌入
 self.embed = nn.Sequential(
 nn.Linear(1, embed_dim),
 nn.SiLU(),
 nn.Linear(embed_dim, embed_dim)
 )

 # 主网络
 self.net = nn.Sequential(
 nn.Linear(data_dim + embed_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, data_dim)
 )

 def forward(self, x, sigma):
 """
 Args:
 x: 输入数据 [batch_size, data_dim]
 sigma: 噪声标准差 [batch_size]
 """
 # 嵌入噪声级别
 sigma_embed = self.embed(sigma.log().unsqueeze(-1))

 # 拼接并通过网络
 h = torch.cat([x, sigma_embed], dim=-1)
 score = self.net(h)

 # 按照理论，分数应该与1/sigma成比例
 # 这里可以选择是否要显式建模这个关系
 return score / sigma.unsqueeze(-1)

# 使用示例
dsm = DenoisingScoreMatching()
model = ScoreNet(data_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 生成一些2D数据用于演示
x_data = torch.randn(100, 2) * 0.5 + torch.tensor([[2.0, 2.0]])

# 训练步骤
model.train()
loss, info = dsm.loss_fn(model, x_data)
loss.backward()
optimizer.step()

print(f"DSM Loss: {info['loss']:.4f}")
print(f"Mean Score Norm: {info['mean_score_norm']:.4f}")
print(f"Mean Sigma: {info['mean_sigma']:.4f}")



#### 理论保证


 去噪分数匹配不仅实用，还有坚实的理论基础：



 DSM的一致性

在适当的正则性条件下，最小化DSM损失等价于最小化以下KL散度：


 $$\min_\theta \text{KL}(p_\sigma(\tilde{x}) \| p_\theta(\tilde{x}))$$


其中 $p_\theta$ 是由分数函数 $s_\theta$ 诱导的分布。




#### 与DDPM的联系



去噪分数匹配与DDPM有深刻的联系：




#### 统一视角


DDPM的去噪目标：


 $$\mathcal{L}_{DDPM} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$



通过变量替换 $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$，这等价于：


 $$\mathcal{L}_{DSM} = \mathbb{E}\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t)\|^2\right]$$



因此，**DDPM本质上就是在做去噪分数匹配**！




### 4.2.3 切片分数匹配（Sliced Score Matching）



切片分数匹配（Sliced Score Matching, SSM）是Song et al. (2020)提出的另一个避免计算雅可比矩阵的方法。核心思想是将高维分数匹配问题投影到随机选择的一维方向上。



#### 基本思想



对于任意单位向量 $v \in \mathbb{R}^d$（$\|v\| = 1$），定义投影分数：



 $$s_v(x) = v^T \nabla_x \log p(x) = v^T s(x)$$



这是分数函数在方向 $v$ 上的投影。关键观察是：如果我们知道所有方向上的投影，就能重构完整的分数函数。



 切片分数匹配定理

分数匹配目标可以重写为：


 $$\mathbb{E}_{x \sim p_{data}}\left[\|s_\theta(x) - \nabla_x \log p(x)\|^2\right] = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{v \sim \mathcal{N}(0,I)}\left[\left(v^T s_\theta(x) - v^T \nabla_x \log p(x)\right)^2\right]$$



更重要的是，使用分部积分后：


 $$= \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{v \sim \mathcal{N}(0,I)}\left[2 v^T \nabla_x (v^T s_\theta(x)) + (v^T s_\theta(x))^2\right] + C$$



其中 $v^T \nabla_x (v^T s_\theta(x))$ 只需要计算一个方向导数，而不是完整的雅可比矩阵！




#### 计算优势




#### 为什么SSM更高效？



 - **朴素分数匹配**：需要计算 $d \times d$ 的雅可比矩阵的迹，需要 $O(d)$ 次反向传播
 - **切片分数匹配**：只需要计算方向导数 $v^T \nabla_x (v^T s_\theta(x))$，只需要 $O(1)$ 次反向传播
 - **随机性**：通过对多个随机方向 $v$ 求平均来降低方差





#### 实现细节




# 切片分数匹配的实现
import torch
import torch.nn as nn

def sliced_score_matching_loss(model, x, n_slices=1):
 """计算切片分数匹配损失

 Args:
 model: 分数模型 s_θ(x)
 x: 数据点 [batch_size, dim]
 n_slices: 每个样本使用的随机投影数

 Returns:
 loss: SSM损失
 """
 batch_size = x.shape[0]
 dim = x.shape[1]

 # 需要梯度来计算二阶导数
 x = x.requires_grad_(True)

 # 计算分数
 score = model(x) # [batch_size, dim]

 # 初始化损失
 loss = 0

 for _ in range(n_slices):
 # 采样随机方向
 v = torch.randn(batch_size, dim, device=x.device)
 v = v / v.norm(dim=1, keepdim=True) # 归一化

 # 计算投影分数 v^T s(x)
 s_v = (score * v).sum(dim=1) # [batch_size]

 # 计算方向导数 v^T ∇_x (v^T s(x))
 grad_s_v = torch.autograd.grad(
 s_v.sum(), x,
 create_graph=True,
 retain_graph=True
 )[0] # [batch_size, dim]

 # v^T ∇_x (v^T s(x))
 tr_grad_s_v = (grad_s_v * v).sum(dim=1) # [batch_size]

 # 累加损失：2 * tr_grad_s_v + s_v^2
 loss = loss + (2 * tr_grad_s_v + s_v ** 2)

 # 平均over slices和batch
 return loss.mean() / n_slices

# 改进版：使用Hutchinson估计进一步优化
def sliced_score_matching_loss_v2(model, x, n_slices=1):
 """SSM的另一种实现，计算上等价但可能数值更稳定"""

 x = x.requires_grad_(True)
 score = model(x)

 loss = 0

 for _ in range(n_slices):
 # 随机投影向量
 v = torch.randn_like(x)
 v = v / v.norm(dim=1, keepdim=True)

 # 计算 v^T J_s v，其中 J_s 是分数的雅可比矩阵
 # 这等价于计算 ∇_x (v^T s(x)) · v
 v_dot_score = (score * v).sum(dim=1)

 # 直接计算二阶导数
 grad2 = torch.autograd.grad(
 outputs=v_dot_score,
 inputs=x,
 grad_outputs=torch.ones_like(v_dot_score),
 create_graph=True,
 retain_graph=True
 )[0]

 # 方向二阶导数
 tr_hessian = (grad2 * v).sum(dim=1)

 # SSM损失
 loss = loss + (v_dot_score ** 2 + 2 * tr_hessian)

 return loss.mean() / n_slices

# 结合去噪的切片分数匹配
class SlicedDenoisingScoreMatching:
 """结合切片技术和去噪技术"""

 def __init__(self, sigma_min=0.01, sigma_max=50, num_scales=10):
 self.sigmas = torch.exp(
 torch.linspace(np.log(sigma_min), np.log(sigma_max), num_scales)
 )

 def loss_fn(self, model, x, n_slices=1):
 """切片去噪分数匹配损失

 这结合了SSM的计算效率和DSM的稳定性
 """
 batch_size = x.shape[0]

 # 随机选择噪声级别
 sigma_idx = torch.randint(0, len(self.sigmas), (batch_size,))
 sigma = self.sigmas[sigma_idx].to(x.device)

 # 添加噪声
 noise = torch.randn_like(x)
 x_noisy = x + sigma.view(-1, 1) * noise

 # 模型预测
 score_pred = model(x_noisy, sigma)

 # 真实分数
 score_true = -noise / sigma.view(-1, 1)

 # 切片损失：不需要计算雅可比矩阵！
 loss = 0
 for _ in range(n_slices):
 v = torch.randn_like(x)
 v = v / v.norm(dim=1, keepdim=True)

 # 投影误差
 proj_pred = (score_pred * v).sum(dim=1)
 proj_true = (score_true * v).sum(dim=1)

 loss = loss + (proj_pred - proj_true) ** 2

 # 加权
 weighted_loss = (loss * sigma).mean() / n_slices

 return weighted_loss

# 演示使用
print("切片分数匹配的计算效率比较：")
print("设数据维度 d = 1000")
print("- 朴素分数匹配：需要 1000 次反向传播")
print("- 切片分数匹配（10个切片）：只需要 10 次反向传播")
print("- 加速比：100x！")
print("\n注意：SSM引入了额外的方差，需要更多的训练步数")



#### 理论分析


 切片分数匹配的收敛性和效率取决于几个因素：



 方差分析

设 $m$ 是每个样本使用的切片数，则SSM估计器的方差为：


 $$\text{Var}[\mathcal{L}_{SSM}] \approx \frac{1}{m} \text{Var}[\mathcal{L}_{SM}]$$


这意味着：



 - 使用更多切片可以降低方差
 - 但计算成本线性增长
 - 实践中，$m = 1$ 到 $m = 10$ 通常就足够了





#### 实用建议




#### 什么时候使用SSM？



 - **高维数据**：当 $d > 100$ 时，SSM的计算优势明显
 - **实时应用**：需要快速训练时
 - **与DSM结合**：可以同时使用去噪和切片技术
 - **注意事项**：


 SSM可能需要更多的训练迭代
 - 对于低维问题，朴素方法可能更稳定
 - 批量大小要足够大以降低方差


 





#### 与其他方法的比较



 分数匹配方法对比
 
 
 方法
 计算复杂度
 稳定性
 适用场景
 
 
 朴素SM
 $O(d)$ 反向传播
 高
 低维、小规模
 
 
 去噪SM
 $O(1)$ 反向传播
 很高
 通用，特别是生成模型
 
 
 切片SM
 $O(m)$ 反向传播
 中等
 高维、大规模
 
 
 切片去噪SM
 $O(1)$ 反向传播
 高
 高维生成模型
 
 



## 4.3 噪声条件分数网络（NCSN）



### 4.3.1 多尺度噪声的动机



噪声条件分数网络（Noise Conditional Score Networks, NCSN）是Song & Ermon (2019)提出的一个突破性方法。它解决了单一噪声水平分数匹配的根本限制。



#### 单一噪声水平的问题



考虑在单一噪声水平 $\sigma$ 下学习分数函数。我们面临一个困境：




#### 噪声水平的两难选择



 - **低噪声（$\sigma$ 小）**：


 ✓ 保留数据细节
 - ✗ 只覆盖数据流形附近的区域
 - ✗ 模式之间没有连接，采样困难


 
 - **高噪声（$\sigma$ 大）**：


 ✓ 覆盖整个空间
 - ✓ 模式之间有连接
 - ✗ 丢失数据细节
 - ✗ 分数函数过于平滑


 





 具体例子：二维混合高斯

考虑数据分布是两个分离的高斯分布：


 $$p_{data}(x) = \frac{1}{2}\mathcal{N}(x; [-5, 0], I) + \frac{1}{2}\mathcal{N}(x; [5, 0], I)$$



不同噪声水平下的问题：



 - **$\sigma = 0.1$**：两个模式完全分离，Langevin采样会困在一个模式中
 - **$\sigma = 5.0$**：两个模式混合成一个大的高斯，丢失了双峰结构
 - **$\sigma = 1.0$**：折中方案，但仍不理想





#### 多尺度方法的洞察



NCSN的核心思想是使用一系列递增的噪声水平 $\{\sigma_i\}_{i=1}^L$，并学习所有噪声水平下的分数函数：



 $$s_\theta(x, \sigma_i) \approx \nabla_x \log p_{\sigma_i}(x)$$



其中 $p_{\sigma_i}(x) = \int p_{data}(x') \mathcal{N}(x; x', \sigma_i^2 I) dx'$。



 多尺度的优势


 - **全局到局部的探索**：高噪声水平提供全局连通性，低噪声水平恢复局部细节
 - **平滑的过渡**：相邻噪声水平之间的分布相似，便于学习和采样
 - **稳定的训练**：每个噪声水平的分数函数都相对平滑
 - **退火采样**：可以从高噪声逐步退火到低噪声，类似模拟退火





#### 噪声调度的设计



选择合适的噪声水平序列至关重要：




# 不同的噪声调度策略
import numpy as np
import torch

class NoiseSchedule:
 """噪声调度的各种策略"""

 @staticmethod
 def geometric(sigma_min=0.01, sigma_max=50.0, num_scales=10):
 """几何级数：最常用，确保比例恒定

 σ_i = σ_min * (σ_max/σ_min)^((i-1)/(L-1))
 """
 return torch.exp(
 torch.linspace(
 np.log(sigma_min),
 np.log(sigma_max),
 num_scales
 )
 )

 @staticmethod
 def linear(sigma_min=0.01, sigma_max=50.0, num_scales=10):
 """线性间隔：简单但通常不是最优"""
 return torch.linspace(sigma_min, sigma_max, num_scales)

 @staticmethod
 def quadratic(sigma_min=0.01, sigma_max=50.0, num_scales=10):
 """二次间隔：在低噪声区域更密集"""
 t = torch.linspace(0, 1, num_scales)
 return sigma_min + (sigma_max - sigma_min) * t**2

 @staticmethod
 def cosine(sigma_min=0.01, sigma_max=50.0, num_scales=10):
 """余弦调度：平滑过渡"""
 t = torch.linspace(0, 1, num_scales)
 return sigma_min + (sigma_max - sigma_min) * (1 - torch.cos(t * np.pi)) / 2

 @staticmethod
 def adaptive(data_samples, num_scales=10, percentiles=[1, 99]):
 """自适应调度：基于数据分布

 根据数据点之间的距离分布来选择噪声水平
 """
 # 计算数据点之间的成对距离
 n = min(1000, len(data_samples)) # 采样以提高效率
 idx = torch.randperm(len(data_samples))[:n]
 samples = data_samples[idx]

 # 计算成对距离
 dists = torch.cdist(samples, samples)
 dists = dists[torch.triu(torch.ones_like(dists), diagonal=1).bool()]

 # 基于距离分布选择噪声水平
 sigma_min = torch.quantile(dists, percentiles[0]/100)
 sigma_max = torch.quantile(dists, percentiles[1]/100)

 return NoiseSchedule.geometric(sigma_min, sigma_max, num_scales)

# 分析不同调度的特性
def analyze_schedule(schedule_name, sigmas):
 """分析噪声调度的特性"""
 ratios = sigmas[1:] / sigmas[:-1]

 print(f"\n{schedule_name} Schedule:")
 print(f" Range: [{sigmas[0]:.3f}, {sigmas[-1]:.3f}]")
 print(f" Ratios: min={ratios.min():.3f}, max={ratios.max():.3f}, mean={ratios.mean():.3f}")
 print(f" First 3: {sigmas[:3].numpy()}")
 print(f" Last 3: {sigmas[-3:].numpy()}")

# 比较不同调度
num_scales = 10
schedules = {
 "Geometric": NoiseSchedule.geometric(num_scales=num_scales),
 "Linear": NoiseSchedule.linear(num_scales=num_scales),
 "Quadratic": NoiseSchedule.quadratic(num_scales=num_scales),
 "Cosine": NoiseSchedule.cosine(num_scales=num_scales)
}

for name, sigmas in schedules.items():
 analyze_schedule(name, sigmas)



#### 理论依据：退火重要性采样


 多尺度方法的理论基础来自于退火重要性采样（Annealed Importance Sampling, AIS）：



 退火采样的收敛性

设 $\{p_i\}_{i=0}^L$ 是一系列分布，满足：



 - $p_0$ 容易采样（如标准高斯）
 - $p_L = p_{data}$ 是目标分布
 - 相邻分布"足够接近"：$D_{KL}(p_i \| p_{i+1}) 简单直接
 - 可能对某些噪声水平欠拟合或过拟合


 
 - **与噪声成比例**：$\lambda(\sigma) = \sigma / \sum_j \sigma_j$


 高噪声水平获得更多权重
 - 有助于全局结构的学习


 
 - **与噪声平方成比例**：$\lambda(\sigma) = \sigma^2 / \sum_j \sigma_j^2$


 补偿不同噪声水平下的方差差异
 - Song & Ermon (2019) 的推荐选择


 





#### 实践考虑




#### 设计多尺度系统的要点



 - **噪声范围**：


 $\sigma_{max}$ 应该足够大，使得 $p_{\sigma_{max}} \approx \mathcal{N}(0, \sigma_{max}^2 I)$
 - $\sigma_{min}$ 应该足够小以保留数据细节，但不能太小导致数值不稳定


 
 - **尺度数量**：


 太少：相邻尺度差距大，退火效果差
 - 太多：计算成本高，可能过拟合
 - 典型选择：10-100个尺度


 
 - **条件架构**：


 网络必须能够根据噪声水平调整行为
 - 常用方法：FiLM conditioning、时间嵌入等


 





### 4.3.2 NCSN架构设计



设计一个有效的噪声条件分数网络需要考虑多个方面：网络如何处理不同尺度的输入、如何编码噪声水平信息，以及如何确保输出的正确尺度。



#### 核心设计原则




#### NCSN架构的关键要求



 - **噪声条件化**：网络必须根据噪声水平 $\sigma$ 调整其行为
 - **尺度等变性**：分数函数具有特定的尺度关系：$\nabla_x \log p_\sigma(x) \propto 1/\sigma^2$
 - **多尺度特征**：需要捕捉从粗到细的不同尺度特征
 - **计算效率**：单个网络处理所有噪声水平





#### 条件化机制



有几种将噪声水平 $\sigma$ 融入网络的方法：



 1. 拼接方法（Concatenation）

最简单的方法是将噪声水平直接拼接到输入：


 $$s_\theta(x, \sigma) = f_\theta([x, \sigma \cdot \mathbf{1}])$$


其中 $\mathbf{1}$ 是与 $x$ 同维度的全1向量。




 - ✓ 简单直接
 - ✗ 可能不够灵活
 - ✗ 噪声信息可能在深层丢失





 2. FiLM条件化（Feature-wise Linear Modulation）

通过学习的缩放和偏移来调制特征：


 $$h' = \gamma(\sigma) \odot h + \beta(\sigma)$$


其中 $\gamma(\sigma)$ 和 $\beta(\sigma)$ 是从噪声水平学习的调制参数。




 - ✓ 更灵活的条件化
 - ✓ 可以应用于多个层
 - ✓ 保持特征的语义





 3. 位置编码方法（Positional Encoding）

借鉴Transformer的思想，使用正弦编码：


 $$\text{embed}(\sigma) = [\sin(2^0 \pi \sigma), \cos(2^0 \pi \sigma), ..., \sin(2^{L-1} \pi \sigma), \cos(2^{L-1} \pi \sigma)]$$




 - ✓ 能够表示连续的噪声水平
 - ✓ 具有良好的插值性质
 - ✓ 在Transformer架构中特别有效






# NCSN架构实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalEmbedding(nn.Module):
 """正弦位置编码用于噪声水平"""

 def __init__(self, dim, scale=1.0):
 super().__init__()
 self.dim = dim
 self.scale = scale

 def forward(self, x):
 """
 Args:
 x: 噪声水平 [batch_size] 或 [batch_size, 1]
 Returns:
 嵌入 [batch_size, dim]
 """
 if x.dim() == 1:
 x = x.unsqueeze(-1)

 half_dim = self.dim // 2
 emb = np.log(10000) / half_dim
 emb = torch.exp(-emb * torch.arange(half_dim, device=x.device))
 emb = self.scale * x * emb
 emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

 return emb

class ConditionalInstanceNorm2d(nn.Module):
 """条件实例归一化，用于FiLM"""

 def __init__(self, num_features, num_classes):
 super().__init__()
 self.num_features = num_features
 self.norm = nn.InstanceNorm2d(num_features, affine=False)

 # 学习的调制参数
 self.embed = nn.Linear(num_classes, num_features * 2)
 self.embed.weight.data[:, :num_features].normal_(1, 0.02) # gamma
 self.embed.weight.data[:, num_features:].zero_() # beta

 def forward(self, x, y):
 """
 Args:
 x: 特征图 [batch_size, num_features, height, width]
 y: 条件嵌入 [batch_size, num_classes]
 """
 out = self.norm(x)
 gamma, beta = self.embed(y).chunk(2, dim=1)
 gamma = gamma.view(-1, self.num_features, 1, 1)
 beta = beta.view(-1, self.num_features, 1, 1)

 return gamma * out + beta

class ResBlock(nn.Module):
 """带条件化的残差块"""

 def __init__(self, in_channels, out_channels, embed_dim, dropout=0.1):
 super().__init__()
 self.in_channels = in_channels
 self.out_channels = out_channels

 self.norm1 = ConditionalInstanceNorm2d(in_channels, embed_dim)
 self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

 self.norm2 = ConditionalInstanceNorm2d(out_channels, embed_dim)
 self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

 if in_channels != out_channels:
 self.skip = nn.Conv2d(in_channels, out_channels, 1)
 else:
 self.skip = nn.Identity()

 self.dropout = nn.Dropout(dropout)

 def forward(self, x, embed):
 """
 Args:
 x: 输入特征
 embed: 噪声嵌入
 """
 h = self.norm1(x, embed)
 h = F.silu(h)
 h = self.conv1(h)

 h = self.norm2(h, embed)
 h = F.silu(h)
 h = self.dropout(h)
 h = self.conv2(h)

 return h + self.skip(x)

class NCSN(nn.Module):
 """噪声条件分数网络"""

 def __init__(
 self,
 channels=3,
 embed_dim=128,
 channels_mult=[1, 2, 2, 2],
 num_res_blocks=2,
 dropout=0.1,
 use_scale_shift_norm=True
 ):
 super().__init__()

 # 噪声嵌入
 self.embed = nn.Sequential(
 SinusoidalEmbedding(embed_dim),
 nn.Linear(embed_dim, embed_dim * 4),
 nn.SiLU(),
 nn.Linear(embed_dim * 4, embed_dim * 4),
 nn.SiLU(),
 nn.Linear(embed_dim * 4, embed_dim)
 )

 # U-Net编码器
 self.conv_in = nn.Conv2d(channels, embed_dim, 3, padding=1)

 down_blocks = []
 ch = embed_dim
 for level, mult in enumerate(channels_mult):
 for _ in range(num_res_blocks):
 down_blocks.append(ResBlock(ch, mult * embed_dim, embed_dim, dropout))
 ch = mult * embed_dim
 if level != len(channels_mult) - 1:
 down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

 self.down = nn.ModuleList(down_blocks)

 # 中间块
 self.mid = nn.ModuleList([
 ResBlock(ch, ch, embed_dim, dropout),
 ResBlock(ch, ch, embed_dim, dropout)
 ])

 # U-Net解码器
 up_blocks = []
 for level, mult in list(enumerate(channels_mult))[::-1]:
 for _ in range(num_res_blocks + 1):
 up_blocks.append(ResBlock(ch, mult * embed_dim, embed_dim, dropout))
 ch = mult * embed_dim
 if level != 0:
 up_blocks.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

 self.up = nn.ModuleList(up_blocks)

 # 输出层
 self.norm_out = nn.GroupNorm(8, ch)
 self.conv_out = nn.Conv2d(ch, channels, 3, padding=1)

 def forward(self, x, sigma):
 """
 Args:
 x: 输入图像 [batch_size, channels, height, width]
 sigma: 噪声水平 [batch_size]

 Returns:
 分数估计 [batch_size, channels, height, width]
 """
 # 噪声嵌入
 embed = self.embed(sigma.log())

 # 输入卷积
 h = self.conv_in(x)

 # 下采样
 hs = [h]
 for layer in self.down:
 if isinstance(layer, ResBlock):
 h = layer(h, embed)
 else:
 h = layer(h)
 hs.append(h)

 # 中间处理
 for layer in self.mid:
 h = layer(h, embed)

 # 上采样
 for layer in self.up:
 if isinstance(layer, ResBlock):
 h = layer(torch.cat([h, hs.pop()], dim=1), embed)
 else:
 h = layer(h)

 # 输出
 h = self.norm_out(h)
 h = F.silu(h)
 h = self.conv_out(h)

 # 重要：按照理论，分数应该与 1/sigma 成比例
 # 但在实践中，我们让网络自己学习这个关系
 return h

# 轻量级版本用于演示
class SimpleNCSN(nn.Module):
 """简化的NCSN用于低维数据"""

 def __init__(self, data_dim, hidden_dim=128, embed_dim=32):
 super().__init__()

 self.embed = SinusoidalEmbedding(embed_dim)

 self.net = nn.Sequential(
 nn.Linear(data_dim + embed_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, data_dim)
 )

 def forward(self, x, sigma):
 sigma_embed = self.embed(sigma)
 h = torch.cat([x, sigma_embed], dim=-1)
 return self.net(h) / sigma.unsqueeze(-1)

# 测试架构
def test_ncsn():
 model = SimpleNCSN(data_dim=2)
 x = torch.randn(10, 2)
 sigma = torch.rand(10) * 10
 score = model(x, sigma)

 print(f"输入形状: {x.shape}")
 print(f"噪声水平: {sigma[:3]}")
 print(f"分数输出形状: {score.shape}")
 print(f"分数尺度: {score.abs().mean():.3f}")

test_ncsn()



#### 架构设计考虑




#### 最佳实践



 - **归一化层的选择**：


 GroupNorm通常比BatchNorm更稳定
 - 条件InstanceNorm适合图像生成
 - LayerNorm适合Transformer架构


 

 - **激活函数**：


 SiLU (Swish) 通常优于ReLU
 - 平滑的激活函数有助于分数估计


 

 - **残差连接**：


 深度网络必需
 - 有助于梯度流动
 - 稳定训练


 

 - **输出尺度**：


 理论上分数 $\propto 1/\sigma$
 - 可以显式建模或让网络学习
 - 注意数值稳定性


 





#### 与其他架构的比较



 架构演进
 从NCSN到后续工作的架构改进：



 - **NCSN (2019)**：RefineNet架构，实例归一化
 - **NCSN++ (2020)**：改进的U-Net，GroupNorm，自注意力
 - **DDPM (2020)**：简化的U-Net，时间嵌入
 - **Score SDE (2021)**：统一框架，更灵活的架构选择





### 4.3.3 退火Langevin动力学



退火Langevin动力学（Annealed Langevin Dynamics, ALD）是NCSN用于生成样本的核心算法。它结合了Langevin MCMC和模拟退火的思想，通过逐步降低噪声水平来生成高质量样本。



#### 基础Langevin动力学



回顾标准的Langevin动力学，对于目标分布 $p(x)$：



 $$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t, \quad z_t \sim \mathcal{N}(0, I)$$



其中 $\epsilon$ 是步长。当 $\epsilon \to 0$ 且迭代次数 $\to \infty$ 时，这个马尔可夫链收敛到 $p(x)$。



#### 退火策略



ALD的关键创新是使用一系列递减的噪声水平 $\{\sigma_i\}_{i=1}^L$，在每个噪声水平运行Langevin动力学：



 退火Langevin动力学算法


 - 初始化：$x_0 \sim \mathcal{N}(0, \sigma_1^2 I)$
 - 对于每个噪声水平 $\sigma_i$，$i = 1, ..., L$：


 运行 $T$ 步Langevin更新：

 $$x_{t+1} = x_t + \epsilon_i s_\theta(x_t, \sigma_i) + \sqrt{2\epsilon_i} z_t$$


 - 使用最后一步的样本作为下一个噪声水平的初始值


 
 - 返回最终样本 $x$






#### 为什么退火有效？



 - **全局探索 → 局部细化**：


 高噪声水平：探索整个空间，跨越模式
 - 低噪声水平：细化局部结构，恢复细节


 
 - **平滑过渡**：相邻噪声水平的分布相似，采样链保持连续性
 - **避免局部最小值**：类似模拟退火，早期的高温阶段帮助逃离局部陷阱





#### 步长选择



步长 $\epsilon_i$ 的选择对算法性能至关重要：



 最优步长（理论）

对于噪声水平 $\sigma_i$，理论最优步长为：


 $$\epsilon_i^* = \frac{c \sigma_i^2}{|\Sigma|^{1/d}}$$


其中 $c$ 是常数，$|\Sigma|$ 是数据协方差矩阵的行列式，$d$ 是维度。



实践中常用的简化：


 $$\epsilon_i = \epsilon \cdot \min(1, \sigma_i^2 / \sigma_L^2)$$





# 退火Langevin动力学实现
import torch
import numpy as np
from tqdm import tqdm

class AnnealedLangevinSampler:
 """退火Langevin动力学采样器"""

 def __init__(
 self,
 score_model,
 sigmas,
 epsilon=2e-5,
 T=100,
 denoise=True,
 device='cuda'
 ):
 """
 Args:
 score_model: 训练好的分数模型
 sigmas: 噪声水平序列（降序）
 epsilon: 基础步长
 T: 每个噪声水平的Langevin步数
 denoise: 是否在最后一步去噪
 """
 self.score_model = score_model
 self.sigmas = sigmas.to(device)
 self.epsilon = epsilon
 self.T = T
 self.denoise = denoise
 self.device = device

 def get_epsilon(self, sigma):
 """自适应步长"""
 # Song & Ermon (2019) 的建议
 return self.epsilon * (sigma / self.sigmas[-1]) ** 2

 @torch.no_grad()
 def sample(self, batch_size, shape, show_progress=True):
 """生成样本

 Args:
 batch_size: 批量大小
 shape: 数据形状（不包括batch维度）
 show_progress: 是否显示进度条

 Returns:
 samples: 生成的样本
 trajectory: 采样轨迹（可选）
 """
 # 初始化：从高斯噪声开始
 x = torch.randn(batch_size, *shape, device=self.device)
 x = x * self.sigmas[0]

 trajectory = [x.clone()]

 # 退火过程
 sigma_levels = tqdm(self.sigmas, desc="Annealing") if show_progress else self.sigmas

 for i, sigma in enumerate(sigma_levels):
 # 当前噪声水平的步长
 epsilon_i = self.get_epsilon(sigma)

 # 在当前噪声水平运行T步Langevin
 for t in range(self.T):
 # 计算分数
 score = self.score_model(x, sigma.expand(batch_size))

 # Langevin更新
 noise = torch.randn_like(x)
 x = x + epsilon_i * score + np.sqrt(2 * epsilon_i) * noise

 trajectory.append(x.clone())

 # 可选的最后去噪步骤
 if self.denoise:
 # 使用最小噪声水平再做一次预测
 score = self.score_model(x, self.sigmas[-1].expand(batch_size))
 x = x + self.sigmas[-1] ** 2 * score

 return x, trajectory

 @torch.no_grad()
 def sample_with_initialization(self, x_init, start_sigma_idx=0):
 """从特定初始值和噪声水平开始采样

 用于图像编辑、插值等任务
 """
 batch_size = x_init.shape[0]
 x = x_init.clone()

 # 从指定的噪声水平开始
 for i in range(start_sigma_idx, len(self.sigmas)):
 sigma = self.sigmas[i]
 epsilon_i = self.get_epsilon(sigma)

 for t in range(self.T):
 score = self.score_model(x, sigma.expand(batch_size))
 noise = torch.randn_like(x)
 x = x + epsilon_i * score + np.sqrt(2 * epsilon_i) * noise

 return x

# 改进的采样器：使用预测-校正方法
class PredictorCorrectorSampler(AnnealedLangevinSampler):
 """预测-校正采样器（Song et al., 2021）"""

 def __init__(self, *args, corrector_steps=1, snr=0.16, **kwargs):
 super().__init__(*args, **kwargs)
 self.corrector_steps = corrector_steps
 self.snr = snr # 信噪比

 @torch.no_grad()
 def sample(self, batch_size, shape, show_progress=True):
 """使用预测-校正方法采样"""
 x = torch.randn(batch_size, *shape, device=self.device)
 x = x * self.sigmas[0]

 sigma_levels = tqdm(
 enumerate(self.sigmas[:-1]),
 total=len(self.sigmas)-1,
 desc="PC Sampling"
 ) if show_progress else enumerate(self.sigmas[:-1])

 for i, sigma in sigma_levels:
 sigma_next = self.sigmas[i + 1]

 # 预测步骤（大步跳跃）
 score = self.score_model(x, sigma.expand(batch_size))
 x_mean = x + (sigma ** 2 - sigma_next ** 2) * score
 noise = torch.randn_like(x)
 x = x_mean + torch.sqrt(sigma_next ** 2 - 0) * noise

 # 校正步骤（Langevin MCMC）
 for _ in range(self.corrector_steps):
 score = self.score_model(x, sigma_next.expand(batch_size))
 noise = torch.randn_like(x)
 epsilon = self.snr * sigma_next ** 2
 x = x + epsilon * score + torch.sqrt(2 * epsilon) * noise

 return x, None

# 实用工具函数
def interpolate_samples(x1, x2, sampler, num_steps=10):
 """在潜在空间中插值两个样本"""
 alphas = torch.linspace(0, 1, num_steps)
 interpolated = []

 for alpha in alphas:
 # 球面线性插值（保持范数）
 x_interp = slerp(x1, x2, alpha)

 # 从中间噪声水平开始去噪
 start_idx = len(sampler.sigmas) // 2
 x_denoised = sampler.sample_with_initialization(
 x_interp, start_sigma_idx=start_idx
 )
 interpolated.append(x_denoised)

 return torch.stack(interpolated)

def slerp(x1, x2, alpha):
 """球面线性插值"""
 x1_norm = x1 / x1.norm(dim=-1, keepdim=True)
 x2_norm = x2 / x2.norm(dim=-1, keepdim=True)

 omega = torch.acos((x1_norm * x2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1))

 return (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * x1 + \
 (torch.sin(alpha * omega) / torch.sin(omega)) * x2

# 使用示例
print("退火Langevin动力学示例：")
print("1. 基础采样器：使用固定步长和T步Langevin")
print("2. 自适应步长：根据噪声水平调整步长")
print("3. 预测-校正：更快的采样，更好的质量")
print("4. 条件采样：从部分噪声开始，用于编辑任务")



#### 收敛性分析



 退火Langevin的收敛保证
 在适当条件下，ALD算法有以下保证：



 - **单个噪声水平**：对于固定的 $\sigma_i$，Langevin动力学以速率 $O(\epsilon)$ 收敛到 $p_{\sigma_i}$
 - **退火过程**：如果相邻分布足够接近（$D_{KL}(p_{\sigma_i} \| p_{\sigma_{i+1}}) 在中等噪声水平使用更多步数
 - 最高和最低噪声水平可以用较少步数


 

 - **最后的去噪步**：

 $$x_{final} = x + \sigma_{min}^2 \nabla_x \log p_{\sigma_{min}}(x)$$


这一步通常能显著提升视觉质量



 - **温度调节**：


 降低温度（减小噪声）：更确定但可能缺乏多样性
 - 提高温度（增大噪声）：更多样但可能质量下降


 

 - **早停策略**：


 不一定要运行到最小噪声
 - 在适当的噪声水平停止可能得到更好的感知质量


 





#### 与其他采样方法的比较



 采样方法对比
 
 
 方法
 速度
 质量
 特点
 
 
 退火Langevin
 慢
 高
 理论保证，稳定
 
 
 DDIM
 快
 高
 确定性，可逆
 
 
 预测-校正
 中等
 很高
 平衡速度和质量
 
 
 DPM-Solver
 很快
 高
 高阶ODE求解器
 
 



## 4.4 Langevin动力学与采样



### 4.4.1 Langevin方程的基础



Langevin方程最初由法国物理学家Paul Langevin在1908年提出，用于描述布朗运动。在机器学习中，Langevin动力学成为了一种强大的采样方法，特别适合基于梯度的采样。



#### 物理起源



考虑一个在势能场 $U(x)$ 中运动的粒子，受到两种力的作用：



 - **确定性力**：$-\nabla U(x)$（势能的负梯度）
 - **随机力**：热噪声，建模为白噪声




过阻尼Langevin方程（忽略惯性项）为：



 $$\frac{dx}{dt} = -\nabla U(x) + \sqrt{2\beta^{-1}} \eta(t)$$



其中 $\beta = 1/(k_B T)$ 是逆温度，$\eta(t)$ 是白噪声（满足 $\langle \eta(t) \rangle = 0$，$\langle \eta(t)\eta(t') \rangle = \delta(t-t')$）。



#### 概率论视角



从概率论角度，如果我们想从分布 $p(x) \propto \exp(-U(x))$ 采样，注意到：



 $$\nabla \log p(x) = -\nabla U(x)$$



因此Langevin方程可以重写为：



 $$dx = \nabla \log p(x) dt + \sqrt{2} dW_t$$



这是一个随机微分方程（SDE），其中 $W_t$ 是标准布朗运动。



 Fokker-Planck方程

Langevin SDE对应的概率密度演化由Fokker-Planck方程描述：


 $$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t \nabla \log p) + \Delta p_t = \nabla \cdot (\nabla p_t + p_t \nabla \log p)$$



稳态解（$\partial p_t/\partial t = 0$）满足：


 $$\nabla p_{\infty} + p_{\infty} \nabla \log p = 0$$



这给出 $p_{\infty} = p$，即目标分布！




#### 离散化方案



为了数值实现，需要离散化SDE。最常用的是Euler-Maruyama方法：



 $$x_{k+1} = x_k + \epsilon \nabla \log p(x_k) + \sqrt{2\epsilon} z_k, \quad z_k \sim \mathcal{N}(0, I)$$



其中 $\epsilon$ 是步长。这个更新规则有两个解释：




#### 两种视角的统一



 - **梯度上升 + 噪声**：在对数概率上做梯度上升，加上探索性噪声
 - **MCMC**：这是Metropolis-adjusted Langevin algorithm (MALA)的特例（当接受率为1时）



这两种视角解释了为什么Langevin动力学既能找到高概率区域，又能正确采样。





# Langevin动力学的基础实现
import torch
import numpy as np

class LangevinDynamics:
 """基础Langevin动力学采样器"""

 def __init__(self, score_fn, step_size=0.01, noise_scale=1.0):
 """
 Args:
 score_fn: 分数函数 ∇log p(x)
 step_size: 步长 ε
 noise_scale: 噪声缩放（温度控制）
 """
 self.score_fn = score_fn
 self.step_size = step_size
 self.noise_scale = noise_scale

 def step(self, x):
 """单步Langevin更新"""
 score = self.score_fn(x)
 noise = torch.randn_like(x)

 x_new = x + self.step_size * score + \
 np.sqrt(2 * self.step_size * self.noise_scale) * noise

 return x_new

 def sample(self, x_init, num_steps, return_trajectory=False):
 """运行Langevin动力学

 Args:
 x_init: 初始点
 num_steps: 步数
 return_trajectory: 是否返回整个轨迹

 Returns:
 最终样本或整个轨迹
 """
 x = x_init.clone()
 trajectory = [x.clone()] if return_trajectory else None

 for _ in range(num_steps):
 x = self.step(x)
 if return_trajectory:
 trajectory.append(x.clone())

 return trajectory if return_trajectory else x

# Metropolis-adjusted Langevin Algorithm (MALA)
class MALA:
 """带Metropolis-Hastings校正的Langevin算法"""

 def __init__(self, log_prob_fn, score_fn, step_size=0.01):
 self.log_prob_fn = log_prob_fn
 self.score_fn = score_fn
 self.step_size = step_size

 def proposal(self, x):
 """Langevin提议分布"""
 score = self.score_fn(x)
 noise = torch.randn_like(x)

 mean = x + self.step_size * score
 x_prop = mean + np.sqrt(2 * self.step_size) * noise

 return x_prop, mean

 def log_proposal_ratio(self, x_new, x_old):
 """计算提议分布的比率 q(x_old|x_new) / q(x_new|x_old)"""
 # 前向提议: x_old -> x_new
 score_old = self.score_fn(x_old)
 mean_forward = x_old + self.step_size * score_old

 # 反向提议: x_new -> x_old
 score_new = self.score_fn(x_new)
 mean_backward = x_new + self.step_size * score_new

 # 高斯提议的对数比率
 log_q_backward = -0.5 * torch.sum((x_old - mean_backward)**2) / (2 * self.step_size)
 log_q_forward = -0.5 * torch.sum((x_new - mean_forward)**2) / (2 * self.step_size)

 return log_q_backward - log_q_forward

 def step(self, x):
 """MALA的一步"""
 # Langevin提议
 x_prop, _ = self.proposal(x)

 # Metropolis-Hastings接受率
 log_alpha = self.log_prob_fn(x_prop) - self.log_prob_fn(x) + \
 self.log_proposal_ratio(x_prop, x)

 # 接受或拒绝
 if torch.rand(1) 



#### 收敛性质



 Langevin动力学的收敛定理
 在适当的正则性条件下（如对数凹分布），离散Langevin动力学有以下性质：




 - **偏差-方差权衡**：


 小步长：小偏差，慢收敛
 - 大步长：大偏差，可能不稳定


 

 - **收敛速率**：对于 $m$-强凸势能，以步长 $\epsilon 使用预条件矩阵：$x_{k+1} = x_k + \epsilon M \nabla \log p(x_k) + \sqrt{2\epsilon M} z_k$
 - $M$ 可以是协方差矩阵的逆或其近似


 

 - **自适应步长**：


 根据局部曲率调整步长
 - 使用Robbins-Monro类型的递减步长


 

 - **方差减少**：


 控制变量方法
 - 使用历史信息（如SAGA类型的更新）


 

 - **并行化**：


 多链并行
 - 异步更新


 





### 4.4.2 离散化与数值稳定性

 [内容待补充]


### 4.4.3 采样算法实现

 [内容待补充]


## 4.5 分数模型与扩散模型的统一



### 4.5.1 两种观点的等价性

 [内容待补充]


### 4.5.2 SDE框架下的统一

 [内容待补充]


### 4.5.3 实践中的差异与选择

 [内容待补充]


## 4.6 实现：训练分数模型



### 4.6.1 数据预处理与噪声调度

 [内容待补充]


### 4.6.2 损失函数与优化

 [内容待补充]


### 4.6.3 评估与可视化

 [内容待补充]


## 4.7 高级主题



### 4.7.1 分数模型的理论性质

 [内容待补充]


### 4.7.2 改进的采样技术

 [内容待补充]


### 4.7.3 条件分数模型

 [内容待补充]


## 4.8 练习题




练习 4.1：分数函数的性质

证明以下性质：



 - 对于高斯分布 $p(x) = \mathcal{N}(x; \mu, \Sigma)$，证明分数函数为 $\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)$
 - 证明 $\mathbb{E}_{x \sim p}[\nabla_x \log p(x)] = 0$（提示：使用分部积分）
 - 对于能量函数 $E(x) = \frac{1}{2}x^T A x + b^T x + c$，求对应的分数函数

**解答：**



 - 高斯分布：
 $$\log p(x) = -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) + C$$
 $$\nabla_x \log p(x) = -\Sigma^{-1}(x-\mu)$$

 - 期望为零：
 $$\mathbb{E}[\nabla_x \log p(x)] = \int p(x) \frac{\nabla p(x)}{p(x)} dx = \int \nabla p(x) dx = \nabla \int p(x) dx = \nabla 1 = 0$$

 - 二次能量函数：
 $$p(x) \propto \exp(-E(x))$$
 $$\nabla_x \log p(x) = -\nabla_x E(x) = -(Ax + b)$$






练习 4.2：实现去噪分数匹配

为二维数据实现一个完整的去噪分数匹配训练流程：



 - 生成"瑞士卷"（Swiss roll）数据集
 - 实现多尺度去噪分数匹配损失
 - 训练一个简单的MLP作为分数网络
 - 使用退火Langevin动力学生成新样本

# 完整实现框架
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_swiss_roll

# 1. 生成数据
def generate_swiss_roll(n_samples=1000):
 X, _ = make_swiss_roll(n_samples, noise=0.1)
 X = X[:, [0, 2]] # 使用x和z坐标
 return torch.tensor(X, dtype=torch.float32)

# 2. 分数网络
class ScoreNet(nn.Module):
 def __init__(self, dim=2, hidden_dim=128):
 super().__init__()
 self.net = nn.Sequential(
 nn.Linear(dim + 1, hidden_dim), # +1 for time/sigma
 nn.ReLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, dim)
 )

 def forward(self, x, sigma):
 t = sigma.log().view(-1, 1)
 h = torch.cat([x, t], dim=1)
 return self.net(h) / sigma.view(-1, 1)

# 3. 训练循环
def train_dsm(model, data, sigmas, num_epochs=1000):
 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

 for epoch in range(num_epochs):
 # 随机批次
 idx = torch.randint(0, len(data), (128,))
 x = data[idx]

 # 随机噪声水平
 sigma_idx = torch.randint(0, len(sigmas), (len(x),))
 sigma = sigmas[sigma_idx]

 # 添加噪声
 noise = torch.randn_like(x)
 x_noisy = x + sigma.view(-1, 1) * noise

 # 预测分数
 score_pred = model(x_noisy, sigma)

 # 损失
 loss = ((score_pred + noise/sigma.view(-1, 1))**2).mean()

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

# 4. 采样
def sample_ald(model, sigmas, num_samples=100):
 x = torch.randn(num_samples, 2) * sigmas[0]

 for sigma in sigmas:
 for _ in range(100): # T steps per sigma
 score = model(x, sigma.expand(num_samples))
 noise = torch.randn_like(x)
 step_size = 0.1 * (sigma/sigmas[-1])**2
 x = x + step_size * score + np.sqrt(2*step_size) * noise

 return x





 练习 4.3：分析NCSN的多尺度效应
 实验分析不同噪声调度对NCSN性能的影响：



 - 实现几何、线性、余弦三种噪声调度
 - 在相同数据上训练三个模型
 - 比较它们的收敛速度和生成质量
 - 可视化不同噪声水平下的分数场





 练习 4.4：从分数模型到扩散模型

推导并验证分数模型与DDPM的等价性：



 - 从DDPM的训练目标出发，推导其与分数匹配的关系
 - 实现一个统一的框架，可以在两种参数化之间转换
 - 验证两种方法在相同设置下产生相同的结果






## 本章小结


在本章中，我们从分数函数的角度深入理解了生成模型，建立了分数模型与扩散模型之间的桥梁：



#### 主要收获



 - **分数函数基础**：理解了分数函数的定义、性质和几何意义
 - **分数匹配技术**：掌握了去噪分数匹配、切片分数匹配等实用方法
 - **NCSN架构**：学习了多尺度噪声的设计和条件化网络架构
 - **Langevin动力学**：深入理解了基于分数的采样算法
 - **统一视角**：看到了分数模型和扩散模型的内在联系




#### 关键要点



 - 分数函数 $\nabla_x \log p(x)$ 提供了无需归一化常数的概率建模方法
 - 去噪分数匹配巧妙地避开了直接计算雅可比矩阵的难题
 - 多尺度方法解决了单一噪声水平的局限性
 - 退火Langevin动力学提供了理论保证的采样方法
 - 分数匹配与DDPM在数学上是等价的，只是参数化不同




#### 与DDPM的对比

 
 
 方面
 DDPM
 分数模型
 
 
 核心思想
 学习去噪
 学习分数函数
 
 
 参数化
 预测噪声 $\epsilon$
 预测分数 $\nabla \log p$
 
 
 理论基础
 变分推断
 分数匹配
 
 
 采样方法
 祖先采样
 Langevin动力学
 
 


#### 展望


基于本章的知识，我们为后续学习打下了坚实基础：



 - **第5章**：将离散时间推广到连续时间，学习SDE/ODE框架
 - **第6章**：探索流匹配等新方法，进一步统一生成模型
 - **第8章**：基于分数函数的理解，开发更快的采样算法




分数视角不仅提供了理解扩散模型的新方式，还启发了许多改进和扩展。在下一章，我们将进入连续时间的世界，看看如何用随机微分方程统一描述所有这些方法！