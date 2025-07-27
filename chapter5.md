[← 上一章](chapter4.md)
 第5章 / 共14章
 [下一章 →](chapter6.md)



# 第5章：连续时间扩散模型 (PDE/SDE)



 到目前为止，我们学习的扩散模型都是在离散时间步上定义的。但如果我们让时间步数趋于无穷，会发生什么？答案是：我们得到了随机微分方程（SDE）！Song等人在2021年的工作"Score-Based Generative Modeling through Stochastic Differential Equations"统一了之前的所有方法，并开启了连续时间建模的新纪元。本章将深入探讨SDE框架，以及相关的概率流ODE和Fokker-Planck方程。



## 5.1 从离散到连续：极限过程



### 5.1.1 离散扩散过程的回顾



在前面的章节中，我们学习了离散时间的扩散模型。让我们回顾其核心结构，为理解连续时间做准备。



#### DDPM的前向过程



DDPM定义了一个马尔可夫链，逐步向数据添加噪声：



 $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$



其中 $\beta_t$ 是预定义的噪声调度。通过重参数化技巧，我们可以直接从 $x_0$ 采样 $x_t$：



 $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$



其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。



#### 关键观察：小步长近似




#### 离散步长的含义


当 $\beta_t$ 很小时，前向过程可以理解为：


 $$x_t = x_{t-1} - \frac{\beta_t}{2} x_{t-1} + \sqrt{\beta_t} z_{t-1}, \quad z_{t-1} \sim \mathcal{N}(0, I)$$



这看起来像是一个离散化的方程：



 - 漂移项：$-\frac{\beta_t}{2} x_{t-1}$（向原点收缩）
 - 扩散项：$\sqrt{\beta_t} z_{t-1}$（添加随机性）





#### Score-based模型的视角



在NCSN中，我们考虑不同噪声水平下的分布：



 $$p_\sigma(x) = \int p_{data}(x') \mathcal{N}(x; x', \sigma^2 I) dx'$$



如果我们让 $\sigma$ 随时间连续变化，$\sigma(t)$，会发生什么？



 离散与连续的对应
 
 
 离散时间
 连续时间
 含义
 
 
 $t \in \{0, 1, ..., T\}$
 $t \in [0, T]$
 时间变量
 
 
 $x_t - x_{t-1}$
 $dx_t$
 无穷小变化
 
 
 $\beta_t$
 $\beta(t)dt$
 噪声强度
 
 
 $z_t \sim \mathcal{N}(0, I)$
 $dW_t$
 布朗运动增量
 
 




# 可视化离散步长的影响
import torch
import numpy as np

def discrete_diffusion_path(x0, betas, return_all=True):
 """模拟离散扩散路径"""
 x = x0.clone()
 path = [x.clone()]

 for beta in betas:
 # 前向扩散步
 noise = torch.randn_like(x)
 x = np.sqrt(1 - beta) * x + np.sqrt(beta) * noise

 if return_all:
 path.append(x.clone())

 return torch.stack(path) if return_all else x

# 比较不同步数的路径
def compare_discretizations(x0, T=1.0, num_steps_list=[10, 50, 1000]):
 """比较不同离散化的效果"""
 paths = {}

 for num_steps in num_steps_list:
 # 线性噪声调度
 betas = torch.linspace(0.0001, 0.02, num_steps)

 # 调整到相同的总噪声量
 betas = betas * T / num_steps

 path = discrete_diffusion_path(x0, betas)
 paths[num_steps] = path

 # 分析路径的统计性质
 for num_steps, path in paths.items():
 final_mean = path[-1].mean()
 final_std = path[-1].std()
 print(f"步数={num_steps:4d}: 最终均值={final_mean:.3f}, 标准差={final_std:.3f}")

 return paths

# 演示
x0 = torch.randn(1000, 2) # 1000个2D点
paths = compare_discretizations(x0)
print("\n观察：随着步数增加，离散路径趋于某个极限过程")



#### 为什么需要连续时间？




#### 连续时间的优势



 - **理论优雅**：可以使用强大的SDE理论工具
 - **灵活采样**：可以在任意时刻停止或评估
 - **数值方法**：可以使用高阶ODE/SDE求解器
 - **统一框架**：不同的离散模型成为同一SDE的不同离散化





### 5.1.2 时间步趋于无穷的极限


 现在让我们严格地推导当时间步数趋于无穷时会发生什么。这个过程揭示了SDE的自然出现。



#### 极限过程的设置



考虑将时间区间 $[0, T]$ 分成 $N$ 份，每份长度 $\Delta t = T/N$。在离散设置中：



 $$x_{k+1} = \sqrt{1 - \beta_k} x_k + \sqrt{\beta_k} z_k, \quad z_k \sim \mathcal{N}(0, I)$$



为了保持合理的扩散速度，我们需要让 $\beta_k = \tilde{\beta}(t_k) \Delta t$，其中 $\tilde{\beta}(t)$ 是连续函数。



 关键洞察：Taylor展开

当 $\Delta t \to 0$ 时，我们可以展开：


 $$\sqrt{1 - \tilde{\beta}(t)\Delta t} = 1 - \frac{\tilde{\beta}(t)}{2}\Delta t + O((\Delta t)^2)$$



因此离散更新变为：


 $$x_{k+1} - x_k = -\frac{\tilde{\beta}(t_k)}{2} x_k \Delta t + \sqrt{\tilde{\beta}(t_k)\Delta t} z_k$$




#### 布朗运动的出现



关键观察：$\sqrt{\Delta t} z_k$ 在极限下收敛到布朗运动的增量！




#### 从离散噪声到布朗运动


定义 $W_N(t) = \sum_{k=0}^{\lfloor t/\Delta t \rfloor} \sqrt{\Delta t} z_k$，则：



 - $\mathbb{E}[W_N(t)] = 0$
 - $\mathbb{E}[W_N(t)^2] = t$
 - 增量独立且正态分布



根据Donsker定理，$W_N(t) \xrightarrow{d} W(t)$（标准布朗运动）。




#### SDE的导出



取极限 $N \to \infty$（即 $\Delta t \to 0$），我们得到：



 $$dx_t = -\frac{\tilde{\beta}(t)}{2} x_t dt + \sqrt{\tilde{\beta}(t)} dW_t$$



更一般地，我们可以写成：



 $$dx_t = f(x_t, t) dt + g(t) dW_t$$



其中 $f(x_t, t) = -\frac{\tilde{\beta}(t)}{2} x_t$ 是漂移系数，$g(t) = \sqrt{\tilde{\beta}(t)}$ 是扩散系数。



 具体例子：VP (Variance Preserving) SDE

DDPM的连续时间极限给出VP-SDE：


 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dW_t$$



其中 $\beta(t)$ 是连续的噪声调度函数。常见选择：



 - 线性：$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$
 - 余弦：$\beta(t) = \pi \sin^2(\frac{t}{T} \cdot \frac{\pi}{2})$






# 数值验证：离散过程收敛到SDE
import torch
import numpy as np
from scipy.stats import kstest

class DiscreteToSDE:
 """验证离散过程收敛到SDE的数值实验"""

 def __init__(self, beta_fn, T=1.0):
 self.beta_fn = beta_fn
 self.T = T

 def discrete_evolution(self, x0, N):
 """离散扩散过程"""
 dt = self.T / N
 x = x0.clone()

 for k in range(N):
 t = k * dt
 beta_dt = self.beta_fn(t) * dt

 # 离散更新
 noise = torch.randn_like(x)
 x = np.sqrt(1 - beta_dt) * x + np.sqrt(beta_dt) * noise

 return x

 def sde_solution(self, x0, t):
 """VP-SDE的解析解（当beta为常数时）"""
 # 对于一般的beta(t)，需要数值积分
 # 这里简化为常数情况
 beta_avg = self.beta_fn(self.T/2) # 近似

 mean_factor = np.exp(-0.5 * beta_avg * t)
 var_factor = 1 - np.exp(-beta_avg * t)

 mean = mean_factor * x0
 std = np.sqrt(var_factor)

 return mean, std

 def test_convergence(self, x0, N_values=[10, 50, 100, 500, 1000]):
 """测试收敛性"""
 n_samples = 10000
 x0_batch = x0.repeat(n_samples, 1)

 results = {}

 for N in N_values:
 # 运行离散过程
 x_final = self.discrete_evolution(x0_batch, N)

 # 理论分布
 mean_theory, std_theory = self.sde_solution(x0, self.T)

 # 比较统计量
 mean_empirical = x_final.mean(dim=0)
 std_empirical = x_final.std(dim=0)

 # KS检验（对第一个维度）
 x_normalized = (x_final[:, 0] - mean_theory[0]) / std_theory
 ks_stat, p_value = kstest(x_normalized.numpy(), 'norm')

 results[N] = {
 'mean_error': torch.norm(mean_empirical - mean_theory).item(),
 'std_error': torch.norm(std_empirical - std_theory).item(),
 'ks_stat': ks_stat,
 'p_value': p_value
 }

 return results

# 运行实验
def demonstrate_convergence():
 # 定义beta函数
 beta_fn = lambda t: 0.1 + 10 * t # 线性调度

 # 初始点
 x0 = torch.tensor([1.0, -0.5])

 # 测试收敛
 tester = DiscreteToSDE(beta_fn)
 results = tester.test_convergence(x0)

 print("离散过程 → SDE 收敛性分析")
 print("="*60)
 print(f"{'N':>6} | {'均值误差':>10} | {'标准差误差':>10} | {'KS统计量':>10} | {'p值':>10}")
 print("-"*60)

 for N, res in results.items():
 print(f"{N:6d} | {res['mean_error']:10.6f} | {res['std_error']:10.6f} | "
 f"{res['ks_stat']:10.6f} | {res['p_value']:10.6f}")

 print("\n结论：随着N增加，离散过程的分布收敛到SDE的理论分布")

demonstrate_convergence()



#### 数学严格性



 收敛定理（简化版）
 设离散过程 $\{X^N_k\}$ 由以下递归定义：


 $$X^N_{k+1} = X^N_k + f(X^N_k, t_k)\Delta t + g(t_k)\sqrt{\Delta t} Z_k$$



在适当的正则性条件下（Lipschitz连续性等），当 $N \to \infty$ 时：


 $$X^N_{\lfloor t/\Delta t \rfloor} \xrightarrow{d} X_t$$



其中 $X_t$ 是SDE的解：$dX_t = f(X_t, t)dt + g(t)dW_t$。




### 5.1.3 SDE的直观理解



随机微分方程（SDE）初看起来可能很抽象，但它描述的是一个非常自然的现象：带有随机扰动的动力系统。让我们通过多个角度来建立直观理解。



#### 粒子运动的视角




#### 布朗运动的发现


1827年，植物学家Robert Brown观察到花粉在水中的无规则运动。Einstein在1905年解释了这一现象：



 - 花粉受到水分子的随机碰撞
 - 宏观运动 = 确定性漂移 + 随机扰动
 - 这正是SDE描述的内容！





SDE的一般形式 $dx_t = f(x_t, t)dt + g(t)dW_t$ 可以理解为：



 物理类比
 
 
 SDE项
 物理含义
 在扩散模型中
 
 
 $f(x_t, t)dt$
 确定性力（如重力、摩擦）
 向噪声分布的漂移
 
 
 $g(t)dW_t$
 随机碰撞（热运动）
 注入的高斯噪声
 
 
 $x_t$
 粒子位置
 数据点的状态
 
 



#### 信号处理的视角



在信号处理中，SDE描述了信号如何被噪声逐渐破坏：



 $$\text{带噪信号}(t) = \text{衰减} \cdot \text{原始信号}(t) + \text{累积噪声}(t)$$



这正对应于扩散模型的前向过程：清晰图像逐渐变成噪声。



#### 概率演化的视角



 从点到分布

SDE不仅描述单个粒子的轨迹，更重要的是描述概率分布的演化：



 - $x_0$ 开始是一个确定的点（或某个初始分布）
 - 随着时间推移，不确定性增加
 - $p(x_t|x_0)$ 变得越来越分散
 - 最终收敛到某个稳态分布（如标准正态）






# 可视化SDE的直观含义
import torch
import numpy as np

class SDEVisualization:
 """通过模拟展示SDE的不同方面"""

 def __init__(self, drift_fn, diffusion_fn):
 """
 Args:
 drift_fn: f(x, t) - 漂移函数
 diffusion_fn: g(t) - 扩散系数函数
 """
 self.f = drift_fn
 self.g = diffusion_fn

 def simulate_paths(self, x0, T, dt=0.01, n_paths=100):
 """模拟多条SDE路径"""
 n_steps = int(T / dt)
 paths = torch.zeros(n_paths, n_steps + 1, x0.shape[-1])
 paths[:, 0] = x0

 for i in range(n_steps):
 t = i * dt
 x = paths[:, i]

 # Euler-Maruyama方法
 drift = self.f(x, t) * dt
 diffusion = self.g(t) * np.sqrt(dt) * torch.randn_like(x)

 paths[:, i + 1] = x + drift + diffusion

 return paths

 def analyze_distribution_evolution(self, x0, T, checkpoints=[0.1, 0.5, 1.0, 2.0]):
 """分析分布随时间的演化"""
 print("分布演化分析")
 print("="*50)

 for t in checkpoints:
 if t > T:
 continue

 # 模拟到时刻t
 paths = self.simulate_paths(x0, t, n_paths=10000)
 final_x = paths[:, -1]

 # 统计量
 mean = final_x.mean(dim=0)
 std = final_x.std(dim=0)

 print(f"t = {t:.1f}:")
 print(f" 均值: {mean.numpy()}")
 print(f" 标准差: {std.numpy()}")
 print(f" 数据范围: [{final_x.min():.2f}, {final_x.max():.2f}]")
 print()

# 示例1：Ornstein-Uhlenbeck过程（均值回归）
def ou_drift(x, t, theta=1.0, mu=0.0):
 """OU过程的漂移：回归到均值mu"""
 return theta * (mu - x)

def constant_diffusion(t, sigma=1.0):
 """常数扩散系数"""
 return sigma

print("示例1: Ornstein-Uhlenbeck过程（金融中的均值回归模型）")
print("-"*50)
ou_sde = SDEVisualization(ou_drift, constant_diffusion)
x0 = torch.tensor([5.0]) # 从远离均值的点开始
ou_sde.analyze_distribution_evolution(x0, T=5.0)

# 示例2：扩散模型的VP-SDE
def vp_drift(x, t, beta_min=0.1, beta_max=20.0):
 """VP-SDE的漂移"""
 beta_t = beta_min + t * (beta_max - beta_min)
 return -0.5 * beta_t * x

def vp_diffusion(t, beta_min=0.1, beta_max=20.0):
 """VP-SDE的扩散系数"""
 beta_t = beta_min + t * (beta_max - beta_min)
 return np.sqrt(beta_t)

print("\n示例2: VP-SDE（扩散模型）")
print("-"*50)
vp_sde = SDEVisualization(vp_drift, vp_diffusion)
x0 = torch.randn(2) # 2D随机初始点
vp_sde.analyze_distribution_evolution(x0, T=1.0)



#### 几何视角：流形上的随机游走


 在高维空间中，SDE可以理解为数据流形上的随机游走：




#### 数据流形的破坏与重建



 - **前向SDE**：将数据从低维流形"推离"到整个高维空间
 - **反向SDE**：学习如何将散布的点"拉回"到原始流形
 - **分数函数**：在每个点指示回到流形的方向





#### 控制论视角：噪声作为正则化



 为什么要加噪声？

添加噪声看似是破坏信息，但实际上有多个好处：



 - **覆盖支撑集**：确保模型见过所有可能的输入
 - **平滑优化景观**：避免分数函数的奇异性
 - **连接数据点**：在数据点之间建立概率路径
 - **隐式正则化**：防止模型记忆训练数据





#### 信息论视角：熵的增加与减少



 熵的演化

前向SDE增加熵（不确定性），反向SDE减少熵：


 $$H[p_t] = H[p_0] + \int_0^t \mathbb{E}_{p_s}\left[\frac{|g(s)|^2}{2}\right] ds$$


这解释了为什么：



 - 前向过程最终收敛到最大熵分布（高斯分布）
 - 反向过程需要学习分数函数来"注入"信息






# 演示信息论视角
def entropy_evolution_demo():
 """展示熵随时间的变化"""
 import torch.distributions as dist

 # 初始分布：混合高斯（低熵）
 mix_weights = torch.tensor([0.3, 0.7])
 components = [
 dist.Normal(-2.0, 0.5),
 dist.Normal(2.0, 0.5)
 ]

 def estimate_entropy(samples):
 """估计样本的差分熵（使用KDE）"""
 # 简化：使用高斯核密度估计
 n = len(samples)
 h = 1.06 * samples.std() * (n ** (-1/5)) # Silverman's rule

 # 计算每个点的密度
 densities = []
 for x in samples[:100]: # 子采样以加速
 kde = torch.exp(-0.5 * ((samples - x) / h) ** 2) / (h * np.sqrt(2 * np.pi))
 density = kde.mean()
 densities.append(density)

 # 熵 = -E[log p(x)]
 log_densities = torch.log(torch.tensor(densities) + 1e-10)
 entropy = -log_densities.mean()
 return entropy.item()

 # 模拟扩散过程
 t_values = [0, 0.1, 0.5, 1.0, 2.0]
 n_samples = 5000

 print("熵的演化（扩散过程）")
 print("="*40)

 for t in t_values:
 # 采样初始分布
 component_idx = torch.multinomial(mix_weights, n_samples, replacement=True)
 samples = torch.zeros(n_samples)
 for i, comp in enumerate(components):
 mask = component_idx == i
 samples[mask] = comp.sample((mask.sum(),))

 # 应用扩散（简化：直接加噪声）
 noise_scale = np.sqrt(1 - np.exp(-t)) # 对应VP-SDE
 signal_scale = np.exp(-t/2)

 diffused_samples = signal_scale * samples + noise_scale * torch.randn_like(samples)

 # 估计熵
 entropy = estimate_entropy(diffused_samples)

 # 理论最大熵（标准正态分布）
 max_entropy = 0.5 * np.log(2 * np.pi * np.e)

 print(f"t = {t:.1f}: 熵 ≈ {entropy:.3f} (最大熵 = {max_entropy:.3f})")

 print("\n观察：熵单调增加，趋向于高斯分布的最大熵")

entropy_evolution_demo()



#### 实践指南：选择SDE的艺术




#### 不同SDE的特点

 
 
 SDE类型
 特点
 适用场景
 
 
 VP-SDE
 方差保持，信号逐渐衰减
 图像生成（DDPM类）
 
 
 VE-SDE
 方差爆炸，信号保持
 分数匹配（NCSN类）
 
 
 sub-VP-SDE
 介于两者之间
 通用框架
 
 



## 5.2 前向SDE：连续时间的扩散过程



### 5.2.1 SDE的一般形式


 现在让我们系统地研究用于扩散模型的SDE。我们将看到，不同的SDE选择对应于不同的离散扩散模型。



#### 扩散SDE的标准形式



用于生成建模的前向SDE通常具有以下形式：



 $$dx = f(x, t) dt + g(t) dW_t$$



其中：



 - $x \in \mathbb{R}^d$ 是状态变量（如图像）
 - $f: \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ 是漂移系数
 - $g: [0, T] \to \mathbb{R}$ 是扩散系数（标量函数）
 - $W_t$ 是标准布朗运动





#### 为什么g(t)是标量？


在大多数扩散模型中，我们假设噪声在各个维度上是独立同分布的。这简化了理论分析和实际实现。更一般的情况下，$g(t)$ 可以是矩阵值函数。




#### 边缘分布的演化



给定初始分布 $p_0(x) = p_{data}(x)$，SDE诱导了一个时变的边缘分布 $p_t(x)$。我们希望：



 设计目标


 - **覆盖数据分布**：$p_0(x) = p_{data}(x)$
 - **收敛到已知分布**：$p_T(x) \approx \pi(x)$，其中 $\pi$ 是易于采样的先验分布
 - **平滑过渡**：$p_t$ 随 $t$ 连续变化
 - **可逆性**：存在反向SDE从 $p_T$ 回到 $p_0$





#### 三种经典SDE家族



Song等人(2021)总结了三种主要的SDE家族：



 1. Variance Exploding (VE) SDE

 $$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dW_t$$


特点：



 - 没有漂移项（$f(x,t) = 0$）
 - 方差随时间增加：$\mathbb{E}[||x_t||^2] = ||x_0||^2 + \sigma^2(t)$
 - 对应于NCSN中的噪声注入过程





 2. Variance Preserving (VP) SDE

 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dW_t$$


特点：



 - 线性漂移项使信号衰减
 - 在适当的 $\beta(t)$ 下，方差保持接近常数
 - 对应于DDPM的连续时间扩展





 3. Sub-VP SDE

 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})} dW_t$$


特点：



 - 漂移项与VP-SDE相同
 - 扩散系数被调整以确保良好的收敛性质
 - 提供更灵活的框架






# 实现三种SDE家族
import torch
import numpy as np
from abc import ABC, abstractmethod

class SDE(ABC):
 """SDE基类"""

 def __init__(self, T=1.0):
 self.T = T

 @abstractmethod
 def drift(self, x, t):
 """漂移系数 f(x,t)"""
 pass

 @abstractmethod
 def diffusion(self, t):
 """扩散系数 g(t)"""
 pass

 @abstractmethod
 def marginal_prob(self, x0, t):
 """边缘分布 p(x_t|x_0) 的均值和标准差"""
 pass

 def sample_trajectory(self, x0, n_steps=1000):
 """使用Euler-Maruyama方法采样轨迹"""
 dt = self.T / n_steps
 trajectory = [x0]
 x = x0.clone()

 for i in range(n_steps):
 t = i * dt
 drift = self.drift(x, t) * dt
 diffusion = self.diffusion(t) * np.sqrt(dt) * torch.randn_like(x)
 x = x + drift + diffusion
 trajectory.append(x.clone())

 return torch.stack(trajectory)

class VESDE(SDE):
 """Variance Exploding SDE"""

 def __init__(self, sigma_min=0.01, sigma_max=50.0, T=1.0):
 super().__init__(T)
 self.sigma_min = sigma_min
 self.sigma_max = sigma_max

 def sigma(self, t):
 """噪声调度函数"""
 return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

 def drift(self, x, t):
 return torch.zeros_like(x)

 def diffusion(self, t):
 sigma_t = self.sigma(t)
 # d\sigma^2/dt = 2\sigma d\sigma/dt
 return sigma_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min) / self.T)

 def marginal_prob(self, x0, t):
 sigma_t = self.sigma(t)
 mean = x0
 std = sigma_t
 return mean, std

class VPSDE(SDE):
 """Variance Preserving SDE"""

 def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
 super().__init__(T)
 self.beta_min = beta_min
 self.beta_max = beta_max

 def beta(self, t):
 """线性噪声调度"""
 return self.beta_min + (self.beta_max - self.beta_min) * t / self.T

 def drift(self, x, t):
 return -0.5 * self.beta(t) * x

 def diffusion(self, t):
 return np.sqrt(self.beta(t))

 def marginal_prob(self, x0, t):
 # 线性SDE的解析解
 log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) / self.T - 0.5 * t * self.beta_min
 mean = torch.exp(log_mean_coeff) * x0
 std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
 return mean, std

class SubVPSDE(SDE):
 """Sub-VP SDE"""

 def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
 super().__init__(T)
 self.beta_min = beta_min
 self.beta_max = beta_max

 def beta(self, t):
 return self.beta_min + (self.beta_max - self.beta_min) * t / self.T

 def drift(self, x, t):
 return -0.5 * self.beta(t) * x

 def diffusion(self, t):
 # 简化：使用近似积分
 integral = 0.5 * t**2 * (self.beta_max - self.beta_min) / self.T + t * self.beta_min
 return np.sqrt(self.beta(t) * (1 - np.exp(-2 * integral)))

 def marginal_prob(self, x0, t):
 # 与VP-SDE相同的边缘均值
 log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) / self.T - 0.5 * t * self.beta_min
 mean = torch.exp(log_mean_coeff) * x0
 # 但标准差不同
 integral = 0.5 * t**2 * (self.beta_max - self.beta_min) / self.T + t * self.beta_min
 std = torch.sqrt(1 - torch.exp(-integral))
 return mean, std

# 比较不同SDE的性质
def compare_sdes():
 """比较三种SDE的边缘分布"""
 x0 = torch.randn(2) # 2D初始点

 sdes = {
 'VE-SDE': VESDE(),
 'VP-SDE': VPSDE(),
 'Sub-VP': SubVPSDE()
 }

 t_values = torch.linspace(0, 1.0, 5)

 print("不同SDE的边缘分布演化")
 print("="*70)
 print(f"{'SDE类型':^10} | {'t':^5} | {'均值范数':^12} | {'标准差':^12} | {'信噪比':^12}")
 print("-"*70)

 for name, sde in sdes.items():
 for t in t_values:
 mean, std = sde.marginal_prob(x0, t)
 mean_norm = torch.norm(mean)
 snr = mean_norm / (std + 1e-8) # 信噪比

 print(f"{name:^10} | {t:5.2f} | {mean_norm:12.6f} | {std:12.6f} | {snr:12.6f}")
 print("-"*70)

compare_sdes()



#### 从ODE视角理解SDE



 确定性 vs 随机性
 SDE可以看作是ODE加上随机扰动：


 $$\underbrace{dx = f(x,t)dt}_{\text{ODE部分}} + \underbrace{g(t)dW_t}_{\text{随机扰动}}$$



这种分解有助于：



 - 理解概率流ODE（去除随机项后的确定性动力学）
 - 设计数值求解器（借鉴ODE方法）
 - 分析稳定性和收敛性





#### 选择SDE的实用指南




#### 如何选择适合的SDE？



 - **VE-SDE**：


 适合高分辨率图像
 - 保留原始信号结构
 - 但最终分布难以控制


 
 - **VP-SDE**：


 最终收敛到标准正态
 - 理论分析更简单
 - 与DDPM兼容


 
 - **Sub-VP-SDE**：


 更灵活的框架
 - 可以调节收敛速度
 - 数值稳定性更好


 





### 5.2.2 常见的SDE选择



在实践中，选择合适的SDE至关重要。不同的选择会影响模型的训练稳定性、生成质量和采样效率。让我们深入探讨实际应用中的SDE设计。



#### 噪声调度的设计



SDE的核心是噪声调度函数，它控制着扩散过程的速度和特性。



 常见的噪声调度


 - **线性调度**（Linear Schedule）
 $$\beta(t) = \beta_{\text{min}} + t(\beta_{\text{max}} - \beta_{\text{min}})$$


 简单直观
 - DDPM的原始选择
 - 可能在开始时太快，结束时太慢


 

 - **余弦调度**（Cosine Schedule）
 $$\bar{\alpha}(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$


 由Nichol & Dhariwal (2021)提出
 - 在整个过程中更均匀地破坏信息
 - 特别适合高分辨率图像


 

 - **二次调度**（Quadratic Schedule）
 $$\beta(t) = \beta_{\text{min}} + (\beta_{\text{max}} - \beta_{\text{min}})t^2$$


 开始时缓慢，后期加速
 - 保留更多的早期信息


 






# 实现和比较不同的噪声调度
import torch
import numpy as np

class NoiseSchedule:
 """噪声调度的基类"""

 def __init__(self, T=1.0):
 self.T = T

 def beta(self, t):
 """返回时刻t的beta值"""
 raise NotImplementedError

 def alpha_bar(self, t):
 """返回累积alpha值"""
 # 对于连续时间，需要积分
 # 这里使用数值近似
 n_steps = 1000
 dt = t / n_steps
 alpha_bar = 1.0

 for i in range(n_steps):
 t_i = i * dt
 alpha_bar *= (1 - self.beta(t_i) * dt)

 return alpha_bar

 def snr(self, t):
 """信噪比 (Signal-to-Noise Ratio)"""
 alpha_bar = self.alpha_bar(t)
 return alpha_bar / (1 - alpha_bar + 1e-8)

class LinearSchedule(NoiseSchedule):
 """线性噪声调度"""

 def __init__(self, beta_min=0.0001, beta_max=0.02, T=1.0):
 super().__init__(T)
 self.beta_min = beta_min
 self.beta_max = beta_max

 def beta(self, t):
 return self.beta_min + (t / self.T) * (self.beta_max - self.beta_min)

class CosineSchedule(NoiseSchedule):
 """余弦噪声调度"""

 def __init__(self, s=0.008, T=1.0):
 super().__init__(T)
 self.s = s

 def alpha_bar(self, t):
 # 直接定义alpha_bar而不是beta
 return np.cos((t / self.T + self.s) / (1 + self.s) * np.pi / 2) ** 2

 def beta(self, t):
 # 从alpha_bar推导beta
 dt = 1e-5
 alpha_bar_t = self.alpha_bar(t)
 alpha_bar_t_dt = self.alpha_bar(min(t + dt, self.T))

 # beta = 1 - alpha_t = 1 - alpha_bar_t / alpha_bar_{t-1}
 return 1 - alpha_bar_t_dt / (alpha_bar_t + 1e-8)

class QuadraticSchedule(NoiseSchedule):
 """二次噪声调度"""

 def __init__(self, beta_min=0.0001, beta_max=0.02, T=1.0):
 super().__init__(T)
 self.beta_min = beta_min
 self.beta_max = beta_max

 def beta(self, t):
 return self.beta_min + (t / self.T) ** 2 * (self.beta_max - self.beta_min)

# 比较不同调度的特性
def compare_schedules():
 """可视化和比较不同的噪声调度"""
 schedules = {
 'Linear': LinearSchedule(),
 'Cosine': CosineSchedule(),
 'Quadratic': QuadraticSchedule()
 }

 t_values = np.linspace(0, 1.0, 11)

 print("噪声调度比较")
 print("="*80)
 print(f"{'Schedule':^10} | {'t':^5} | {'beta(t)':^10} | {'alpha_bar(t)':^12} | {'SNR':^10} | {'log10(SNR)':^10}")
 print("-"*80)

 for name, schedule in schedules.items():
 for t in t_values:
 beta_t = schedule.beta(t)
 alpha_bar_t = schedule.alpha_bar(t)
 snr_t = schedule.snr(t)
 log_snr = np.log10(snr_t + 1e-10)

 print(f"{name:^10} | {t:5.2f} | {beta_t:10.6f} | {alpha_bar_t:12.6f} | {snr_t:10.2f} | {log_snr:10.2f}")
 print("-"*80)

 # 分析关键指标
 print("\n关键观察：")
 print("1. Linear: SNR下降最快，可能导致早期信息丢失过快")
 print("2. Cosine: SNR下降更均匀，在中间阶段保留更多信息")
 print("3. Quadratic: 早期保留最多信息，后期快速下降")

compare_schedules()



#### 离散化与连续时间的对应



 从离散到连续的映射
 给定离散扩散模型的参数 $\{\beta_i\}_{i=1}^T$，如何构造对应的连续SDE？




 - **时间映射**：将离散步骤 $i \in \{1, ..., T\}$ 映射到连续时间 $t \in [0, 1]$：
 $$t = i/T$$


 - **插值beta函数**：
 $$\beta(t) = T \cdot \beta_{\lfloor tT \rfloor}$$
 需要乘以T来保持正确的时间尺度。


 - **验证等价性**：确保离散采样和SDE模拟给出相似的边缘分布。





#### 特殊SDE设计



 1. 保持数据范围的SDE

对于图像数据（通常在[-1, 1]范围内），我们可能希望设计保持这个范围的SDE：



 $$dx = -\frac{\beta(t)}{2}(x - \tanh(x))dt + \sqrt{\beta(t)} dW_t$$



这里的非线性漂移项 $\tanh(x)$ 在边界附近提供"推力"，防止样本逃离有效范围。




 2. 条件SDE

对于条件生成，我们可以修改SDE以包含条件信息 $y$：



 $$dx = f(x, t, y)dt + g(t)dW_t$$



常见选择：



 - 条件漂移：$f(x, t, y) = -\frac{\beta(t)}{2}x + h(y, t)$
 - 条件扩散：$g(t, y) = \sqrt{\beta(t)} \cdot \sigma(y)$






# 特殊SDE的实现
class BoundedSDE(SDE):
 """保持数据在有界范围内的SDE"""

 def __init__(self, beta_fn, bounds=(-1, 1), T=1.0):
 super().__init__(T)
 self.beta_fn = beta_fn
 self.lower, self.upper = bounds
 self.range = self.upper - self.lower

 def drift(self, x, t):
 beta_t = self.beta_fn(t)
 # 归一化到[-1, 1]
 x_norm = 2 * (x - self.lower) / self.range - 1
 # 非线性漂移
 drift_norm = -0.5 * beta_t * (x_norm - torch.tanh(x_norm))
 # 转换回原始范围
 return drift_norm * self.range / 2

 def diffusion(self, t):
 return np.sqrt(self.beta_fn(t))

class ConditionalVPSDE(SDE):
 """条件VP-SDE"""

 def __init__(self, beta_min=0.1, beta_max=20.0, condition_dim=128, T=1.0):
 super().__init__(T)
 self.beta_min = beta_min
 self.beta_max = beta_max
 self.condition_dim = condition_dim

 # 条件编码器（简化示例）
 self.condition_encoder = torch.nn.Sequential(
 torch.nn.Linear(condition_dim, 256),
 torch.nn.ReLU(),
 torch.nn.Linear(256, 1)
 )

 def beta(self, t):
 return self.beta_min + (self.beta_max - self.beta_min) * t / self.T

 def drift(self, x, t, condition=None):
 base_drift = -0.5 * self.beta(t) * x

 if condition is not None:
 # 条件调制
 with torch.no_grad():
 modulation = self.condition_encoder(condition)
 base_drift = base_drift * (1 + 0.1 * modulation)

 return base_drift

 def diffusion(self, t, condition=None):
 base_diffusion = np.sqrt(self.beta(t))

 if condition is not None:
 # 条件可以影响噪声强度
 return base_diffusion

 return base_diffusion

# 测试特殊SDE
def test_special_sdes():
 """测试特殊设计的SDE"""
 print("\n特殊SDE测试")
 print("="*60)

 # 1. 有界SDE
 print("1. 有界SDE（保持数据在[-1, 1]内）")
 beta_fn = lambda t: 0.1 + 10 * t
 bounded_sde = BoundedSDE(beta_fn, bounds=(-1, 1))

 # 测试边界行为
 x_boundary = torch.tensor([0.9, -0.9, 0.0])
 drift = bounded_sde.drift(x_boundary, 0.5)
 print(f"边界点的漂移: {drift.numpy()}")
 print("观察：接近边界的点有向内的漂移\n")

 # 2. 条件SDE
 print("2. 条件SDE")
 cond_sde = ConditionalVPSDE()

 # 不同条件下的漂移
 x = torch.randn(3)
 t = 0.5

 # 无条件
 drift_uncond = cond_sde.drift(x, t, condition=None)

 # 有条件
 condition = torch.randn(128)
 drift_cond = cond_sde.drift(x, t, condition=condition)

 print(f"无条件漂移: {drift_uncond.numpy()}")
 print(f"有条件漂移: {drift_cond.numpy()}")
 print(f"差异: {(drift_cond - drift_uncond).numpy()}")

test_special_sdes()



#### 实用建议：如何选择和调试SDE




#### SDE设计清单



 - **检查信噪比曲线**


 log SNR应该从正值（高信号）单调下降到负值（高噪声）
 - 下降速度影响信息保留和生成质量的平衡


 

 - **验证最终分布**


 $p_T(x)$应该接近先验分布（如标准正态）
 - 可以通过蒙特卡罗模拟验证


 

 - **测试数值稳定性**


 确保drift和diffusion项在整个时间范围内有界
 - 避免在t=0或t=T附近出现数值问题


 

 - **考虑计算效率**


 简单的函数形式（如线性）计算更快
 - 复杂的调度可能提供更好的质量但增加计算成本


 





 经验法则


 - **低分辨率图像**：线性调度通常足够
 - **高分辨率图像**：余弦调度表现更好
 - **非图像数据**：可能需要专门设计的SDE
 - **条件生成**：考虑让条件影响噪声调度





### 5.2.3 边缘分布的演化


 理解边缘分布 $p_t(x)$ 如何随时间演化是掌握扩散模型的关键。这不仅关系到理论分析，更直接影响到实际的训练和采样。



#### 边缘分布的定义



 边缘分布

给定SDE $dx_t = f(x_t, t)dt + g(t)dW_t$ 和初始分布 $p_0(x)$，边缘分布定义为：


 $$p_t(x) = \int p(x_t = x | x_0) p_0(x_0) dx_0$$



其中 $p(x_t | x_0)$ 是转移核（transition kernel），描述了从 $x_0$ 到 $x_t$ 的概率转移。




#### 线性SDE的解析解



对于线性SDE（如VP-SDE），我们可以得到边缘分布的显式解。



 以VP-SDE为例

对于 $dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dW_t$，转移核为：


 $$p(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}(t)} x_0, (1-\bar{\alpha}(t))I)$$



其中：


 $$\bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s) ds\right)$$



这个结果告诉我们：



 - 均值随时间指数衰减
 - 方差逐渐增加到接近1
 - 最终分布接近标准正态






# 可视化边缘分布的演化
import torch
import numpy as np
from scipy import stats

class MarginalDistribution:
 """计算和分析SDE的边缘分布"""

 def __init__(self, sde):
 self.sde = sde

 def sample_marginal(self, x0, t, n_samples=1000):
 """通过蒙特卡罗采样边缘分布"""
 if hasattr(self.sde, 'marginal_prob'):
 # 如果有解析解，直接使用
 mean, std = self.sde.marginal_prob(x0, t)
 samples = mean + std * torch.randn(n_samples, *x0.shape)
 else:
 # 否则通过模拟
 samples = []
 for _ in range(n_samples):
 trajectory = self.sde.sample_trajectory(x0, n_steps=100)
 samples.append(trajectory[-1])
 samples = torch.stack(samples)

 return samples

 def analyze_evolution(self, x0, time_points):
 """分析边缘分布随时间的变化"""
 results = []

 for t in time_points:
 samples = self.sample_marginal(x0, t, n_samples=5000)

 # 统计量
 mean = samples.mean(dim=0)
 std = samples.std(dim=0)

 # 峰度和偏度（用于检测非高斯性）
 kurtosis = ((samples - mean) ** 4).mean() / (std ** 4) - 3
 skewness = ((samples - mean) ** 3).mean() / (std ** 3)

 # KL散度（与标准正态的距离）
 # 简化：使用moment matching估计
 kl_div = 0.5 * (mean.norm()**2 + std.norm()**2 - std.log().sum() - len(mean))

 results.append({
 't': t,
 'mean': mean.numpy(),
 'std': std.numpy(),
 'kurtosis': kurtosis.item(),
 'skewness': skewness.item(),
 'kl_to_normal': kl_div.item()
 })

 return results

# 演示不同SDE的边缘分布演化
def demonstrate_marginal_evolution():
 """演示不同SDE的边缘分布演化"""
 # 初始化不同SDE
 from functools import partial

 # 重新定义简单的SDE类以避免循环引用
 class SimpleVPSDE:
 def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
 self.beta_min = beta_min
 self.beta_max = beta_max
 self.T = T

 def marginal_prob(self, x0, t):
 log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) / self.T - 0.5 * t * self.beta_min
 mean = torch.exp(log_mean_coeff) * x0
 std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
 return mean, std

 class SimpleVESDE:
 def __init__(self, sigma_min=0.01, sigma_max=50.0, T=1.0):
 self.sigma_min = sigma_min
 self.sigma_max = sigma_max
 self.T = T

 def marginal_prob(self, x0, t):
 sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)
 mean = x0
 std = sigma_t
 return mean, std

 # 初始化
 x0 = torch.tensor([1.0, -0.5]) # 2D初始点
 time_points = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

 sdes = {
 'VP-SDE': SimpleVPSDE(),
 'VE-SDE': SimpleVESDE()
 }

 print("边缘分布演化分析")
 print("="*90)

 for sde_name, sde in sdes.items():
 print(f"\n{sde_name}：")
 print("-"*90)
 print(f"{'t':^5} | {'均值范数':^12} | {'标准差':^12} | {'峰度':^10} | {'偏度':^10} | {'KL散度':^12}")
 print("-"*90)

 analyzer = MarginalDistribution(sde)
 results = analyzer.analyze_evolution(x0, time_points)

 for res in results:
 mean_norm = np.linalg.norm(res['mean'])
 std_avg = np.mean(res['std'])

 print(f"{res['t']:5.2f} | {mean_norm:12.6f} | {std_avg:12.6f} | "
 f"{res['kurtosis']:10.4f} | {res['skewness']:10.4f} | {res['kl_to_normal']:12.6f}")

 print("\n关键观察：")
 print("1. VP-SDE: 均值逐渐衰减到零，标准差趋近于1")
 print("2. VE-SDE: 均值保持不变，标准差爆炸式增长")
 print("3. 两者的峰度和偏度都接近零，说明分布接近高斯")

demonstrate_marginal_evolution()



#### Fokker-Planck方程：密度演化的PDE


 边缘分布的演化可以用Fokker-Planck方程（也称为Kolmogorov前向方程）来描述：



 Fokker-Planck方程

对于SDE $dx = f(x,t)dt + g(t)dW_t$，概率密度 $p_t(x)$ 满足：


 $$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (f(x,t)p_t(x)) + \frac{g(t)^2}{2} \Delta p_t(x)$$



其中：



 - $\nabla \cdot$ 是散度算子
 - $\Delta$ 是拉普拉斯算子
 - 第一项是漂移项（传输）
 - 第二项是扩散项（平滑）





#### 分数函数与边缘分布



分数函数 $\nabla \log p_t(x)$ 在扩散模型中扮演着核心角色。它描述了概率密度增长最快的方向。




#### 分数函数的性质



 - **梯度流**：$\nabla \log p_t(x)$ 指向高概率区域
 - **归一化**：不需要知道归一化常数
 - **光滑性**：随着噪声增加，分数函数变得更平滑
 - **可学习性**：可以用神经网络近似





 特殊情况：高斯分布

对于高斯分布 $p(x) = \mathcal{N}(x; \mu, \Sigma)$：


 $$\nabla \log p(x) = -\Sigma^{-1}(x - \mu)$$



这是一个线性函数，指向均值点！





# 分析分数函数随时间的变化
class ScoreFunctionAnalysis:
 """分析分数函数的演化"""

 def __init__(self, sde):
 self.sde = sde

 def analytical_score(self, x, x0, t):
 """计算解析分数函数（仅适用于线性SDE）"""
 if hasattr(self.sde, 'marginal_prob'):
 mean, std = self.sde.marginal_prob(x0, t)
 # 对于高斯分布：score = -(x - mean) / std^2
 score = -(x - mean) / (std ** 2 + 1e-8)
 return score
 else:
 raise NotImplementedError("需要解析边缘分布")

 def score_magnitude_analysis(self, x0, t_values, x_test_points):
 """分析分数函数的幅度"""
 results = []

 for t in t_values:
 scores = []
 for x in x_test_points:
 score = self.analytical_score(x, x0, t)
 scores.append(torch.norm(score).item())

 results.append({
 't': t,
 'mean_magnitude': np.mean(scores),
 'max_magnitude': np.max(scores),
 'min_magnitude': np.min(scores)
 })

 return results

# 演示分数函数的演化
def demonstrate_score_evolution():
 """演示分数函数随时间的变化"""
 # 使用VP-SDE
 class SimpleVPSDE:
 def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
 self.beta_min = beta_min
 self.beta_max = beta_max
 self.T = T

 def marginal_prob(self, x0, t):
 log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) / self.T - 0.5 * t * self.beta_min
 mean = torch.exp(log_mean_coeff) * x0
 std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
 return mean, std

 sde = SimpleVPSDE()
 analyzer = ScoreFunctionAnalysis(sde)

 # 设置
 x0 = torch.tensor([1.0])
 t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
 x_test_points = [torch.tensor([x]) for x in np.linspace(-3, 3, 20)]

 print("\n分数函数幅度分析")
 print("="*60)
 print(f"{'t':^5} | {'平均幅度':^12} | {'最大幅度':^12} | {'最小幅度':^12}")
 print("-"*60)

 results = analyzer.score_magnitude_analysis(x0, t_values, x_test_points)

 for res in results:
 print(f"{res['t']:5.2f} | {res['mean_magnitude']:12.6f} | "
 f"{res['max_magnitude']:12.6f} | {res['min_magnitude']:12.6f}")

 print("\n观察：")
 print("1. 随着t增加，分数函数的幅度逐渐减小")
 print("2. 这反映了分布变得更加平坦（接近均匀分布）")
 print("3. 在噪声很大时，分数函数几乎为零")

demonstrate_score_evolution()



#### 实际应用：训练时的边缘分布




#### 训练时的采样策略

 在训练扩散模型时，我们需要：



 - **采样时间 $t \sim \mathcal{U}[0, T]$**
 - **采样数据点 $x_0 \sim p_{data}$**
 - **根据 $p(x_t|x_0)$ 生成噪声样本 $x_t$**
 - **训练模型预测噪声或分数**




边缘分布的解析形式使得第3步变得非常高效！




 重要性采样

不同的时间点对学习的难度不同。我们可以使用重要性采样：



 $$p(t) \propto \mathbb{E}_{x_0, x_t}[||\nabla_{x_t} \log p(x_t|x_0)||^2]$$



这使得模型更多地关注"难"的时间点。




## 5.3 反向时间SDE：去噪过程



### 5.3.1 Anderson定理



Anderson定理是扩散模型理论的基石之一。它告诉我们，任何满足一定条件的前向SDE都存在一个对应的反向时间SDE，而这正是生成过程的数学基础。



#### 定理的背景



在物理学中，时间反演（time reversal）是一个重要概念。对于确定性系统，时间反演通常很简单。但对于随机过程，情况就复杂得多。Anderson在1982年的工作回答了这个问题。



 Anderson定理（简化版）

考虑前向SDE：


 $$dx = f(x, t) dt + g(t) dW_t, \quad t \in [0, T]$$



定义反向时间 $\tau = T - t$，则存在反向SDE：


 $$dx = [f(x, T-\tau) - g(T-\tau)^2 \nabla_x \log p_{T-\tau}(x)] d\tau + g(T-\tau) d\bar{W}_\tau$$



其中：



 - $\bar{W}_\tau$ 是关于反向时间的布朗运动
 - $p_t(x)$ 是前向过程在时刻 $t$ 的边缘分布
 - $\nabla_x \log p_t(x)$ 是分数函数





#### 直观理解




#### 为什么需要分数函数？


反向过程不仅仅是前向过程的"倒带"。关键差异在于：



 - **信息不对称**：前向过程丢失信息，反向过程需要恢复信息
 - **概率流**：反向过程需要知道"哪里来的概率更高"
 - **分数作为指引**：$\nabla \log p_t(x)$ 正好指向高概率方向





 一个简单的例子：Ornstein-Uhlenbeck过程

考虑前向OU过程：


 $$dx = -\theta x dt + \sigma dW_t$$



其稳态分布为 $\mathcal{N}(0, \frac{\sigma^2}{2\theta})$。分数函数为：


 $$\nabla \log p_{\infty}(x) = -\frac{2\theta}{\sigma^2} x$$



因此反向SDE为：


 $$dx = \left[-\theta x - \sigma^2 \cdot \left(-\frac{2\theta}{\sigma^2} x\right)\right] d\tau + \sigma d\bar{W}_\tau = \theta x d\tau + \sigma d\bar{W}_\tau$$



注意漂移项的符号变了！





# 验证Anderson定理
import torch
import numpy as np

class SDEReversal:
 """验证和演示时间反演SDE"""

 def __init__(self, forward_drift, forward_diffusion, score_fn):
 """
 Args:
 forward_drift: f(x, t) - 前向漂移
 forward_diffusion: g(t) - 前向扩散
 score_fn: \nabla log p_t(x) - 分数函数
 """
 self.f = forward_drift
 self.g = forward_diffusion
 self.score = score_fn

 def reverse_drift(self, x, t, T):
 """计算反向SDE的漂移项"""
 # 时间变换
tau = t
 forward_time = T - tau

 # Anderson公式
 f_reverse = self.f(x, forward_time) - self.g(forward_time)**2 * self.score(x, forward_time)

 return f_reverse

 def simulate_forward_backward(self, x0, T, n_steps=100):
 """模拟前向和反向过程"""
 dt = T / n_steps

 # 前向过程
 forward_path = [x0]
 x = x0.clone()

 for i in range(n_steps):
 t = i * dt
 drift = self.f(x, t) * dt
 diffusion = self.g(t) * np.sqrt(dt) * torch.randn_like(x)
 x = x + drift + diffusion
 forward_path.append(x.clone())

 # 反向过程
 reverse_path = [x]

 for i in range(n_steps):
 tau = i * dt
 drift = self.reverse_drift(x, tau, T) * dt
 diffusion = self.g(T - tau) * np.sqrt(dt) * torch.randn_like(x)
 x = x + drift + diffusion
 reverse_path.append(x.clone())

 return torch.stack(forward_path), torch.stack(reverse_path)

# 示例：VP-SDE的时间反演
def demonstrate_vp_sde_reversal():
 """演示VP-SDE的时间反演"""
 # VP-SDE参数
 beta_min, beta_max = 0.1, 20.0
 T = 1.0

 def beta(t):
 return beta_min + (beta_max - beta_min) * t / T

 def forward_drift(x, t):
 return -0.5 * beta(t) * x

 def forward_diffusion(t):
 return np.sqrt(beta(t))

 def score_fn(x, t):
 # 对于VP-SDE，边缘分布是高斯的
 # p_t(x|x_0) = N(x; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
 # 但我们需要边缘分数 \nabla log p_t(x)
 # 这在实践中是通过神经网络学习的
 # 这里用一个简化的近似
 alpha_bar = np.exp(-0.5 * beta_min * t - 0.25 * (beta_max - beta_min) * t**2 / T)
 return -x / (1 - alpha_bar + 1e-8)

 # 创建反演器
 reverser = SDEReversal(forward_drift, forward_diffusion, score_fn)

 # 模拟
 x0 = torch.randn(2) # 2D初始点
 forward_path, reverse_path = reverser.simulate_forward_backward(x0, T, n_steps=100)

 # 分析结果
 print("时间反演SDE分析")
 print("="*60)
 print(f"初始点: {x0.numpy()}")
 print(f"前向终点: {forward_path[-1].numpy()}")
 print(f"反向终点: {reverse_path[-1].numpy()}")
 print(f"\n前向过程统计:")
 print(f" 初始范数: {torch.norm(forward_path[0]).item():.3f}")
 print(f" 终点范数: {torch.norm(forward_path[-1]).item():.3f}")
 print(f"\n反向过程统计:")
 print(f" 初始范数: {torch.norm(reverse_path[0]).item():.3f}")
 print(f" 终点范数: {torch.norm(reverse_path[-1]).item():.3f}")

 # 路径对比
 forward_norms = [torch.norm(x).item() for x in forward_path]
 reverse_norms = [torch.norm(x).item() for x in reverse_path]

 print(f"\n路径分析:")
 print(f"前向路径范数变化: {forward_norms[0]:.3f} → {forward_norms[-1]:.3f}")
 print(f"反向路径范数变化: {reverse_norms[0]:.3f} → {reverse_norms[-1]:.3f}")
 print("\n注意: 由于随机性，反向过程不会完美回到原点，")
 print("但会回到同样的分布！")

demonstrate_vp_sde_reversal()



#### 数学严格性



 存在性条件
 Anderson定理成立需要以下条件：



 - **正则性**：$f(x,t)$ 和 $g(t)$ 满足Lipschitz条件
 - **非退化性**：$g(t) > 0$ 对所有 $t \in [0,T]$
 - **分数存在**：$\nabla \log p_t(x)$ 存在且满足适当的增长条件





#### 与其他理论的联系




#### 联系与应用



 - **Jarzynski等式**：在非平衡统计物理中的应用
 - **最优传输**：Schrödinger bridge问题的特例
 - **信息论**：与信息熵的增减相关
 - **BSDE**：反向SDE可以看作一类特殊的BSDE





 实践意义

Anderson定理对扩散模型的重要性：



 - **理论保证**：确保了反向过程的存在性
 - **学习目标**：明确了需要学习的是分数函数
 - **采样算法**：提供了从噪声生成数据的数学公式
 - **理论分析**：可以分析生成过程的性质





### 5.3.2 反向SDE的推导



反向SDE是扩散模型的核心，它告诉我们如何从噪声生成数据。这个推导虽然技术性较强，但其物理直觉非常清晰。



#### 时间反演的基本想法




#### 直觉：电影倒放


想象你录制了墨水在水中扩散的过程：



 - **正向播放**：墨水从一点扩散到整个水体
 - **反向播放**：分散的墨水神奇地聚集回一点



反向SDE就是找到这个"倒放过程"的数学描述。




#### Anderson定理的应用



根据Anderson定理，如果前向过程是：



 $$dx = f(x, t)dt + g(t)dw$$



那么反向过程（时间从T到0）是：



 $$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{w}$$



其中$\bar{w}$是反向布朗运动。



#### 关键洞察：分数函数的作用



 为什么需要分数函数？

比较前向和反向SDE，唯一的区别是多了一项：$-g(t)^2 \nabla_x \log p_t(x)$


这项的作用是：



 - **补偿扩散**：抵消随机项带来的扩散效应
 - **指引方向**：指向概率密度增加的方向
 - **时变修正**：随时间调整"拉回"的强度





#### 推导步骤（简化版）



 关键步骤


 - **考虑联合分布**：$(x_t, t)$的演化
 - **应用Fokker-Planck方程**：得到$p_t(x)$的演化
 - **时间反演**：令$\tau = T - t$
 - **匹配系数**：使反向过程的FP方程与原方程一致





#### 具体例子：VP-SDE的反向过程



对于VP-SDE（方差保持）：



 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$$



其反向SDE是：



 $$dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right]dt + \sqrt{\beta(t)}d\bar{w}$$




# 验证反向SDE的正确性
import torch
import torch.nn as nn

class ReverseSDE:
 """反向SDE的数值验证"""

 def __init__(self, beta_schedule):
 """
 Args:
 beta_schedule: 函数，返回时刻t的beta(t)
 """
 self.beta = beta_schedule

 def forward_marginal(self, x0, t):
 """计算前向过程的边缘分布 p(x_t|x_0)"""
 # 对于VP-SDE，有解析解
 alpha_bar = torch.exp(-0.5 * self.integral_beta(t))
 mean = alpha_bar * x0
 var = 1 - alpha_bar**2
 return mean, var

 def integral_beta(self, t):
 """计算 ∫_0^t beta(s) ds"""
 # 简单起见，假设beta(t) = beta_min + t*(beta_max - beta_min)
 beta_min, beta_max = 0.1, 20.0
 return beta_min * t + 0.5 * (beta_max - beta_min) * t**2

 def score_function(self, xt, x0, t):
 """计算真实的分数函数（用于验证）"""
 mean, var = self.forward_marginal(x0, t)
 score = -(xt - mean) / var
 return score

 def verify_reverse_sde(self, x0, T=1.0, dt=0.01):
 """验证反向SDE确实能恢复x0"""
 print("验证反向SDE")
 print("="*50)

 # 1. 前向过程：x0 -> xT
 t = 0
 x = x0.clone()
 trajectory_forward = [x.clone()]

 while t  xT = {xT.numpy():.3f}")

 # 2. 反向过程：xT -> x0
 t = T
 x = xT.clone()
 trajectory_reverse = [x.clone()]

 while t > dt:
 beta_t = self.beta(t)
 # 使用真实分数（实际中需要学习）
 score = self.score_function(x, x0, t)

 # 反向SDE
 drift = -0.5 * beta_t * x - beta_t * score
 diffusion = torch.sqrt(beta_t * dt) * torch.randn_like(x)
 x = x + drift * dt + diffusion
 t -= dt
 trajectory_reverse.append(x.clone())

 x0_recovered = x
 print(f"反向过程完成: xT = {xT.numpy():.3f} -> x0_recovered = {x0_recovered.numpy():.3f}")
 print(f"恢复误差: {torch.abs(x0 - x0_recovered).item():.4f}")

 # 3. 分析轨迹
 forward_std = torch.stack(trajectory_forward).std()
 reverse_std = torch.stack(trajectory_reverse).std()
 print(f"\n轨迹分析:")
 print(f"前向轨迹标准差: {forward_std:.3f} (扩散)")
 print(f"反向轨迹标准差: {reverse_std:.3f} (聚集)")

 return trajectory_forward, trajectory_reverse

# 测试
def beta_schedule(t, beta_min=0.1, beta_max=20.0):
 """线性beta调度"""
 return beta_min + t * (beta_max - beta_min)

reverse_sde = ReverseSDE(beta_schedule)
x0 = torch.tensor([2.0])
reverse_sde.verify_reverse_sde(x0)



#### 物理类比：势能场中的粒子




#### 分数函数作为"力"

 反向SDE可以理解为粒子在势能场中的运动：



 - **原始漂移项** $f(x,t)$：外加的确定性力
 - **分数修正项** $-g^2\nabla\log p$：势能场的梯度力
 - **随机项** $g(t)d\bar{w}$：热运动



分数函数创造了一个"势能井"，将粒子拉向数据分布的高概率区域。




#### 实用考虑



 实现反向SDE的挑战


 - **分数估计**：需要神经网络学习$\nabla\log p_t(x)$
 - **数值稳定性**：小的$dt$带来大的计算成本
 - **边界条件**：$t=0$附近需要特殊处理
 - **随机性控制**：如何平衡确定性和随机性






# 实用的反向采样器
class PracticalReverseSDESampler:
 """实际使用的反向SDE采样器"""

 def __init__(self, score_model, beta_schedule):
 self.score_model = score_model
 self.beta = beta_schedule

 def sample(self, shape, T=1.0, dt=0.01, device='cpu'):
 """从噪声生成样本"""
 # 初始化为高斯噪声
 x = torch.randn(shape, device=device)

 # 时间步
 timesteps = torch.linspace(T, dt, int(T/dt), device=device)

 for t in timesteps:
 # 估计分数
 with torch.no_grad():
 score = self.score_model(x, t)

 # 计算系数
 beta_t = self.beta(t)
 drift_coeff = -0.5 * beta_t
 score_coeff = -beta_t
 noise_coeff = torch.sqrt(beta_t * dt)

 # 反向SDE更新
 drift = drift_coeff * x + score_coeff * score
 noise = torch.randn_like(x)

 x = x + drift * dt + noise_coeff * noise

 # 可选：动态调整步长或使用高阶求解器
 if t 



#### 总结：反向SDE的意义



 核心要点


 - 反向SDE = 前向SDE + 分数修正
 - 分数函数是连接前向和反向过程的桥梁
 - 物理直觉：从无序到有序需要"信息注入"
 - 计算挑战：准确估计分数函数是关键





### 5.3.3 分数函数的作用


 分数函数 $\nabla_x \log p_t(x)$ 是连续时间扩散模型的核心。它不仅是数学上的必需品，更有深刻的几何和物理意义。



#### 几何意义：指向高概率区域




#### 梯度上升的视角


分数函数指向概率密度增加最快的方向：



 - 在低概率区域，它指向高概率区域
 - 在概率峰值附近，它的幅度较小
 - 它定义了概率景观上的"最陡上升路径"





#### 动力学意义：时变的引导



分数函数在不同时刻扮演不同角色：



 时间演化的三个阶段


 - **早期（$t \approx 0$）**：


 数据分布还很清晰
 - 分数函数提供精确的局部结构信息
 - 主要作用：保持数据的精细特征


 
 - **中期（$0 数据结构部分模糊
 - 分数函数引导全局结构的形成
 - 主要作用：建立大尺度的模式


 
 - **后期（$t \approx T$）**：


 接近纯噪声分布
 - 分数函数提供初始的方向指引
 - 主要作用：从噪声中"点燃"生成过程


 





#### 信息论意义：负熵流



 分数函数与信息

从信息论角度看，分数函数代表了"信息梯度"：



 - **前向过程**：信息逐渐丢失，熵增加
 - **反向过程**：分数函数注入信息，熵减少
 - **平衡点**：分数函数恰好补偿扩散造成的信息损失





#### 与其他概念的联系




#### 1. 与Stein分数的关系


分数函数满足Stein恒等式：


 $$\mathbb{E}_{p_t}[\nabla_x \log p_t(x)] = 0$$


这保证了分数函数的"平衡性"——它不会整体偏向某个方向。





#### 2. 与最优传输的关系


在某些条件下，分数函数定义了从$p_t$到$p_0$的最优传输映射的速度场。这建立了扩散模型与最优传输理论的桥梁。





#### 3. 与能量模型的关系


如果定义能量函数$E_t(x) = -\log p_t(x)$，则：


 $$\nabla_x \log p_t(x) = -\nabla_x E_t(x)$$


分数函数就是能量函数的负梯度，指向能量下降的方向。




#### 实践中的重要性



 为什么学习分数函数？


 - **参数化简单**：


 不需要归一化常数
 - 可以用标准神经网络表示
 - 训练目标明确（去噪任务）


 
 - **局部性质**：


 只需要局部信息
 - 不需要全局积分
 - 计算效率高


 
 - **稳定性好**：


 梯度匹配是稳定的优化问题
 - 避免了密度估计的数值问题
 - 适合高维数据


 





#### 分数函数的多尺度特性



 跨尺度的信息编码

分数函数在不同噪声水平下编码不同尺度的信息：



 - **低噪声**：编码精细纹理、边缘等高频信息
 - **中等噪声**：编码物体形状、整体结构
 - **高噪声**：编码全局布局、大尺度模式



这种多尺度特性使得扩散模型能够生成具有丰富细节的高质量样本。




#### 总结：分数函数的核心地位



 关键认识

分数函数是扩散模型的"灵魂"：



 - 它编码了数据分布的所有信息
 - 它连接了前向扩散和反向生成
 - 它统一了多个理论视角（SDE、ODE、能量模型）
 - 它提供了实用的参数化和训练方法



理解分数函数就是理解扩散模型的关键。




## 5.4 概率流ODE：确定性的替代



### 5.4.1 从SDE到ODE



一个令人惊讶的发现是：每个SDE都有一个对应的ODE，它们产生相同的边缘分布演化。这个ODE被称为概率流ODE（Probability Flow ODE）。



#### 核心思想：去除随机性




#### 从随机到确定


SDE包含两部分：



 - **确定性漂移**：$f(x,t)dt$
 - **随机扩散**：$g(t)dw$



概率流ODE通过修改漂移项来补偿随机项的效果，使得整体演化变成确定性的。




#### 概率流ODE的形式



对于前向SDE：


 $$dx = f(x,t)dt + g(t)dw$$



对应的概率流ODE是：


 $$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2\nabla_x \log p_t(x)$$



 关键性质


 - ODE是确定性的——给定初始条件，轨迹唯一确定
 - 边缘分布相同——$p_t(x)$的演化与SDE一致
 - 可逆性——可以精确地前向和反向求解





#### 直观理解：流场视角



 流体动力学类比

可以把概率流ODE理解为不可压缩流体的流动：



 - **流体元素**：概率质量的小块
 - **速度场**：$v(x,t) = f(x,t) - \frac{1}{2}g(t)^2\nabla\log p_t$
 - **流线**：ODE的解轨迹
 - **守恒律**：概率总量守恒





#### 为什么这个ODE有效？



概率流ODE的设计基于以下观察：




#### Fokker-Planck方程的分解


SDE对应的Fokker-Planck方程可以写成：


 $$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t v_{total})$$


其中总速度场：


 $$v_{total} = \underbrace{f(x,t)}_{\text{漂移}} - \underbrace{\frac{1}{2}g(t)^2\nabla\log p_t}_{\text{扩散修正}}$$




#### SDE vs ODE：轨迹的差异



 两种演化方式的对比
 
 
 特性
 SDE
 概率流ODE
 
 
 轨迹性质
 随机、不可预测
 确定性、可预测
 
 
 计算复杂度
 需要多次采样
 单次求解即可
 
 
 可逆性
 统计意义上可逆
 精确可逆
 
 
 适用场景
 生成多样性样本
 图像编辑、插值
 
 



#### 概率流ODE的优势



 为什么使用ODE？


 - **精确编码**：


 可以将数据精确编码为潜在表示
 - 支持语义操作和编辑


 
 - **数值稳定性**：


 可以使用高阶ODE求解器
 - 自适应步长控制


 
 - **理论分析**：


 更容易分析收敛性
 - 可以研究流形结构


 





#### 与神经ODE的联系




#### 统一框架


概率流ODE可以看作一种特殊的神经ODE：



 - 状态：数据点 $x$
 - 时间：噪声水平 $t$
 - 动力学：由分数函数参数化



这建立了扩散模型与连续深度模型之间的桥梁。




### 5.4.2 概率流的性质



概率流ODE具有一系列优美的数学性质，这些性质使它成为理解和应用扩散模型的重要工具。



#### 1. 体积保持性（Liouville定理）



 概率质量守恒

概率流保持相空间的体积元：


 $$\nabla \cdot v(x,t) = 0$$


其中 $v(x,t) = f(x,t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)$ 是速度场。


这意味着：



 - 流动是不可压缩的
 - 局部概率密度沿流线保持不变
 - 拓扑性质得以保持





#### 2. 双射性与可逆性




#### 一一对应关系


概率流ODE建立了以下双射：



 - $\phi_t: \mathcal{X}_0 \rightarrow \mathcal{X}_t$（前向流）
 - $\phi_t^{-1}: \mathcal{X}_t \rightarrow \mathcal{X}_0$（反向流）



每个数据点 $x_0$ 对应唯一的噪声表示 $x_T$，反之亦然。




#### 3. 最优传输视角



 动态最优传输

在某些条件下，概率流ODE给出了从 $p_0$ 到 $p_T$ 的最优传输路径：



 - **路径最短**：在适当的度量下，流线是测地线
 - **能量最优**：最小化传输成本
 - **保持结构**：相邻点保持相邻





#### 4. 连续性与正则性



 光滑演化

如果分数函数 $\nabla\log p_t(x)$ 满足适当的正则性条件，则：



 - **解的存在唯一性**：给定初值，ODE有唯一解
 - **连续依赖性**：解连续依赖于初始条件
 - **时间可逆性**：可以精确地前向和后向求解





#### 5. 与Wasserstein梯度流的关系




#### 能量泛函的梯度流


概率流ODE可以理解为某个能量泛函的Wasserstein梯度流：


 $$\frac{\partial p_t}{\partial t} = \nabla \cdot \left(p_t \nabla \frac{\delta \mathcal{F}[p_t]}{\delta p_t}\right)$$


其中 $\mathcal{F}[p]$ 是适当选择的能量泛函。




#### 6. 信息几何性质



 Fisher信息的演化

沿着概率流，Fisher信息矩阵的演化遵循特定规律：



 - **信息损失**：前向流中Fisher信息单调递减
 - **度量保持**：某些几何结构得以保持
 - **自然梯度**：流动方向与自然梯度相关





#### 7. 数值性质



 计算优势
 
 
 性质
 含义
 
 
 Lipschitz连续性
 数值稳定，可用标准ODE求解器
 
 
 自适应步长
 可根据局部误差调整步长
 
 
 高阶方法适用
 Runge-Kutta等方法有效
 
 
 并行化友好
 批量样本可并行处理
 
 



#### 8. 语义插值性质




#### 平滑的语义过渡


概率流ODE的轨迹提供了自然的插值路径：



 - 两个数据点之间的插值通过其噪声表示的线性插值实现
 - 插值路径反映了数据流形的几何结构
 - 中间状态保持语义连贯性





#### 应用价值



 实际应用中的重要性


 - **精确反演**：可以精确重构原始数据
 - **潜在空间操作**：在噪声空间进行语义编辑
 - **概率估计**：通过变换公式计算似然
 - **轨迹分析**：研究生成过程的动力学





### 5.4.3 ODE vs SDE：权衡与选择



在实际应用中，选择使用SDE还是概率流ODE需要考虑多个因素。每种方法都有其优势和局限性。



#### 生成质量对比



 质量-多样性权衡
 
 
 方面
 SDE
 ODE
 
 
 **样本质量**
 通常更高，随机性有助于避免局部缺陷
 可能陷入次优路径
 
 
 **多样性**
 自然产生多样化样本
 确定性导致多样性受限
 
 
 **模式覆盖**
 更好地覆盖所有模式
 可能错过某些模式
 
 
 **细节保真度**
 随机噪声可能模糊细节
 精确轨迹保持细节
 
 



#### 计算效率分析




#### 速度与精度的平衡


**SDE的计算特点：**



 - 需要固定的小步长（通常1000步）
 - 每步需要生成随机噪声
 - 难以使用自适应步长
 - 并行化效率高



**ODE的计算特点：**



 - 可使用高阶求解器（如RK45）
 - 自适应步长大幅减少NFE
 - 通常只需100-200次函数评估
 - 数值误差可控





#### 应用场景适配



 选择指南

**适合使用SDE的场景：**



 - **纯生成任务**：需要高质量、多样化的样本
 - **数据增强**：随机性带来的变化是优势
 - **对抗鲁棒性**：随机性增强模型鲁棒性
 - **探索性应用**：需要发现新的样本模式



**适合使用ODE的场景：**



 - **图像编辑**：需要精确的编码-解码
 - **插值任务**：生成中间过渡状态
 - **反演重构**：从噪声恢复原始输入
 - **可解释性研究**：分析生成轨迹





#### 混合策略



 结合两者优势

实践中常用的混合策略：



 - **分段切换**：


 早期阶段（高噪声）使用ODE快速去噪
 - 后期阶段（低噪声）使用SDE精细化


 
 - **温度调节**：


 引入温度参数 $\tau$ 控制随机性
 - $dx = f dt + \tau \cdot g dw$
 - $\tau=0$ 退化为ODE，$\tau=1$ 为标准SDE


 
 - **条件切换**：


 根据当前状态的置信度动态选择
 - 高置信区域用ODE，低置信区域用SDE


 





#### 数值稳定性考虑




#### 数值挑战与解决方案


**SDE的数值挑战：**



 - 步长过大导致数值爆炸
 - 累积误差难以控制
 - 需要仔细选择离散化方案



**ODE的数值挑战：**



 - 刚性问题需要隐式求解器
 - 分数函数的数值误差会累积
 - 需要监控局部截断误差





#### 理论保证对比



 收敛性分析
 
 
 理论性质
 SDE
 ODE
 
 
 收敛阶
 弱收敛 O(√dt)
 可达高阶 O(dt^p)
 
 
 误差界
 概率意义上的界
 确定性误差界
 
 
 长时间行为
 遍历性保证
 轨迹稳定性
 
 



#### 实践建议



 最佳实践总结


 - **默认选择**：对于大多数生成任务，SDE仍是首选
 - **速度优先**：当推理速度关键时，考虑ODE
 - **精度要求**：需要精确控制时使用ODE
 - **实验验证**：具体选择应基于实际效果
 - **混合使用**：不同阶段可以使用不同方法





## 5.5 Fokker-Planck方程：密度视角



### 5.5.1 从粒子到密度



Fokker-Planck方程提供了扩散过程的另一个视角：从跟踪单个粒子转向描述整体概率密度的演化。这是理解扩散模型的关键数学工具。



#### 两种描述方式的对偶性




#### 微观 vs 宏观

 
 
 SDE（微观）
 Fokker-Planck（宏观）
 
 
 描述单个粒子的随机轨迹
 描述概率密度的确定性演化
 
 
 $dx_t = f(x_t,t)dt + g(t)dw_t$
 $\frac{\partial p}{\partial t} = -\nabla \cdot (fp) + \frac{g^2}{2}\Delta p$
 
 
 随机微分方程
 偏微分方程
 
 



#### Fokker-Planck方程的推导直觉



 守恒律视角

Fokker-Planck方程本质上是概率的守恒律：


 $$\frac{\partial p}{\partial t} + \nabla \cdot J = 0$$


其中概率流 $J$ 包含两部分：



 - **漂移流**：$J_{drift} = f(x,t)p(x,t)$
 - **扩散流**：$J_{diff} = -\frac{g(t)^2}{2}\nabla p(x,t)$





#### 标准形式与物理意义



对于一般的SDE，Fokker-Planck方程为：



 $$\frac{\partial p(x,t)}{\partial t} = -\sum_i \frac{\partial}{\partial x_i}[f_i(x,t)p(x,t)] + \frac{1}{2}\sum_{i,j}\frac{\partial^2}{\partial x_i \partial x_j}[g_{ij}(t)p(x,t)]$$



 各项的物理解释


 - **时间导数项** $\frac{\partial p}{\partial t}$：密度的局部变化率
 - **对流项** $-\nabla \cdot (fp)$：由确定性漂移引起的概率流动
 - **扩散项** $\frac{g^2}{2}\Delta p$：由随机涨落引起的概率扩散





#### 特殊情况：线性Fokker-Planck方程



对于扩散模型中常见的线性SDE：


 $$dx = -\frac{\beta(t)}{2}x dt + \sqrt{\beta(t)}dw$$



对应的Fokker-Planck方程是：


 $$\frac{\partial p}{\partial t} = \frac{\beta(t)}{2}\nabla \cdot (xp) + \frac{\beta(t)}{2}\Delta p$$




#### 解的高斯性


线性Fokker-Planck方程的一个重要性质是：如果初始分布是高斯的，那么任意时刻的分布都保持高斯形式。这解释了为什么扩散模型的前向过程最终收敛到高斯分布。




#### 与热方程的联系



 扩散作为热传导

在纯扩散情况下（$f=0$），Fokker-Planck方程退化为热方程：


 $$\frac{\partial p}{\partial t} = D\Delta p$$


这建立了以下类比：



 - 概率密度 ↔ 温度分布
 - 扩散系数 ↔ 热导率
 - 概率流 ↔ 热流





#### 稳态与平衡分布



 长时间行为

当 $t \to \infty$ 时，系统趋向稳态：$\frac{\partial p}{\partial t} = 0$


稳态分布 $p_\infty(x)$ 满足：


 $$\nabla \cdot (f p_\infty) = \frac{g^2}{2}\Delta p_\infty$$


对于扩散模型，这通常是标准高斯分布 $\mathcal{N}(0,I)$。




#### 边界条件的重要性




#### 自然边界条件


在 $\mathbb{R}^d$ 上，通常采用自然边界条件：



 - $p(x,t) \to 0$ 当 $|x| \to \infty$
 - $\int_{\mathbb{R}^d} p(x,t)dx = 1$（概率守恒）



这些条件确保了物理意义和数学良定性。




#### 数值求解的挑战



 维数灾难

直接求解Fokker-Planck方程面临严重的维数灾难：



 - 对于 $d$ 维问题，计算复杂度为 $O(N^d)$
 - 存储需求随维数指数增长
 - 高维空间中的数值格式不稳定



这就是为什么扩散模型选择通过学习分数函数来间接求解。




### 5.5.2 Fokker-Planck方程的推导



Fokker-Planck方程的推导展示了随机过程与偏微分方程之间的深刻联系。这里我们从直观到严格，逐步推导这个方程。



#### 方法一：从Chapman-Kolmogorov方程出发



 基本思路

考虑转移概率密度 $p(x,t|x_0,t_0)$，它满足Chapman-Kolmogorov方程：


 $$p(x,t+\Delta t|x_0,t_0) = \int p(x,t+\Delta t|y,t)p(y,t|x_0,t_0)dy$$


对小时间步 $\Delta t$，展开转移核并取极限即可得到Fokker-Planck方程。




#### 方法二：Itô公式方法（更直观）



这是理解Fokker-Planck方程的现代方法：




#### 核心步骤



 - **考虑测试函数**：对任意光滑函数 $\phi(x)$，计算期望值的演化
 - **应用Itô公式**：

 $$d\phi(x_t) = \nabla\phi \cdot dx_t + \frac{1}{2}\text{Tr}(\nabla^2\phi \cdot d\langle x\rangle_t)$$


 - **取期望**：利用 $\mathbb{E}[dw_t] = 0$
 - **分部积分**：将作用在 $\phi$ 上的算子转移到 $p$ 上





#### 详细推导：一维情况



 从SDE到Fokker-Planck

设一维SDE为：$dx_t = f(x_t,t)dt + g(t)dw_t$



**步骤1**：对测试函数 $\phi(x)$ 应用Itô公式


 $$d\phi(x_t) = \phi'(x_t)dx_t + \frac{1}{2}\phi''(x_t)(dx_t)^2$$



**步骤2**：计算二次变分 $(dx_t)^2 = g(t)^2dt$



**步骤3**：代入并取期望


 $$\frac{d}{dt}\mathbb{E}[\phi(x_t)] = \mathbb{E}[f(x_t,t)\phi'(x_t) + \frac{g(t)^2}{2}\phi''(x_t)]$$



**步骤4**：用密度函数表示期望


 $$\frac{d}{dt}\int \phi(x)p(x,t)dx = \int \left[f(x,t)\phi'(x) + \frac{g(t)^2}{2}\phi''(x)\right]p(x,t)dx$$



**步骤5**：分部积分


 $$\int \phi(x)\frac{\partial p}{\partial t}dx = \int \phi(x)\left[-\frac{\partial}{\partial x}(f p) + \frac{g^2}{2}\frac{\partial^2 p}{\partial x^2}\right]dx$$



由于 $\phi$ 任意，得到Fokker-Planck方程。




#### 高维推广



 多维Fokker-Planck方程

对于 $d$ 维SDE：$dx_i = f_i(x,t)dt + \sum_j g_{ij}(t)dw_j$


Fokker-Planck方程为：


 $$\frac{\partial p}{\partial t} = -\sum_i \frac{\partial}{\partial x_i}(f_i p) + \frac{1}{2}\sum_{i,j}\frac{\partial^2}{\partial x_i \partial x_j}(D_{ij}p)$$


其中扩散矩阵 $D_{ij} = \sum_k g_{ik}g_{jk}$。




#### 反向Fokker-Planck方程




#### 时间反演


对于反向SDE，对应的Fokker-Planck方程（也称为Kolmogorov反向方程）是：


 $$-\frac{\partial p}{\partial t} = f \cdot \nabla p + \frac{g^2}{2}\Delta p$$


注意时间导数的符号变化，这反映了时间反演的本质。




#### 与分数函数的关系



 分数函数的出现

Fokker-Planck方程可以重写为：


 $$\frac{\partial p}{\partial t} = -\nabla \cdot \left[p\left(f - \frac{g^2}{2}\nabla \log p\right)\right] - \frac{g^2}{2}\Delta p$$


这里自然出现了分数函数 $\nabla \log p$，暗示了它在扩散过程中的核心作用。




#### 物理解释：概率流的分解



 流的物理图像

Fokker-Planck方程描述了两种概率流的竞争：



 - **确定性流**：由漂移 $f(x,t)$ 驱动，可以聚集或分散概率
 - **扩散流**：总是使概率分散，趋向均匀分布



扩散模型巧妙地平衡这两种流，实现数据与噪声之间的可逆转换。




#### 数学性质




#### 重要性质



 - **线性性**：Fokker-Planck方程对 $p$ 是线性的
 - **保正性**：如果初值 $p_0 \geq 0$，则 $p_t \geq 0$ 对所有 $t$ 成立
 - **质量守恒**：$\int p(x,t)dx = 1$ 对所有 $t$ 成立
 - **最大值原理**：密度的最大值不会增加（纯扩散情况）





### 5.5.3 与分数函数的联系



Fokker-Planck方程与分数函数之间存在深刻的联系。这种联系不仅是数学上的巧合，更揭示了扩散模型的核心机制。



#### 分数函数在Fokker-Planck方程中的出现



 概率流的分解

Fokker-Planck方程可以写成流的形式：


 $$\frac{\partial p}{\partial t} = -\nabla \cdot J$$


其中总概率流 $J$ 可以分解为：


 $$J = \underbrace{fp}_{\text{漂移流}} - \underbrace{\frac{g^2}{2}\nabla p}_{\text{扩散流}} = p\left(f - \frac{g^2}{2}\nabla \log p\right)$$


这里自然出现了分数函数 $\nabla \log p$。




#### 分数函数的物理意义




#### 三种解释



 - **热力学力**：分数函数代表将系统推向平衡态的"力"
 - **信息梯度**：指向信息含量增加的方向
 - **最可能路径**：在给定约束下的最可能演化方向





#### 稳态条件与分数函数



 平衡态的特征

在稳态（$\frac{\partial p}{\partial t} = 0$）时，概率流必须为零：


 $$J = p\left(f - \frac{g^2}{2}\nabla \log p\right) = 0$$


这给出稳态条件：


 $$f(x) = \frac{g^2}{2}\nabla \log p_{\infty}(x)$$


即漂移必须恰好平衡扩散引起的概率流出。




#### 分数函数与熵产生



 熵的演化

相对熵（KL散度）的时间导数为：


 $$\frac{d}{dt}D_{KL}(p_t \| p_{\infty}) = -\int p_t |\nabla \log p_t - \nabla \log p_{\infty}|^2 dx \leq 0$$


这表明：



 - 分数函数的差异驱动系统向平衡态演化
 - 演化速率正比于分数函数差的平方
 - 当且仅当 $p_t = p_{\infty}$ 时演化停止





#### 反向过程中分数函数的作用




#### 时间反演的关键


考虑前向Fokker-Planck方程：


 $$\frac{\partial p}{\partial t} = -\nabla \cdot (fp) + \frac{g^2}{2}\Delta p$$


对应的反向Fokker-Planck方程需要额外的分数项：


 $$\frac{\partial p}{\partial \tau} = -\nabla \cdot \left[\left(-f + g^2\nabla \log p\right)p\right] + \frac{g^2}{2}\Delta p$$


分数函数 $\nabla \log p$ 提供了反向演化所需的"信息"。




#### 分数匹配与Fokker-Planck方程



 学习目标的等价性

扩散模型的训练可以从两个角度理解：



 - **分数匹配**：最小化 $\mathbb{E}_{p_t}[|\nabla \log p_t - s_\theta|^2]$
 - **密度演化**：使神经网络参数化的流满足Fokker-Planck方程



这两个目标在数学上是等价的。




#### 变分视角：最小作用量原理



 Onsager-Machlup泛函

扩散过程的路径概率可以用作用量表示：


 $$S[x] = \int_0^T \left[\frac{|\dot{x} - f|^2}{2g^2} - \frac{1}{2}\nabla \cdot f\right]dt$$


分数函数通过Fokker-Planck方程进入这个作用量，决定了最可能的演化路径。




#### 计算优势




#### 为什么学习分数而非密度



 - **局部性**：分数函数只需要局部信息，而密度需要全局归一化
 - **维数可扩展**：分数匹配避免了高维空间的积分
 - **数值稳定**：梯度估计比密度估计更稳定
 - **训练简单**：去噪任务提供了自然的训练信号





#### 总结：分数函数的中心地位



 统一视角

分数函数 $\nabla \log p$ 是连接多个概念的桥梁：



 - **SDE视角**：使反向过程成为可能
 - **PDE视角**：出现在Fokker-Planck方程中
 - **ODE视角**：定义概率流的速度场
 - **优化视角**：提供了可学习的参数化
 - **几何视角**：是概率流形上的切向量



理解这些联系是掌握连续时间扩散模型的关键。




## 5.6 统一框架：Score SDE



### 5.6.1 VP-SDE、VE-SDE和sub-VP-SDE



Score SDE框架统一了各种扩散模型，将它们表示为不同的SDE选择。这里介绍三种主要的SDE类型及其特点。



#### VP-SDE（Variance Preserving）



 方差保持SDE

VP-SDE对应于DDPM，其形式为：


 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$$


其中 $\beta(t)$ 是噪声调度函数。


**关键性质**：



 - 保持信号和噪声的总方差近似为1
 - 边缘分布：$p(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
 - 终态分布：$p(x_T) \approx \mathcal{N}(0, I)$





#### VE-SDE（Variance Exploding）



 方差爆炸SDE

VE-SDE对应于NCSN/SMLD，其形式为：


 $$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}dw$$


其中 $\sigma(t)$ 是递增的噪声水平函数。


**关键性质**：



 - 没有漂移项，纯扩散过程
 - 边缘分布：$p(x_t|x_0) = \mathcal{N}(x_0, \sigma^2(t)I)$
 - 方差随时间单调增加至无穷





#### sub-VP-SDE



 次方差保持SDE

sub-VP-SDE是VP-SDE的连续时间极限：


 $$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}dw$$


**关键性质**：



 - 更精确地保持离散DDPM的边缘分布
 - 扩散系数依赖于历史
 - 数值稳定性更好





#### 三种SDE的比较



 特性对比
 
 
 特性
 VP-SDE
 VE-SDE
 sub-VP-SDE
 
 
 漂移项
 线性
 无
 线性
 
 
 扩散系数
 $\sqrt{\beta(t)}$
 $\sqrt{\dot{\sigma}^2(t)}$
 状态依赖
 
 
 终态分布
 $\mathcal{N}(0,I)$
 $\mathcal{N}(0,\sigma_T^2 I)$
 $\mathcal{N}(0,I)$
 
 
 计算效率
 高
 最高
 中等
 
 



#### 噪声调度的选择




#### 常用的噪声调度


**VP-SDE的β调度**：



 - 线性：$\beta(t) = \beta_{min} + t(\beta_{max} - \beta_{min})$
 - 余弦：$\beta(t) = \beta_{max} \cdot \sin^2(\frac{\pi t}{2T})$
 - 改进线性：考虑信噪比的平滑变化



**VE-SDE的σ调度**：



 - 几何级数：$\sigma_i = \sigma_{min} \cdot (\sigma_{max}/\sigma_{min})^{i/N}$
 - 多项式：$\sigma(t) = \sigma_{min} + t^p(\sigma_{max} - \sigma_{min})$





#### 选择指南



 何时使用哪种SDE


 - **VP-SDE**：


 标准选择，适合大多数任务
 - 训练稳定，理论完善
 - 与DDPM兼容


 
 - **VE-SDE**：


 需要精确控制噪声水平
 - 多尺度建模
 - 计算效率要求高


 
 - **sub-VP-SDE**：


 需要精确匹配离散模型
 - 理论研究
 - 高精度要求


 





#### 统一视角的优势



 Score SDE框架的贡献


 - **理论统一**：不同方法只是SDE的不同选择
 - **灵活设计**：可以设计新的SDE形式
 - **统一训练**：相同的分数匹配目标
 - **统一采样**：通用的求解器（如PC采样器）





### 5.6.2 离散模型作为SDE的特例



一个深刻的洞察是：所有离散时间的扩散模型都可以看作连续SDE的数值离散化。这个视角不仅统一了理论，还指导了算法改进。



#### DDPM与VP-SDE的对应关系



 从离散到连续

DDPM的前向过程：


 $$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t$$


当时间步 $\Delta t \to 0$ 时，设 $\alpha_t = 1 - \beta_t\Delta t$，可得：


 $$\frac{x_t - x_{t-1}}{\Delta t} \approx -\frac{\beta_t}{2}x_t + \frac{\sqrt{\beta_t}}{\sqrt{\Delta t}}\epsilon_t$$


这正是VP-SDE的Euler-Maruyama离散化。




#### NCSN与VE-SDE的对应关系




#### 多尺度噪声的连续化


NCSN使用离散的噪声水平 $\{\sigma_i\}_{i=1}^L$，每个水平对应一个加噪分布：


 $$p_{\sigma_i}(x|x_0) = \mathcal{N}(x_0, \sigma_i^2 I)$$


当 $L \to \infty$ 且噪声水平连续化时，得到VE-SDE：


 $$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}dw$$




#### 离散化误差分析



 数值格式的影响
 
 
 离散化方法
 局部误差
 特点
 
 
 Euler-Maruyama
 $O(\Delta t)$
 简单但精度低
 
 
 Heun方法
 $O(\Delta t^2)$
 需要两次函数评估
 
 
 随机Runge-Kutta
 $O(\Delta t^{3/2})$
 高精度但复杂
 
 



#### 时间步的选择



 离散步数与连续时间

离散模型的步数 $T$ 与连续时间的关系：



 - **DDPM**：通常 $T=1000$，对应连续时间 $[0,1]$
 - **改进DDPM**：$T=4000$，更好地逼近连续极限
 - **连续模型**：$T \to \infty$，完全连续



步数越多，离散模型越接近连续SDE，但计算成本也越高。




#### 训练目标的统一




#### 分数匹配的一致性


无论是离散还是连续模型，训练目标都是分数匹配：



 - **DDPM**：$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$
 - **Score SDE**：$\mathcal{L} = \mathbb{E}_{t,x_0,x_t}[\|s_\theta(x_t, t) - \nabla \log p_{t|0}(x_t|x_0)\|^2]$



两者通过关系 $\epsilon = -\sigma_t \nabla \log p_{t|0}$ 联系起来。




#### 采样算法的继承



 从离散到连续的算法迁移


 - **DDPM采样** → **SDE求解器**：


 祖先采样 → Euler-Maruyama方法
 - DDIM → 概率流ODE


 
 - **加速技巧**：


 步长调整在连续框架下更自然
 - 自适应求解器可以自动选择步长


 





#### 连续视角的优势



 为什么采用连续框架？


 - **理论优势**：


 更清晰的数学结构
 - 丰富的SDE/PDE理论可用
 - 更容易分析收敛性


 
 - **算法优势**：


 可以使用成熟的ODE/SDE求解器
 - 自适应时间步长
 - 高阶数值方法


 
 - **灵活性**：


 容易设计新的SDE
 - 可以在不同SDE之间转换
 - 统一的训练和采样框架


 





#### 实践指南




#### 何时使用离散vs连续



 - **使用离散模型**：


 已有成熟的离散实现
 - 固定步数的应用场景
 - 需要与现有DDPM代码兼容


 
 - **使用连续模型**：


 需要灵活的时间步长
 - 追求理论优雅性
 - 探索新的模型设计


 





### 5.6.3 新的可能性



Score SDE框架不仅统一了现有方法，更重要的是开启了设计新型扩散模型的大门。这里探讨一些令人兴奋的新方向。



#### 设计新的SDE



 超越标准选择

除了VP-SDE和VE-SDE，我们可以设计具有特殊性质的新SDE：



 - **自适应SDE**：

 $$dx = f(x,t,\text{SNR}(x))dt + g(t,\text{SNR}(x))dw$$


根据局部信噪比调整扩散速率


 - **各向异性SDE**：

 $$dx = f(x,t)dt + G(x,t)dw$$


$G$是状态依赖的扩散矩阵，不同方向扩散速率不同


 - **非线性SDE**：

 $$dx = -\nabla V(x)dt + \sqrt{2T(t)}dw$$


引入势能函数$V(x)$，实现特定的稳态分布







#### 混合采样策略




#### SDE-ODE混合


在生成过程的不同阶段使用不同的动力学：



 - **初始阶段（高噪声）**：使用ODE快速确定大致结构
 - **中间阶段**：使用SDE增加多样性
 - **最终阶段（低噪声）**：再次使用ODE精确细节



这种策略结合了两者的优势：速度、质量和多样性。




#### 条件生成的新方法



 通过修改SDE实现条件生成


 - **引导漂移**：

 $$dx = [f(x,t) + \lambda\nabla\log p(y|x)]dt + g(t)dw$$


在漂移项中加入条件信息


 - **条件扩散**：

 $$dx = f(x,t)dt + g(t,y)dw$$


扩散系数依赖于条件$y$


 - **约束SDE**：

在流形上定义SDE，自动满足某些约束







#### 加速采样的创新



 新型求解器设计


 - **预测-校正方法**：


 预测步：使用高阶ODE求解器
 - 校正步：使用少量Langevin动力学步骤


 
 - **自适应时间重参数化**：

在关键区域（如$t \approx 0$）使用更密集的时间步


 - **学习型求解器**：

使用神经网络学习最优的离散化方案







#### 多模态扩散




#### 不同模态的联合建模


设计处理多种数据类型的统一SDE：



 - **图像-文本联合SDE**：不同模态使用不同的扩散速率
 - **层次化SDE**：粗粒度和细粒度特征的分层扩散
 - **图结构SDE**：在图上定义的扩散过程





#### 理论创新方向



 推动理论边界


 - **最优传输视角**：


 设计最小化传输成本的SDE
 - 学习数据流形间的最优映射


 
 - **信息几何**：


 在概率流形上设计测地线SDE
 - 利用Fisher信息优化扩散路径


 
 - **控制论方法**：


 将生成过程视为最优控制问题
 - 学习最优的控制策略


 





#### 实际应用的新可能



 突破性应用


 - **科学计算**：


 分子动力学模拟
 - 量子系统建模
 - 气候模型


 
 - **逆问题求解**：


 医学成像重建
 - 地震波反演
 - 超分辨率


 
 - **生成式设计**：


 材料设计
 - 药物发现
 - 建筑设计


 





#### 未来展望




#### 连续时间框架的潜力


Score SDE框架为扩散模型开辟了广阔的研究空间：



 - **理论深度**：与数学物理的深度连接还有待挖掘
 - **算法创新**：新的数值方法和优化技术
 - **应用广度**：从图像生成到科学计算的全方位应用
 - **硬件协同**：为专用硬件设计的SDE



这个框架不仅是技术工具，更是理解生成模型的新范式。




## 5.7 数值方法与实现



### 5.7.1 SDE的数值解法



准确高效地求解SDE是实现扩散模型的关键。这里介绍主要的数值方法及其在扩散模型中的应用。



#### Euler-Maruyama方法



 最基本的SDE求解器

对于SDE：$dx = f(x,t)dt + g(t)dw$


Euler-Maruyama离散化为：


 $$x_{n+1} = x_n + f(x_n, t_n)\Delta t + g(t_n)\sqrt{\Delta t}\cdot z_n$$


其中 $z_n \sim \mathcal{N}(0, I)$。


**特点**：



 - 实现简单，计算效率高
 - 强收敛阶：$O(\sqrt{\Delta t})$
 - 弱收敛阶：$O(\Delta t)$
 - 对于线性SDE是精确的





#### Heun方法（改进Euler）




#### 预测-校正方法


Heun方法通过两步提高精度：



 - **预测步**：

 $$\tilde{x}_{n+1} = x_n + f(x_n, t_n)\Delta t + g(t_n)\sqrt{\Delta t}\cdot z_n$$


 - **校正步**：

 $$x_{n+1} = x_n + \frac{1}{2}[f(x_n, t_n) + f(\tilde{x}_{n+1}, t_{n+1})]\Delta t + g(t_n)\sqrt{\Delta t}\cdot z_n$$





弱收敛阶提高到 $O(\Delta t^2)$。




#### 随机Runge-Kutta方法



 高阶方法

类似于确定性ODE的Runge-Kutta方法，但需要考虑随机积分：



 - **阶数选择**：通常使用1.5阶或2.5阶方法
 - **计算成本**：每步需要多次函数评估
 - **稳定性**：对刚性问题更稳定



适用于需要高精度的场景，但计算成本较高。




#### 指数积分器



 利用线性结构

对于具有线性漂移的SDE（如VP-SDE）：


 $$dx = -\frac{\beta(t)}{2}x dt + \sqrt{\beta(t)}dw$$


可以精确积分线性部分：


 $$x_{n+1} = e^{-\frac{1}{2}\int_{t_n}^{t_{n+1}}\beta(s)ds} x_n + \text{随机项}$$


**优势**：



 - 对线性部分是精确的
 - 数值稳定性好
 - 适合大步长





#### 自适应步长方法




#### 动态调整时间步


根据局部误差估计自动调整步长：



 - **误差估计**：比较不同阶数方法的结果
 - **步长控制**：

 $$\Delta t_{new} = \Delta t_{old} \cdot \left(\frac{\text{容差}}{\text{误差估计}}\right)^{1/p}$$


 - **拒绝机制**：误差过大时重新计算



在扩散模型中，通常在 $t \approx 0$ 附近需要更小的步长。




#### 反向SDE的特殊处理



 数值挑战与解决方案

**挑战**：



 - 分数函数在 $t \approx 0$ 附近可能很大
 - 数值误差累积
 - 需要处理边界条件



**解决方案**：



 - **时间重缩放**：使用 $\tau = \log(t)$ 等变换
 - **截断技巧**：在很小的 $t_{min}$ 停止
 - **方差缩放**：调整最后几步的噪声强度





#### 并行化策略



 提高计算效率


 - **批量并行**：


 同时处理多个样本
 - 共享分数函数计算


 
 - **时间并行**：


 Parareal算法
 - 多重打靶法


 
 - **GPU优化**：


 向量化随机数生成
 - 融合核函数


 





#### 实践建议




#### 选择合适的求解器

 
 
 场景
 推荐方法
 原因
 
 
 快速原型
 Euler-Maruyama
 简单易实现
 
 
 生产部署
 自适应Heun
 精度与效率平衡
 
 
 高质量生成
 高阶RK + 自适应
 最高精度
 
 
 线性SDE
 指数积分器
 利用特殊结构
 
 



### 5.7.2 ODE求解器的应用



概率流ODE为扩散模型带来了确定性采样的可能。这里介绍如何有效地使用ODE求解器，以及相关的技巧和挑战。



#### 标准ODE求解器



 经典方法在扩散模型中的应用

概率流ODE的一般形式：


 $$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 s_\theta(x,t)$$


**常用求解器**：



 - **RK45**：自适应步长的4/5阶Runge-Kutta
 - **DOP853**：8阶Dormand-Prince方法
 - **LSODA**：自动刚性检测
 - **Radau**：隐式方法，适合刚性问题





#### DDIM作为ODE求解器




#### 从离散到连续的视角


DDIM可以理解为概率流ODE的特殊离散化：



 - **DDIM更新规则**：

 $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t,t)$$


 - **对应的ODE**：VP-SDE的概率流ODE
 - **优势**：专门为扩散模型设计，数值性质好





#### DPM-Solver系列



 专用高阶求解器

DPM-Solver利用扩散ODE的特殊结构：



 - **指数积分**：精确处理线性部分
 - **多步方法**：利用历史信息提高精度
 - **阶数**：1阶到3阶版本



**关键创新**：



 - 变量变换：$\lambda_t = \log(\alpha_t/\sigma_t)$
 - 线性多步公式
 - 解析系数计算





#### 自适应步长策略



 智能时间步选择

**误差控制**：



 - 局部截断误差估计
 - 嵌入式Runge-Kutta对
 - Richardson外推



**步长调整**：


 $$h_{new} = h_{old} \cdot \min\left(f_{max}, \max\left(f_{min}, f_{safe}\left(\frac{\epsilon_{tol}}{\epsilon_{est}}\right)^{1/(p+1)}\right)\right)$$


其中 $f_{safe} \approx 0.9$ 是安全因子。




#### 时间重参数化技巧




#### 改善数值性质


通过时间变换改善ODE的条件数：



 - **对数时间**：$\tau = \log(t)$


 在 $t \approx 0$ 附近展开时间
 - 避免奇异性


 
 - **信噪比参数化**：$\tau = \log(\text{SNR}(t))$


 均匀化不同时刻的重要性
 - 改善收敛性


 
 - **学习的时间表**：


 神经网络学习最优时间映射
 - 适应具体任务


 





#### 刚性问题的处理



 数值稳定性挑战

**刚性的来源**：



 - 分数函数的大梯度
 - 多尺度动力学
 - 接近数据流形时的快速变化



**解决方法**：



 - **隐式方法**：向后Euler、Radau
 - **半隐式方法**：IMEX方案
 - **预条件技术**：改善条件数





#### 快速ODE采样技巧



 加速策略


 - **渐进式采样**：


 先用大步长生成粗略结果
 - 在关键区域细化


 
 - **并行ODE求解**：


 多重打靶法
 - 时间分解


 
 - **知识蒸馏**：


 学习少步求解器
 - 直接预测跳跃


 





#### 质量评估指标




#### 如何评价ODE求解质量



 - **轨迹误差**：与高精度参考解比较
 - **不变量保持**：检查概率守恒
 - **生成质量**：FID、IS等感知指标
 - **计算效率**：NFE（函数评估次数）





#### 实用建议



 最佳实践


 - **初始实验**：从DDIM或DPM-Solver开始
 - **精度需求高**：使用自适应RK45
 - **速度优先**：固定步长的专用求解器
 - **调试技巧**：


 可视化ODE轨迹
 - 监控局部误差
 - 检查数值稳定性


 





### 5.7.3 实现细节与技巧



成功实现连续时间扩散模型需要注意许多细节。这里分享一些实践中的关键技巧和常见陷阱。



#### 时间编码的实现



 连续时间的神经网络输入

**时间嵌入方法**：



 - **正弦编码**：

 $$\text{emb}(t) = [\sin(2^0 \pi t), \cos(2^0 \pi t), ..., \sin(2^{L-1} \pi t), \cos(2^{L-1} \pi t)]$$


 - **学习的嵌入**：


 MLP将标量 $t$ 映射到高维向量
 - 更灵活但需要更多参数


 
 - **傅里叶特征**：


 随机频率的正弦基函数
 - 理论保证的表达能力


 





#### 分数参数化选择




#### 不同参数化的权衡


分数函数可以通过不同方式参数化：



 - **直接预测分数**：$s_\theta(x,t) = \nabla \log p_t(x)$
 - **预测噪声**：$\epsilon_\theta(x,t)$，然后 $s_\theta = -\epsilon_\theta/\sigma_t$
 - **预测速度**：$v_\theta(x,t) = \dot{x}_t$
 - **预测去噪数据**：$\hat{x}_0 = f_\theta(x_t, t)$



选择影响训练稳定性和生成质量。




#### 数值稳定性技巧



 避免数值问题


 - **对数域计算**：


 使用 $\log \bar{\alpha}_t$ 而非 $\bar{\alpha}_t$
 - 避免下溢问题


 
 - **方差裁剪**：


 限制 $\beta(t)$ 的范围
 - 防止数值爆炸


 
 - **安全除法**：

 $$\frac{a}{b + \epsilon} \text{ 而非 } \frac{a}{b}$$







#### 边界条件处理



 $t \approx 0$ 和 $t \approx T$ 的特殊处理

**起始时刻 ($t \approx 0$)**：



 - 分数函数可能发散
 - 使用最小时间 $t_{min} = 10^{-5}$
 - 特殊的方差缩放



**终止时刻 ($t \approx T$)**：



 - 确保收敛到先验分布
 - 可能需要额外的噪声





#### 训练技巧




#### 提高训练效果



 - **重要性采样**：


 根据信噪比调整时间采样
 - 在困难区域采样更多


 
 - **损失加权**：


 不同时刻使用不同权重
 - 平衡各尺度的贡献


 
 - **预条件技巧**：


 输入和输出缩放
 - 改善梯度流


 





#### 内存优化



 大规模模型的实现


 - **梯度检查点**：


 时间换空间
 - 在UNet的特定层使用


 
 - **混合精度训练**：


 FP16计算，FP32累积
 - 注意数值稳定性


 
 - **分布式策略**：


 数据并行
 - 模型并行（大模型）


 





#### 调试和验证



 确保正确实现


 - **单元测试**：


 测试时间离散化的一致性
 - 验证概率守恒
 - 检查可逆性


 
 - **渐进测试**：


 从简单分布开始
 - 逐步增加复杂度


 
 - **可视化工具**：


 轨迹可视化
 - 分数场可视化
 - 中间状态检查


 





#### 性能优化清单




#### 实现检查列表

 
 
 优化项
 影响
 难度
 
 
 使用编译优化（torch.compile）
 2-3x加速
 简单
 
 
 融合自定义CUDA核
 10-20%加速
 困难
 
 
 优化注意力计算
 显著内存节省
 中等
 
 
 缓存中间结果
 避免重复计算
 简单
 
 



#### 常见错误和解决方案



 避免常见陷阱


 - **时间方向错误**：


 确保前向是 $0 \to T$
 - 反向是 $T \to 0$


 
 - **方差参数化不一致**：


 统一使用 $\beta$ 或 $\alpha$
 - 注意累积乘积


 
 - **随机种子问题**：


 训练和采样使用不同种子
 - 确保可重现性


 





## 5.8 理论深入



### 5.8.1 存在性与唯一性



SDE解的存在性与唯一性是扩散模型理论基础的重要组成部分。这些结果保证了模型的数学良定性。



#### 基本存在唯一性定理



 Itô SDE的存在唯一性

考虑SDE：$dx_t = f(x_t, t)dt + g(t)dw_t$


如果满足以下条件：



 - **Lipschitz条件**：存在常数 $K$ 使得

 $$|f(x,t) - f(y,t)| \leq K|x-y|$$


 - **线性增长条件**：存在常数 $C$ 使得

 $$|f(x,t)|^2 + |g(t)|^2 \leq C(1 + |x|^2)$$





则对任意初值 $x_0$，SDE存在唯一的强解。




#### 扩散模型中的验证




#### 常见SDE的性质检验


**VP-SDE**：$dx = -\frac{\beta(t)}{2}x dt + \sqrt{\beta(t)}dw$



 - 漂移项线性：自动满足Lipschitz条件
 - 有界的 $\beta(t)$ 保证线性增长
 - 结论：存在唯一解



**VE-SDE**：$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}dw$



 - 无漂移项，条件自动满足
 - 只需 $\sigma(t)$ 连续可微





#### 反向SDE的存在性



 分数函数的正则性要求

反向SDE：$dx = [f(x,t) - g(t)^2\nabla\log p_t(x)]dt + g(t)d\bar{w}$


存在性需要分数函数满足：



 - **局部Lipschitz**：在紧集上Lipschitz连续
 - **多项式增长**：$|\nabla\log p_t(x)| \leq C(1 + |x|^k)$



神经网络通常满足这些条件。




#### 弱解与强解



 解的概念


 - **强解**：给定布朗运动 $w_t$，解 $x_t$ 是 $w_t$ 的函数
 - **弱解**：存在某个概率空间和布朗运动使得SDE成立



对于扩散模型：



 - 训练时只需要弱解（分布匹配）
 - 确定性采样需要强解





#### 爆炸时间与全局解




#### 解的长时间行为


即使局部解存在，也可能在有限时间爆炸。避免爆炸的充分条件：



 - **耗散性**：$\langle x, f(x,t) \rangle \leq -\alpha|x|^2 + \beta$
 - **有界扩散**：$|g(t)| \leq M$



VP-SDE的耗散性保证了全局解的存在。




#### 路径正则性



 解的连续性和可微性


 - **连续性**：SDE的解几乎必然连续
 - **Hölder连续性**：指数 $确保输出Lipschitz连续
 - 使用谱归一化等技术


 
 - **训练稳定性**：


 正则化保证解的存在性
 - 梯度裁剪防止爆炸


 
 - **数值方法选择**：


 强解理论支持显式方法
 - 刚性问题可能需要隐式方法


 





### 5.8.2 收敛性分析



收敛性分析是理解扩散模型长时间行为和采样质量的关键。这里探讨不同层面的收敛性质。



#### 分布收敛性



 前向过程的收敛

对于前向SDE，我们关心 $p_t$ 是否收敛到目标分布 $p_\infty$：



 - **VP-SDE**：$p_t \to \mathcal{N}(0, I)$ 当 $t \to \infty$
 - **VE-SDE**：$p_t$ 的方差趋于无穷，但标准化后收敛



**收敛速率**：通常是指数收敛

 $$W_2(p_t, p_\infty) \leq Ce^{-\lambda t}$$

 其中 $W_2$ 是Wasserstein-2距离。




#### 反向过程的收敛性




#### 生成质量的理论保证


反向SDE的收敛性依赖于：



 - **分数估计误差**：$\mathbb{E}[|s_\theta - \nabla\log p_t|^2]$
 - **离散化误差**：时间步长 $\Delta t$ 的影响
 - **有限时间截断**：在 $t_{min}$ 停止的影响



总误差界：

 $$W_2(p_{gen}, p_{data}) \leq C_1\sqrt{\epsilon_{score}} + C_2\sqrt{\Delta t} + C_3 t_{min}$$





#### 数值方法的收敛阶



 不同离散化的收敛性
 
 
 方法
 强收敛阶
 弱收敛阶
 
 
 Euler-Maruyama
 $O(\sqrt{\Delta t})$
 $O(\Delta t)$
 
 
 Milstein方法
 $O(\Delta t)$
 $O(\Delta t)$
 
 
 高阶Runge-Kutta
 $O(\Delta t)$
 $O(\Delta t^2)$
 
 

注：扩散模型主要关心弱收敛（分布层面）。




#### 遍历性与混合时间



 长时间行为

SDE的遍历性保证了时间平均等于空间平均：


 $$\lim_{T \to \infty} \frac{1}{T}\int_0^T f(x_t)dt = \int f(x)p_\infty(x)dx$$


**混合时间**：达到平衡分布所需时间



 - VP-SDE：$T_{mix} = O(\frac{1}{\beta_{min}}\log\frac{1}{\epsilon})$
 - 影响因素：噪声强度、初始分布、目标精度





#### 分数匹配的收敛性




#### 训练目标的渐近性质


分数匹配损失的最小值：


 $$\mathcal{L}^* = \inf_{s_\theta} \mathbb{E}_{t,x}[|s_\theta(x,t) - \nabla\log p_t(x)|^2]$$


**收敛保证**：



 - 神经网络的通用逼近性
 - 样本复杂度：$O(\frac{d}{\epsilon^2})$
 - 优化收敛：依赖于损失函数的凸性





#### KL散度的演化



 信息论视角

前向过程中KL散度的演化：


 $$\frac{d}{dt}D_{KL}(p_t \| p_\infty) = -\mathcal{I}(p_t)$$


其中 $\mathcal{I}$ 是Fisher信息。这表明：



 - KL散度单调递减
 - 收敛速率由Fisher信息决定
 - 几何解释：沿信息几何的测地线移动





#### 有限样本的影响



 统计误差分析

使用有限样本训练的影响：



 - **估计偏差**：$O(1/n)$
 - **估计方差**：$O(1/\sqrt{n})$
 - **泛化误差**：依赖于模型复杂度



实践建议：样本量 $n$ 应满足 $n \gg d^2/\epsilon^2$。




#### 加速收敛的技术




#### 改善收敛性的方法



 - **方差缩减**：


 控制变量法
 - 重要性采样


 
 - **预条件技术**：


 改变度量使问题更易求解
 - 自适应步长


 
 - **多尺度方法**：


 不同时间尺度的耦合
 - 由粗到细的策略


 





### 5.8.3 与最优传输的联系



扩散模型与最优传输理论有着深刻的联系。这种联系不仅提供了新的理论视角，还启发了新的算法设计。



#### 最优传输问题回顾



 Monge-Kantorovich问题

给定两个概率分布 $p_0$ 和 $p_T$，最优传输问题是找到成本最小的传输方案：


 $$\inf_{\pi \in \Pi(p_0, p_T)} \int c(x,y) d\pi(x,y)$$


其中：



 - $\pi$ 是联合分布，边缘为 $p_0$ 和 $p_T$
 - $c(x,y)$ 是传输成本（通常是 $|x-y|^2$）
 - $\Pi(p_0, p_T)$ 是所有可行传输方案





#### 动态最优传输




#### Benamou-Brenier公式


最优传输可以表示为动态问题：


 $$W_2^2(p_0, p_T) = \inf_{(p_t, v_t)} \int_0^T \int |v_t(x)|^2 p_t(x) dx dt$$


约束条件：


 $$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$$


这将离散传输转化为连续时间的流问题。




#### 扩散模型作为正则化最优传输



 熵正则化的视角

扩散过程可以看作带熵正则化的最优传输：



 - **Schrödinger桥问题**：

 $$\inf_{\mathbb{P}} D_{KL}(\mathbb{P} \| \mathbb{Q}) \text{ s.t. } \mathbb{P}_0 = p_0, \mathbb{P}_T = p_T$$

 其中 $\mathbb{Q}$ 是参考过程（如布朗运动）

 - **与扩散的联系**：


 解是一个扩散过程
 - 漂移由分数函数决定
 - 正则化参数对应扩散强度


 





#### 概率流ODE与位移插值



 McCann插值

在最优传输中，两个分布之间的测地线由位移插值给出：


 $$p_t = ((1-t)\text{Id} + tT)_\# p_0$$


其中 $T$ 是最优传输映射。


**与概率流ODE的关系**：



 - 概率流ODE定义了一种特殊的插值
 - 当扩散趋于0时，收敛到最优传输
 - 提供了计算测地线的实用方法





#### Wasserstein梯度流




#### 能量泛函的梯度流


许多PDE可以写成Wasserstein空间上的梯度流：


 $$\frac{\partial p}{\partial t} = \nabla \cdot \left(p \nabla \frac{\delta \mathcal{F}[p]}{\delta p}\right)$$


扩散模型的联系：



 - 前向过程：相对熵的梯度流
 - 反向过程：可以设计为某个能量的梯度流
 - 提供了变分原理的解释





#### 计算优势



 为什么这个联系重要？


 - **新的算法**：


 基于OT的采样方法
 - 更好的插值路径
 - 加速技术


 
 - **理论保证**：


 收敛性分析
 - 最优性条件
 - 稳定性结果


 
 - **应用扩展**：


 不同度量空间
 - 约束传输问题
 - 多边际问题


 





#### 流匹配与最优传输



 统一框架

流匹配方法直接学习传输速度场：



 - **目标**：学习 $v_t$ 使得 $(p_t, v_t)$ 解决传输问题
 - **优势**：避免学习分数函数
 - **联系**：$v_t = f(x,t) - \frac{g^2}{2}\nabla\log p_t$



这提供了扩散模型的另一种参数化。




#### 未来方向




#### 开放问题



 - **非欧几里得空间**：


 流形上的扩散
 - 图上的最优传输


 
 - **多模态传输**：


 不同空间之间的映射
 - Gromov-Wasserstein距离


 
 - **计算效率**：


 利用OT结构加速
 - 稀疏传输方案


 





## 5.9 本章小结




### 核心概念回顾



#### 1. 从离散到连续的演进



 - 离散时间扩散模型（DDPM）的局限性
 - 连续时间极限的自然性和优势
 - SDE作为描述扩散过程的统一框架




#### 2. 三大支柱



 - **SDE（随机微分方程）**：描述单个粒子的随机演化
 - **ODE（概率流）**：提供确定性的替代方案
 - **PDE（Fokker-Planck）**：刻画概率密度的演化




#### 3. 关键数学对象



 - **分数函数** $\nabla \log p_t(x)$：连接所有视角的核心
 - **Anderson定理**：时间反演的理论基础
 - **Score SDE框架**：统一不同扩散模型




#### 4. 实践要点



 - 数值求解器的选择：SDE vs ODE
 - 时间编码和边界处理
 - 训练稳定性和收敛性保证




#### 5. 理论深度



 - 存在唯一性定理保证数学良定性
 - 收敛性分析指导算法设计
 - 与最优传输的联系开启新方向





## 5.10 练习题




练习 5.1：从离散到连续

考虑DDPM的前向过程：$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t$


设 $\alpha_t = 1 - \beta \Delta t$，其中 $\Delta t = 1/T$，$T$ 是总步数。



 - 推导当 $T \to \infty$ 时的连续时间SDE
 - 计算对应的边缘分布 $p(x_t|x_0)$
 - 验证该SDE是VP-SDE的特例

**解答：**



 - 将离散递推展开：
 $$x_t - x_{t-1} = (\sqrt{1-\beta\Delta t} - 1)x_{t-1} + \sqrt{\beta\Delta t}\cdot\frac{\epsilon_t}{\sqrt{\Delta t}}\sqrt{\Delta t}$$
 利用 $\sqrt{1-\beta\Delta t} \approx 1 - \frac{\beta\Delta t}{2}$，得到：
 $$\frac{x_t - x_{t-1}}{\Delta t} \approx -\frac{\beta}{2}x_{t-1} + \sqrt{\beta}\cdot\frac{\epsilon_t}{\sqrt{\Delta t}}$$
 取极限得到SDE：$dx = -\frac{\beta}{2}x dt + \sqrt{\beta}dw$

 - 这是线性SDE，解为：$x_t = e^{-\frac{\beta t}{2}}x_0 + \int_0^t e^{-\frac{\beta(t-s)}{2}}\sqrt{\beta}dw_s$
 因此：$p(x_t|x_0) = \mathcal{N}(e^{-\frac{\beta t}{2}}x_0, 1-e^{-\beta t})$

 - 这正是VP-SDE在常数 $\beta(t) = \beta$ 情况下的形式。






练习 5.2：反向SDE推导

给定前向SDE：$dx = f(x,t)dt + g(t)dw$



 - 使用Bayes定理说明为什么反向过程需要分数函数
 - 推导反向SDE的漂移项
 - 解释分数函数项 $-g(t)^2\nabla\log p_t(x)$ 的物理意义

**解答：**



 - 反向转移核需要：$p(x_{t-dt}|x_t) = \frac{p(x_t|x_{t-dt})p(x_{t-dt})}{p(x_t)}$
 对数形式：$\log p(x_{t-dt}|x_t) = \log p(x_t|x_{t-dt}) + \log p(x_{t-dt}) - \log p(x_t)$
 这解释了为什么需要 $\nabla \log p_t$。

 - 使用Girsanov定理或直接计算，可得反向漂移：
 $\tilde{f}(x,t) = f(x,t) - g(t)^2\nabla\log p_t(x)$

 - 物理意义：


 $f(x,t)$：原始的确定性漂移
 - $-g(t)^2\nabla\log p_t(x)$：补偿扩散造成的概率流失
 - 总效果：将概率"拉回"高密度区域






练习 5.3：概率流ODE

考虑VP-SDE：$dx = -\frac{\beta(t)}{2}x dt + \sqrt{\beta(t)}dw$



 - 写出对应的概率流ODE
 - 证明ODE和SDE产生相同的边缘分布
 - 讨论何时使用ODE vs SDE进行采样

**解答：**



 - 概率流ODE：$\frac{dx}{dt} = -\frac{\beta(t)}{2}x - \frac{\beta(t)}{2}\nabla\log p_t(x)$
 对于高斯分布，$\nabla\log p_t(x) = -\frac{x-\mu_t}{\sigma_t^2}$

 - 证明：两者对应相同的Fokker-Planck方程
 $$\frac{\partial p}{\partial t} = \frac{\beta(t)}{2}\nabla\cdot(xp) + \frac{\beta(t)}{2}\Delta p$$

 - 使用建议：


 ODE：需要精确编码/解码、插值、编辑
 - SDE：需要多样性、避免模式坍塌、纯生成任务






练习 5.4：Fokker-Planck方程

对于SDE：$dx = -x dt + \sqrt{2}dw$



 - 写出对应的Fokker-Planck方程
 - 求解稳态分布
 - 计算从任意初始分布到稳态的收敛时间

**解答：**



 - Fokker-Planck方程：
 $$\frac{\partial p}{\partial t} = \frac{\partial}{\partial x}(xp) + \frac{\partial^2 p}{\partial x^2}$$

 - 稳态条件 $\frac{\partial p}{\partial t} = 0$ 给出：
 $$\frac{d}{dx}(xp) + \frac{dp}{dx} = 0$$
 解得：$p_\infty(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$（标准正态分布）

 - 收敛速率由最小特征值决定，这里是指数收敛：
 $$\|p_t - p_\infty\|_{L^2} \leq e^{-t}\|p_0 - p_\infty\|_{L^2}$$






练习 5.5：数值方法比较

实现并比较不同的SDE数值解法：



 - 实现Euler-Maruyama和Heun方法
 - 在VP-SDE上测试收敛阶
 - 分析计算成本vs精度的权衡

**提示：**



 - Euler-Maruyama：$x_{n+1} = x_n + f(x_n,t_n)\Delta t + g(t_n)\sqrt{\Delta t}z_n$
 - Heun：先预测再校正，提高确定性部分的精度
 - 使用已知解析解的SDE（如OU过程）验证收敛阶
 - 绘制误差vs步长的对数图，斜率即为收敛阶






练习 5.6：Score SDE框架

证明以下等价关系：



 - DDPM的噪声预测 $\epsilon_\theta$ 与分数函数的关系
 - NCSN的多尺度训练与VE-SDE的联系
 - 设计一个新的SDE并分析其性质

**解答要点：**



 - 关系：$\epsilon_\theta(x_t, t) = -\sigma_t s_\theta(x_t, t)$
 其中 $\sigma_t = \sqrt{1-\bar{\alpha}_t}$

 - NCSN的噪声水平 $\{\sigma_i\}$ 对应VE-SDE在离散时刻的值
 - 新SDE示例：非线性漂移 $dx = -x^3 dt + \sqrt{2}dw$


 多模态稳态分布
 - 局部稳定性分析
 - 数值挑战：刚性问题






挑战题：最优传输视角

探索扩散模型与最优传输的联系：



 - 证明当 $g(t) \to 0$ 时，概率流ODE收敛到最优传输的位移插值
 - 实现基于最优传输的采样加速方法
 - 讨论Schrödinger桥与扩散模型的关系

**研究方向：**



 - 参考Benamou-Brenier公式和动态最优传输理论
 - 考虑熵正则化的作用
 - 流匹配方法提供了实际的算法思路
 - 这是当前研究的活跃领域，有很多开放问题





 [← 第4章：基于分数的生成模型](chapter4.md)
 [返回首页](index.md)
 [第6章：流匹配 →](chapter6.md)