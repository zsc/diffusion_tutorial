[← 返回目录](index.md) | 第8章 / 共14章 | [下一章 →](chapter9.md)

# 第8章：采样算法与加速技术

扩散模型的一个主要挑战是采样速度慢——DDPM需要1000步去噪才能生成高质量样本。本章深入探讨各种加速采样的算法创新，从DDIM的确定性采样到DPM-Solver的高阶求解器，再到最新的一致性模型。您将学习这些方法背后的数学原理，理解速度与质量的权衡，并掌握在实践中选择和调优采样算法的技巧。通过本章的学习，您将能够将采样步数从1000步减少到20步甚至更少，同时保持生成质量。

## 章节大纲

### 8.1 DDIM：去噪扩散隐式模型
- 从随机到确定性：DDIM的核心思想
- 非马尔可夫前向过程的构造
- DDIM采样器的推导与实现
- 插值与图像编辑应用

### 8.2 基于ODE/SDE的统一视角
- 概率流ODE的推导
- SDE与ODE的等价性
- 数值求解器的选择与分析
- 预测-校正采样框架

### 8.3 DPM-Solver系列算法
- 指数积分器与精确解
- DPM-Solver的高阶展开
- DPM-Solver++的改进
- 自适应步长策略

### 8.4 蒸馏与一步生成
- 渐进式蒸馏（Progressive Distillation）
- 引导蒸馏（Guidance Distillation）
- 一致性模型（Consistency Models）
- 对抗蒸馏方法

### 8.5 实践优化技巧
- 采样器的选择指南
- 噪声调度的优化
- 混合采样策略
- 质量-速度权衡分析

## 8.1 DDIM：去噪扩散隐式模型

### 8.1.1 DDPM采样的局限性

回顾DDPM的反向过程：
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$$

每一步都需要添加随机噪声 $\sigma_t \boldsymbol{\epsilon}$ ，这导致：
1. **采样的随机性**：相同的 $\mathbf{x}_T$ 会生成不同的 $\mathbf{x}_0$
2. **步数依赖**：减少步数会显著降低质量
3. **不可逆性**：无法从生成的图像精确重构初始噪声

DDIM通过重新参数化前向过程，巧妙地解决了这些问题。

### 8.1.2 DDIM的核心创新

DDIM的关键洞察是：存在一族非马尔可夫前向过程，它们具有相同的边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ ，但对应的反向过程可以是确定性的。

具体地，DDIM定义了一个新的前向过程：
$$q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \sigma_t^2\mathbf{I})$$

其中：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}$$

当 $\sigma_t = 0$ 时，过程变为完全确定性。

### 8.1.3 DDIM采样算法

DDIM的采样公式为：
$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 } \mathbf{x}_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}_{\text{方向指向 } \mathbf{x}_t} + \underbrace{\sigma_t \boldsymbol{\epsilon}}_{\text{随机噪声}}$$

关键参数 $\eta$ 控制随机性：
- $\eta = 0$ ：完全确定性（DDIM）
- $\eta = 1$ ：等价于DDPM
- $0 < \eta < 1$ ：介于两者之间

💡 **实现技巧：加速采样**  
DDIM允许使用子序列采样。例如，从1000步中均匀选择50步：`timesteps = np.linspace(0, 999, 50).astype(int)`。这可以实现20倍加速！

<details>
<summary>**练习 8.1：理解DDIM的几何意义**</summary>

考虑2D高斯分布的扩散过程。

1. **轨迹可视化**：
   - 实现DDPM和DDIM的采样过程
   - 从相同的 $\mathbf{x}_T$ 开始，绘制多条去噪轨迹
   - 观察DDIM轨迹的确定性 vs DDPM的随机性

2. **插值实验**：
   - 生成两个不同的样本 $\mathbf{x}_0^{(1)}, \mathbf{x}_0^{(2)}$
   - 编码到对应的 $\mathbf{x}_T^{(1)}, \mathbf{x}_T^{(2)}$
   - 在潜在空间插值： $\mathbf{x}_T^{(\lambda)} = (1-\lambda)\mathbf{x}_T^{(1)} + \lambda\mathbf{x}_T^{(2)}$
   - 解码并观察语义插值效果

3. **速度-质量权衡**：
   - 使用不同的步数（10, 20, 50, 100, 1000）
   - 计算FID分数和推理时间
   - 找出最优的步数选择

4. **理论拓展**：
   - 推导DDIM的最优 $\sigma_t$ 选择
   - 研究非均匀时间步长的影响
   - 探索自适应步长策略

</details>

### 8.1.4 DDIM的数学解释

DDIM可以从多个角度理解：

**1. 变分推断视角**

DDIM最小化了一个修改后的变分下界，其中KL散度项被重新加权。

**2. 数值ODE求解器视角**

当 $\eta = 0$ 时，DDIM等价于求解概率流ODE：
$$\frac{d\mathbf{x}_t}{dt} = -\frac{1}{2}\beta_t\left[\mathbf{x}_t + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)\right]$$

**3. 最优传输视角**

DDIM寻找从噪声到数据的"直线"路径，最小化传输成本。

🔬 **研究线索：广义DDIM**  
能否设计更一般的确定性采样器？例如，使用高阶信息（二阶导数）或历史轨迹信息来改进预测？这涉及到数值分析和最优控制理论。

## 8.2 基于ODE/SDE的统一视角

### 8.2.1 从离散到连续：扩散SDE

Song等人(2021)提出了基于随机微分方程(SDE)的统一框架。前向扩散过程可以表示为：
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

其中：
- $\mathbf{f}(\cdot, t)$ ：漂移系数
- $g(t)$ ：扩散系数
- $\mathbf{w}$ ：标准维纳过程

对于DDPM/DDIM，相应的SDE是：
$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$$

### 8.2.2 反向时间SDE

Anderson(1982)证明了反向时间SDE的存在性：
$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2\nabla_\mathbf{x} \log p_t(\mathbf{x})]dt + g(t)d\bar{\mathbf{w}}$$

其中 $\bar{\mathbf{w}}$ 是反向时间的维纳过程， $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ 是分数函数（score function）。

### 8.2.3 概率流ODE

去除随机项，得到确定性的ODE：
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2\nabla_\mathbf{x} \log p_t(\mathbf{x})$$

这个ODE与原始SDE具有相同的边缘分布 $p_t(\mathbf{x})$ 。

**关键性质**：
1. **可逆性**：可以在数据和噪声之间双向转换
2. **确定性**：给定初值，轨迹唯一确定
3. **保持分布**：ODE轨迹保持概率分布的演化

### 8.2.4 数值求解器的选择

不同的ODE求解器对应不同的采样算法：

| 求解器 | 阶数 | 对应算法 | 特点 |
|--------|------|----------|------|
| Euler | 1 | DDIM | 简单快速 |
| Heun | 2 | DPM-Solver-2 | 精度提升 |
| RK4 | 4 | - | 高精度但计算量大 |
| 线性多步 | 可变 | DPM-Solver-3 | 利用历史信息 |

💡 **实践建议：求解器选择**  
- 步数充足(>50)：使用Euler方法(DDIM)
- 步数有限(10-50)：使用2阶或3阶求解器
- 极少步数(<10)：需要专门优化的求解器

<details>
<summary>**练习 8.2：实现和比较ODE求解器**</summary>

实现并比较不同的ODE求解器用于扩散模型采样。

1. **基础实现**：
   - 实现Euler方法（DDIM）
   - 实现Heun方法（2阶）
   - 实现RK4方法（4阶）

2. **误差分析**：
   - 使用已知解析解的toy problem测试
   - 绘制全局误差vs步长的log-log图
   - 验证理论收敛阶

3. **扩散模型应用**：
   - 在训练好的模型上比较不同求解器
   - 固定计算预算，比较生成质量
   - 分析每个求解器的最优步数

4. **高级探索**：
   - 实现自适应步长控制
   - 研究刚性ODE求解器（implicit methods）
   - 探索预测-校正方法

</details>

🌟 **开放问题：最优ODE公式**  
当前的概率流ODE是否是最优的？是否存在收敛更快的等价ODE？这涉及到动力系统理论和最优控制。

## 8.3 DPM-Solver系列算法

### 8.3.1 动机：利用半线性结构

扩散ODE具有特殊的半线性结构：
$$\frac{d\mathbf{x}}{dt} = \alpha(t)\mathbf{x} + \sigma(t)\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)$$

其中线性部分 $\alpha(t)\mathbf{x}$ 有解析解，这启发了DPM-Solver的设计。

### 8.3.2 指数积分器

利用积分因子法，可以得到精确解：
$$\mathbf{x}_s = e^{\int_t^s \alpha(\tau)d\tau}\mathbf{x}_t + \int_t^s e^{\int_\tau^s \alpha(r)dr}\sigma(\tau)\boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \tau)d\tau$$

关键是如何近似积分中的 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \tau)$ 。

### 8.3.3 DPM-Solver的Taylor展开

DPM-Solver使用Taylor展开近似噪声预测：
$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \tau) = \sum_{n=0}^{k-1} \frac{(\tau - t)^n}{n!}\frac{d^n\boldsymbol{\epsilon}_\theta}{d\tau^n}\bigg|_{\tau=t} + O((\tau-t)^k)$$

不同阶数的DPM-Solver：
- **DPM-Solver-1**：常数近似，等价于DDIM
- **DPM-Solver-2**：线性近似，需要2次网络评估
- **DPM-Solver-3**：二次近似，需要3次网络评估

### 8.3.4 DPM-Solver++的改进

DPM-Solver++引入了两个关键改进：

1. **数据预测参数化**：预测 $\mathbf{x}_0$ 而非 $\boldsymbol{\epsilon}$
   $$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sigma_t\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\alpha_t}$$

2. **thresholding**：动态裁剪防止数值不稳定
   $$\mathbf{x}_0 = \text{clip}(\mathbf{x}_0, -1, 1)$$

**算法伪代码**：
```python
def dpm_solver_pp_2nd(x_t, t, s, model, old_pred=None):
    # 第一步预测
    pred_t = predict_x0(x_t, t, model)
    
    if old_pred is None:
        # 使用一阶方法
        x_s = update_rule_1st(x_t, t, s, pred_t)
    else:
        # 使用二阶方法
        x_s = update_rule_2nd(x_t, t, s, pred_t, old_pred)
    
    return x_s, pred_t
```

🔬 **研究方向：高阶求解器的稳定性**  
高阶方法理论上更准确，但在实践中可能不稳定。如何设计既高阶又稳定的求解器？可以借鉴刚性ODE求解器的思想。

### 8.3.5 自适应步长策略

固定步长可能不是最优的。自适应策略根据局部误差调整步长：

$$h_{new} = h_{old} \cdot \left(\frac{\text{tolerance}}{\text{error}}\right)^{1/(p+1)}$$

其中 $p$ 是求解器阶数。

<details>
<summary>**练习 8.3：实现DPM-Solver**</summary>

1. **基础实现**：
   - 实现DPM-Solver-1,2,3
   - 比较不同阶数的收敛速度
   - 分析计算成本vs质量

2. **参数化研究**：
   - 比较噪声预测vs数据预测
   - 研究thresholding的影响
   - 探索不同的时间离散化

3. **自适应步长**：
   - 实现误差估计器
   - 设计步长控制策略
   - 在不同数据集上测试

4. **理论分析**：
   - 推导局部截断误差
   - 分析稳定性区域
   - 研究与SDE离散化的联系

</details>

## 8.4 蒸馏与一步生成

### 8.4.1 渐进式蒸馏

渐进式蒸馏(Progressive Distillation)逐步减少采样步数：
1. 训练教师模型（N步）
2. 训练学生模型（N/2步）匹配教师输出
3. 重复直到达到目标步数

**损失函数**：
$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|f_\theta(\mathbf{x}_t, t) - \text{sg}[f_{\text{teacher}}(\mathbf{x}_t, t)]\|^2\right]$$

其中 `sg` 表示停止梯度。

### 8.4.2 一致性模型

一致性模型(Consistency Models)学习映射函数 $f_\theta$ ，使得同一轨迹上的所有点映射到相同的起点：

$$f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_s, s), \quad \forall s, t \in [0, T]$$

**自一致性损失**：
$$\mathcal{L} = \mathbb{E}\left[\|f_\theta(\mathbf{x}_t, t) - f_{\theta^-}(\mathbf{x}_s, s)\|^2\right]$$

其中 $\theta^-$ 是EMA参数。

💡 **关键创新**：一致性模型可以一步生成，也可以多步精炼，提供了灵活的质量-速度权衡。

### 8.4.3 对抗蒸馏

结合GAN的思想，使用判别器指导蒸馏：
$$\mathcal{L} = \mathcal{L}_{\text{distill}} + \lambda \mathcal{L}_{\text{adv}}$$

这可以进一步提升少步采样的质量。

🌟 **未来方向：理论最优的蒸馏**  
当前的蒸馏方法大多是启发式的。是否存在理论最优的蒸馏策略？这涉及到最优传输理论和信息论。

## 8.5 实践优化技巧

### 8.5.1 采样器选择指南

| 场景 | 推荐采样器 | 步数 | 说明 |
|------|------------|------|------|
| 高质量 | DDPM | 1000 | 最高质量，最慢 |
| 平衡 | DPM-Solver++ | 20-50 | 质量好，速度快 |
| 实时 | 一致性模型 | 1-4 | 最快，质量可接受 |
| 可控编辑 | DDIM | 50-100 | 确定性，支持插值 |

### 8.5.2 噪声调度优化

**1. 端到端优化**：学习最优的 $\beta_t$ 或 $\bar{\alpha}_t$
**2. 截断采样**：跳过信噪比极高的早期步骤
**3. 非均匀步长**：在关键区域使用更密集的步长

### 8.5.3 混合策略

结合不同采样器的优势：
- 前期使用高阶求解器快速去噪
- 后期使用DDPM精细调整
- 关键步骤使用预测-校正

### 8.5.4 实现优化

```python
# 缓存计算
alpha_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

# 向量化操作
def sample_batch(x_T, timesteps, model):
    x = x_T
    for t in reversed(timesteps):
        # 批量处理所有样本
        x = sampling_step(x, t, model)
    return x

# JIT编译
@torch.jit.script
def sampling_step(x, t, noise_pred, alpha, sigma):
    # 采样步骤的高效实现
    ...
```

<details>
<summary>**综合练习：设计自适应采样器**</summary>

设计一个根据图像内容自适应调整采样策略的算法。

1. **难度估计**：
   - 基于中间结果估计剩余去噪难度
   - 设计难度指标（如预测不确定性）

2. **自适应策略**：
   - 简单区域：使用大步长或低阶方法
   - 复杂区域：使用小步长或高阶方法
   - 实现动态步长分配

3. **多尺度处理**：
   - 低分辨率快速预览
   - 高分辨率精细生成
   - 设计多尺度调度策略

4. **基准测试**：
   - 在不同数据集上评估
   - 与固定策略比较
   - 分析计算节省vs质量损失

</details>

本章深入探讨了扩散模型的各种采样加速技术，从DDIM的确定性采样到基于ODE的统一框架，再到最新的一致性模型。这些方法将采样速度提升了数十倍，使扩散模型的实际应用成为可能。下一章，我们将探讨如何通过条件机制控制生成过程。

[← 返回目录](index.md) | 第8章 / 共14章 | [下一章 →](chapter9.md)