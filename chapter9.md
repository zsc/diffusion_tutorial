[← 返回目录](index.md) | 第9章 / 共14章 | [下一章 →](chapter10.md)

# 第9章：条件生成与引导技术

条件生成是扩散模型最重要的应用之一，它使我们能够控制生成过程，产生符合特定要求的样本。本章深入探讨各种条件生成技术，从基于分类器的引导到无分类器引导，再到最新的控制方法。您将学习如何在数学上理解这些引导机制，掌握在不同场景下选择和实现条件生成的技巧，并了解如何平衡生成质量与条件遵循度。通过本章的学习，您将能够构建强大的可控生成系统。

## 章节大纲

### 9.1 条件扩散模型的基础
- 条件分布的建模
- 条件信息的注入方式
- 架构设计考虑
- 训练策略

### 9.2 分类器引导（Classifier Guidance）
- 理论推导与直觉
- 梯度计算与实现
- 引导强度的影响
- 局限性分析

### 9.3 无分类器引导（Classifier-Free Guidance）
- 动机与核心思想
- 条件与无条件模型的联合训练
- 引导公式推导
- 实践中的技巧

### 9.4 高级引导技术
- 多条件组合
- 负向提示（Negative Prompting）
- 动态引导强度
- ControlNet与适配器方法

### 9.5 评估与优化
- 条件一致性度量
- 多样性与质量权衡
- 引导失效的诊断
- 实际应用案例

## 9.1 条件扩散模型的基础

### 9.1.1 条件分布的数学框架

在条件扩散模型中，我们的目标是建模条件分布 $p(\mathbf{x}|\mathbf{c})$，其中 $\mathbf{x}$ 是数据（如图像），$\mathbf{c}$ 是条件信息（如类别标签、文本描述等）。

条件扩散过程定义为：
- **前向过程**：$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$（与条件无关）
- **反向过程**：$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, \mathbf{c}), \sigma_t^2\mathbf{I})$

关键在于如何设计和训练条件去噪网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$。

### 9.1.2 条件信息的注入方式

**1. 拼接（Concatenation）**

最直接的方式是将条件信息与输入拼接：
```python
# 对于图像条件
x_with_cond = torch.cat([x_t, c_image], dim=1)
# 对于向量条件
c_embed = condition_encoder(c)
x_with_cond = torch.cat([x_t, c_embed.unsqueeze(-1).unsqueeze(-1).expand(...)])
```

**2. 自适应归一化（Adaptive Normalization）**

通过条件信息调制归一化参数：
```python
# AdaIN, AdaGN, AdaLN等
gamma, beta = mlp(c_embed)
h = normalize(h)
h = gamma * h + beta
```

**3. 交叉注意力（Cross-Attention）**

特别适合序列条件（如文本）：
```python
# Q来自图像特征，K,V来自文本编码
attn_output = CrossAttention(
    query=image_features,
    key=text_features,
    value=text_features
)
```

**4. 特征调制（Feature-wise Modulation）**

FiLM层通过条件信息缩放和偏移特征：
```python
gamma, beta = film_generator(c)
h = gamma * h + beta
```

🔬 **研究线索：最优注入位置**  
应该在网络的哪些层注入条件信息？早期层vs后期层？所有层vs特定层？这可能依赖于条件类型和任务。

### 9.1.3 架构设计原则

**1. 条件编码器设计**

不同类型的条件需要不同的编码器：
- **类别标签**：嵌入层 + MLP
- **文本**：预训练语言模型（CLIP, T5等）
- **图像**：预训练视觉模型或专用CNN
- **音频**：频谱图编码器

**2. 多尺度条件注入**

在U-Net的不同分辨率注入条件：
```python
class ConditionalUNet(nn.Module):
    def forward(self, x, t, c):
        # 编码路径
        h1 = self.down1(x, t, c)  # 高分辨率条件
        h2 = self.down2(h1, t, c)  # 中分辨率条件
        h3 = self.down3(h2, t, c)  # 低分辨率条件
        
        # 解码路径也注入条件
        ...
```

**3. 时间-条件交互**

时间步和条件信息可能需要交互：
```python
# 联合编码
t_embed = self.time_embed(t)
c_embed = self.cond_embed(c)
joint_embed = self.joint_mlp(t_embed + c_embed)
```

### 9.1.4 训练策略

**1. 条件dropout**

随机丢弃条件信息，训练模型同时处理条件和无条件生成：
```python
def training_step(x, c):
    # 以概率p_uncond丢弃条件
    if random.random() < p_uncond:
        c = null_condition
    
    # 正常训练
    noise = torch.randn_like(x)
    x_t = add_noise(x, noise, t)
    pred_noise = model(x_t, t, c)
    loss = F.mse_loss(pred_noise, noise)
```

这是无分类器引导的基础。

**2. 条件增强**

对条件信息进行数据增强：
- 文本：同义词替换、改写
- 图像：几何变换、颜色扰动
- 类别：标签平滑、混合

**3. 多任务学习**

同时训练多种条件：
```python
loss = loss_uncond + λ1*loss_class + λ2*loss_text + λ3*loss_image
```

💡 **实践技巧：条件缩放**  
不同条件的强度可能需要不同的缩放。使用可学习的缩放因子：`c_scaled = c * self.condition_scale`

<details>
<summary>**练习 9.1：实现多模态条件扩散模型**</summary>

设计一个支持多种条件类型的扩散模型。

1. **基础架构**：
   - 实现支持类别、文本、图像条件的U-Net
   - 设计灵活的条件注入机制
   - 处理条件缺失的情况

2. **条件编码器**：
   - 类别：可学习嵌入
   - 文本：使用预训练CLIP
   - 图像：轻量级CNN编码器

3. **训练实验**：
   - 比较不同注入方式的效果
   - 研究条件dropout率的影响
   - 测试多条件组合

4. **扩展研究**：
   - 设计条件强度的自适应调整
   - 实现条件插值
   - 探索新的条件类型（如草图、深度图）

</details>

### 9.1.5 条件一致性的理论保证

**变分下界的条件版本**：

$$\log p_\theta(\mathbf{x}_0|\mathbf{c}) \geq \mathbb{E}_q\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1, \mathbf{c}) - \sum_{t=2}^T D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}))\right]$$

这保证了模型学习的是真实的条件分布。

**条件独立性假设**：

在许多实现中，我们假设：
$$q(\mathbf{x}_t|\mathbf{x}_0, \mathbf{c}) = q(\mathbf{x}_t|\mathbf{x}_0)$$

即前向过程与条件无关。这简化了训练但可能限制了模型能力。

🌟 **开放问题：条件相关的前向过程**  
是否可以设计依赖于条件的前向过程？例如，对不同类别使用不同的噪声调度？这可能提供更好的归纳偏置。

### 9.1.6 实现细节与优化

**内存优化**：
```python
# 使用gradient checkpointing节省内存
class ConditionalBlock(nn.Module):
    @torch.utils.checkpoint.checkpoint
    def forward(self, x, c):
        # 计算密集的操作
        ...
```

**计算优化**：
```python
# 缓存条件编码
class CachedConditionEncoder:
    def __init__(self):
        self.cache = {}
    
    def encode(self, c):
        if c not in self.cache:
            self.cache[c] = self.encoder(c)
        return self.cache[c]
```

**数值稳定性**：
```python
# 防止条件编码的数值问题
c_encoded = F.normalize(c_encoded, dim=-1) * self.scale
```

## 9.2 分类器引导（Classifier Guidance）

### 9.2.1 理论推导

分类器引导的核心思想是使用外部分类器的梯度来引导扩散模型的采样过程。我们从贝叶斯规则开始：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{c}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t)$$

第一项是无条件分数，第二项是分类器的梯度。这给出了条件采样的更新规则：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

其中 $p_\phi(\mathbf{c}|\mathbf{x}_t)$ 是在噪声数据上训练的分类器。

### 9.2.2 噪声条件分类器

关键挑战是训练一个能在所有噪声水平 $t$ 上工作的分类器。训练过程：

```python
def train_noise_conditional_classifier(classifier, diffusion, dataloader):
    for x, c in dataloader:
        # 随机采样时间步
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],))
        
        # 添加相应的噪声
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise)
        
        # 分类器预测
        logits = classifier(x_t, t)
        loss = F.cross_entropy(logits, c)
        
        loss.backward()
```

分类器架构需要：
1. 时间条件：了解当前噪声水平
2. 鲁棒性：在高噪声下仍能提取有用特征
3. 梯度质量：提供有意义的引导信号

### 9.2.3 引导强度与采样

引导强度 $s$ 控制条件的影响程度：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s\sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

- $s = 0$：无条件生成
- $s = 1$：标准条件生成
- $s > 1$：强化条件，可能降低多样性
- $s < 0$：负向引导，远离条件

**采样算法**：
```python
def classifier_guided_sampling(x_T, model, classifier, c, s=1.0):
    x = x_T
    for t in reversed(range(T)):
        # 无条件预测
        epsilon = model(x, t)
        
        # 计算分类器梯度
        x.requires_grad_(True)
        logits = classifier(x, t)
        log_prob = F.log_softmax(logits, dim=-1)[range(len(c)), c]
        grad = torch.autograd.grad(log_prob.sum(), x)[0]
        x.requires_grad_(False)
        
        # 组合预测
        epsilon_tilde = epsilon - s * sqrt(1 - alphas_cumprod[t]) * grad
        
        # 采样步骤
        x = sampling_step(x, epsilon_tilde, t)
    
    return x
```

### 9.2.4 梯度计算的实践考虑

**1. 梯度缩放**

不同时间步的梯度量级差异很大，需要自适应缩放：
```python
# 根据噪声水平调整梯度
grad_scale = 1.0 / (1 - alphas_cumprod[t]).sqrt()
scaled_grad = grad * grad_scale
```

**2. 梯度裁剪**

防止梯度爆炸：
```python
grad_norm = grad.flatten(1).norm(dim=1, keepdim=True)
grad = grad / grad_norm.clamp(min=1e-8)
```

**3. 多步梯度**

使用多步梯度累积获得更稳定的方向：
```python
grad_accum = 0
for _ in range(n_grad_steps):
    grad = compute_classifier_grad(x + noise_scale * torch.randn_like(x))
    grad_accum += grad
grad = grad_accum / n_grad_steps
```

💡 **实践技巧：温度调节**  
对分类器输出使用温度缩放可以控制引导的锐度：`logits = classifier(x, t) / temperature`

### 9.2.5 局限性分析

**1. 需要额外的分类器**
- 增加训练成本
- 分类器质量影响生成质量
- 需要为每个条件类型训练分类器

**2. 梯度质量问题**
- 高噪声下梯度可能无意义
- 对抗样本问题
- 梯度消失/爆炸

**3. 模式崩溃风险**
- 过强的引导导致多样性丧失
- 生成分布偏离真实分布
- 难以平衡质量和多样性

**4. 计算开销**
- 每步需要额外的前向和反向传播
- 内存占用增加
- 采样速度显著降低

<details>
<summary>**练习 9.2：分析分类器引导的行为**</summary>

深入研究分类器引导在不同设置下的表现。

1. **引导强度实验**：
   - 在MNIST上训练扩散模型和分类器
   - 测试不同引导强度 s ∈ [0, 0.5, 1, 2, 5, 10]
   - 绘制生成质量vs多样性曲线

2. **梯度可视化**：
   - 可视化不同时间步的分类器梯度
   - 分析梯度方向的语义含义
   - 研究梯度范数的变化

3. **失效模式分析**：
   - 识别分类器引导失败的案例
   - 分析过度引导的表现
   - 设计改进策略

4. **理论拓展**：
   - 推导最优引导强度的理论
   - 研究引导对生成分布的影响
   - 探索自适应引导强度

</details>

### 9.2.6 改进与变体

**1. 截断引导**

只在特定时间范围内应用引导：
```python
if t > T_start and t < T_end:
    epsilon = epsilon - s * grad
```

**2. 局部引导**

只对图像的特定区域应用引导：
```python
mask = compute_attention_mask(x, c)
epsilon = epsilon - s * grad * mask
```

**3. 多分类器集成**

使用多个分类器的组合：
```python
grad_ensemble = 0
for classifier in classifiers:
    grad_ensemble += compute_grad(classifier, x, t, c)
grad = grad_ensemble / len(classifiers)
```

🔬 **研究方向：隐式分类器**  
能否从扩散模型本身提取分类器，避免训练额外模型？这涉及到对扩散模型内部表示的深入理解。

### 9.2.7 与其他方法的联系

分类器引导与其他生成模型技术有深刻联系：

**1. 与GAN的判别器引导类似**
- 都使用外部模型提供梯度信号
- 都面临训练不稳定的问题

**2. 与能量模型的关系**
- 分类器定义了能量景观
- 引导相当于在能量景观上的梯度下降

**3. 与强化学习的奖励引导**
- 分类器概率类似奖励信号
- 可以借鉴RL中的技术（如PPO）

🌟 **未来展望：统一的引导框架**  
是否存在一个统一的理论框架，涵盖所有类型的引导？这可能需要从最优控制或变分推断的角度重新思考。

## 9.3 无分类器引导（Classifier-Free Guidance）

### 9.3.1 动机与核心洞察

无分类器引导（CFG）解决了分类器引导的主要限制：不需要训练额外的分类器。核心思想是同时训练条件和无条件扩散模型，然后在采样时组合它们的预测。

基本原理基于：
$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{c}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t)$$

CFG通过隐式估计 $\nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t)$：
$$\nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{c}) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$$

### 9.3.2 训练策略：条件Dropout

关键创新是在训练时随机丢弃条件：

```python
def train_classifier_free(model, x, c, p_uncond=0.1):
    # 随机决定是否使用条件
    if torch.rand(1).item() < p_uncond:
        # 无条件训练
        c = null_token  # 特殊的空条件标记
    
    # 标准扩散模型训练
    t = torch.randint(0, num_timesteps, (x.shape[0],))
    noise = torch.randn_like(x)
    x_t = add_noise(x, noise, t)
    
    # 预测噪声
    pred_noise = model(x_t, t, c)
    loss = F.mse_loss(pred_noise, noise)
    
    return loss
```

这使得单个模型能够处理条件和无条件生成。

### 9.3.3 采样公式

CFG的采样公式：
$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1 + w)\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - w\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$$

其中：
- $w$：引导权重（guidance weight）
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$：条件预测
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$：无条件预测

这可以重写为：
$$\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

显示了从无条件预测出发，朝条件方向移动的解释。

### 9.3.4 实现细节

**高效采样实现**：
```python
def cfg_sampling(model, shape, c, w=7.5, eta=0):
    # 初始噪声
    x = torch.randn(shape)
    
    # 准备条件和无条件输入
    c_in = torch.cat([c, null_token])  # 批量处理
    
    for t in tqdm(reversed(range(num_timesteps))):
        # 单次前向传播获得两个预测
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        noise_pred = model(x_in, t_in, c_in)
        
        # 分离条件和无条件预测
        noise_c, noise_u = noise_pred.chunk(2)
        
        # CFG组合
        noise_pred = noise_u + w * (noise_c - noise_u)
        
        # DDIM/DDPM采样步骤
        x = sampling_step(x, noise_pred, t, eta)
    
    return x
```

**内存优化版本**：
```python
def memory_efficient_cfg(model, x, t, c, w):
    # 使用gradient checkpointing
    with torch.no_grad():
        # 无条件预测
        noise_u = model(x, t, null_token)
    
    # 只对条件预测计算梯度（如果需要）
    noise_c = model(x, t, c)
    
    return noise_u + w * (noise_c - noise_u)
```

### 9.3.5 引导权重的选择

不同的 $w$ 值产生不同效果：

| $w$ 值 | 效果 | 典型应用 |
|--------|------|----------|
| 0 | 无条件生成 | 测试基线 |
| 1 | 标准条件生成 | 保守生成 |
| 3-5 | 轻度引导 | 平衡质量 |
| 7.5 | 标准引导 | 默认设置 |
| 10-20 | 强引导 | 高保真度 |
| >20 | 极端引导 | 可能过饱和 |

**动态引导调度**：
```python
def dynamic_guidance_weight(t, T):
    # 早期使用强引导，后期减弱
    progress = t / T
    w = w_start * (1 - progress) + w_end * progress
    return w
```

💡 **实践洞察：引导权重与条件类型**  
不同条件类型需要不同的引导强度。文本条件通常需要 w=7.5，而类别条件可能只需要 w=3。

### 9.3.6 理论分析

**1. 为什么CFG有效？**

CFG隐式地增强了条件的对数似然：
$$\log \tilde{p}(\mathbf{x}|\mathbf{c}) = \log p(\mathbf{x}|\mathbf{c}) + w\log p(\mathbf{c}|\mathbf{x})$$

这相当于在采样时重新加权条件的重要性。

**2. 与变分推断的联系**

CFG可以视为变分推断中的重要性加权：
- 提高高条件似然区域的采样概率
- 减少低条件似然区域的采样概率

**3. 几何解释**

在噪声预测空间中，CFG执行外推：
- 从无条件预测出发
- 沿着指向条件预测的方向移动
- 可能超越条件预测（当 $w > 1$）

<details>
<summary>**练习 9.3：CFG的深入分析**</summary>

探索CFG的各种特性和改进方法。

1. **引导权重调度**：
   - 实现线性、余弦、指数调度
   - 比较不同调度对生成质量的影响
   - 找出最优的调度策略

2. **条件dropout率研究**：
   - 测试 p_uncond ∈ [0.05, 0.1, 0.2, 0.5]
   - 分析对模型泛化的影响
   - 研究与引导权重的交互

3. **多条件CFG**：
   - 实现支持多个条件的CFG
   - 设计条件权重分配策略
   - 处理条件冲突

4. **理论扩展**：
   - 推导CFG的最优引导权重
   - 分析CFG对生成分布的影响
   - 研究CFG与其他采样方法的组合

</details>

### 9.3.7 高级技巧

**1. 负向提示（Negative Prompting）**

使用负条件来避免特定内容：
```python
def cfg_with_negative(x_t, t, c_pos, c_neg, w_pos=7.5, w_neg=7.5):
    noise_uncond = model(x_t, t, null_token)
    noise_pos = model(x_t, t, c_pos)
    noise_neg = model(x_t, t, c_neg)
    
    # 组合公式
    noise = noise_uncond + w_pos * (noise_pos - noise_uncond) - w_neg * (noise_neg - noise_uncond)
    return noise
```

**2. 多尺度引导**

在不同时间步使用不同的引导策略：
```python
def multiscale_cfg(t, T):
    if t > 0.8 * T:  # 早期：强语义引导
        return cfg_semantic(w=10)
    elif t > 0.3 * T:  # 中期：平衡引导
        return cfg_balanced(w=7.5)
    else:  # 后期：细节引导
        return cfg_detail(w=3)
```

**3. 自适应CFG**

根据预测的不确定性调整引导：
```python
def adaptive_cfg(noise_c, noise_u):
    # 计算预测差异
    diff = (noise_c - noise_u).abs().mean()
    
    # 差异大时减小引导权重
    w = base_weight * torch.exp(-alpha * diff)
    
    return noise_u + w * (noise_c - noise_u)
```

🔬 **研究方向：理论最优的引导**  
当前的线性组合是否是最优的？是否存在非线性的组合方式能产生更好的结果？这需要从信息论角度深入分析。

### 9.3.8 CFG的优势与局限

**优势**：
1. **简洁性**：不需要额外模型
2. **灵活性**：易于调整引导强度
3. **通用性**：适用于任何条件类型
4. **效果好**：实践中表现优异

**局限**：
1. **计算开销**：需要两次前向传播
2. **训练要求**：需要条件dropout
3. **分布偏移**：强引导可能导致分布偏离
4. **模式丢失**：可能降低多样性

### 9.3.9 与其他方法的比较

| 方法 | 额外模型 | 计算成本 | 灵活性 | 效果 |
|------|----------|----------|---------|------|
| 分类器引导 | 需要 | 高（梯度） | 中 | 好 |
| CFG | 不需要 | 中（2x前向） | 高 | 很好 |
| 原始条件 | 不需要 | 低 | 低 | 一般 |

🌟 **未来趋势：统一引导理论**  
CFG的成功启发了许多后续工作。未来可能出现统一的引导理论，涵盖所有条件生成方法，并提供最优引导策略的理论保证。

## 9.4 高级引导技术

### 9.4.1 多条件组合

现实应用中常需要同时满足多个条件。多条件组合的关键是如何平衡不同条件的影响。

**1. 线性组合**

最简单的方法是线性加权：
```python
def multi_condition_cfg(x_t, t, conditions, weights):
    # 无条件预测
    noise_uncond = model(x_t, t, null_token)
    
    # 组合多个条件
    combined_direction = 0
    for c, w in zip(conditions, weights):
        noise_c = model(x_t, t, c)
        combined_direction += w * (noise_c - noise_uncond)
    
    return noise_uncond + combined_direction
```

**2. 层次化条件**

不同条件在不同尺度起作用：
```python
class HierarchicalConditioning:
    def __init__(self):
        self.global_conditions = []  # 影响整体
        self.local_conditions = []   # 影响细节
    
    def apply(self, x_t, t, T):
        if t > 0.5 * T:  # 早期：全局条件
            return apply_conditions(self.global_conditions)
        else:  # 后期：局部条件
            return apply_conditions(self.local_conditions)
```

**3. 条件图结构**

使用图定义条件之间的关系：
```python
class ConditionalGraph:
    def __init__(self):
        self.nodes = {}  # 条件节点
        self.edges = {}  # 条件关系
    
    def propagate(self, x_t, t):
        # 根据图结构传播条件影响
        for node in topological_sort(self.nodes):
            parents = self.get_parents(node)
            node.update(parents, x_t, t)
```

### 9.4.2 负向提示技术

负向提示（Negative Prompting）是避免特定内容的强大工具。

**1. 基础负向提示**
```python
def negative_prompting(x_t, t, pos_prompt, neg_prompt, w_pos=7.5, w_neg=3.0):
    noise_uncond = model(x_t, t, null_token)
    noise_pos = model(x_t, t, pos_prompt)
    noise_neg = model(x_t, t, neg_prompt)
    
    # 朝正向移动，远离负向
    direction = w_pos * (noise_pos - noise_uncond) - w_neg * (noise_neg - noise_uncond)
    return noise_uncond + direction
```

**2. 多负向提示**
```python
def multi_negative_prompting(x_t, t, pos, neg_list, w_pos, w_neg_list):
    noise_uncond = model(x_t, t, null_token)
    noise_pos = model(x_t, t, pos)
    
    # 正向
    direction = w_pos * (noise_pos - noise_uncond)
    
    # 多个负向
    for neg, w_neg in zip(neg_list, w_neg_list):
        noise_neg = model(x_t, t, neg)
        direction -= w_neg * (noise_neg - noise_uncond)
    
    return noise_uncond + direction
```

**3. 自适应负向强度**
```python
def adaptive_negative(x_t, t, pos, neg):
    # 计算正负向的相似度
    sim = cosine_similarity(encode(pos), encode(neg))
    
    # 相似度高时增强负向强度
    w_neg = base_w_neg * (1 + alpha * sim)
    
    return negative_prompting(x_t, t, pos, neg, w_pos, w_neg)
```

💡 **实践技巧：负向提示的艺术**  
好的负向提示应该具体但不过于限制。例如，"低质量"比"模糊"更通用，"过度饱和"比"太亮"更精确。

### 9.4.3 动态引导强度

固定的引导强度可能不是最优的。动态调整可以获得更好的结果。

**1. 时间相关的引导**
```python
def time_dependent_guidance(t, T):
    # 余弦调度
    progress = t / T
    w = w_min + (w_max - w_min) * (1 + np.cos(np.pi * progress)) / 2
    return w
```

**2. 内容相关的引导**
```python
def content_aware_guidance(x_t, t, c):
    # 基于当前生成内容调整
    content_features = extract_features(x_t)
    
    # 检测是否需要强引导
    needs_strong_guidance = check_alignment(content_features, c)
    
    w = w_strong if needs_strong_guidance else w_normal
    return w
```

**3. 不确定性相关的引导**
```python
def uncertainty_based_guidance(model, x_t, t, c, n_samples=5):
    # 多次采样估计不确定性
    predictions = []
    for _ in range(n_samples):
        noise = model(x_t + small_noise(), t, c)
        predictions.append(noise)
    
    # 高不确定性时增强引导
    uncertainty = torch.stack(predictions).std(0).mean()
    w = w_base * (1 + beta * uncertainty)
    return w
```

### 9.4.4 ControlNet与适配器方法

ControlNet提供了精确的空间控制，通过额外的条件输入（如边缘图、深度图）引导生成。

**1. ControlNet基础架构**
```python
class ControlNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 复制基础模型的编码器
        self.control_encoder = copy.deepcopy(base_model.encoder)
        # 零初始化的投影层
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(...)) for _ in range(n_layers)
        ])
    
    def forward(self, x, t, c, control):
        # 处理控制信号
        control_feats = self.control_encoder(control, t)
        
        # 注入到基础模型
        for i, feat in enumerate(control_feats):
            base_feats[i] += self.zero_convs[i](feat)
```

**2. 多控制组合**
```python
def multi_control_generation(x_t, t, text, controls):
    # controls = {"depth": depth_map, "edge": edge_map, "pose": pose_map}
    
    # 基础文本引导
    noise_text = model(x_t, t, text)
    
    # 添加多个控制
    noise_combined = noise_text
    for control_type, control_input in controls.items():
        control_noise = control_nets[control_type](x_t, t, text, control_input)
        noise_combined += control_weights[control_type] * control_noise
    
    return noise_combined
```

**3. 适配器方法**

轻量级的条件注入：
```python
class Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.down = nn.Linear(in_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        
        # 零初始化
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x, condition):
        # 下投影
        h = self.down(condition)
        h = self.act(h)
        # 上投影
        h = self.up(h)
        # 残差连接
        return x + h
```

<details>
<summary>**练习 9.4：设计复杂的引导系统**</summary>

构建一个支持多种高级引导技术的系统。

1. **组合引导器**：
   - 实现支持文本、图像、布局的多模态引导
   - 设计条件优先级系统
   - 处理条件冲突

2. **动态调度器**：
   - 实现基于生成进度的引导调度
   - 根据生成质量自适应调整
   - 设计早停机制

3. **控制网络集成**：
   - 实现简化版ControlNet
   - 支持边缘、深度、分割图控制
   - 设计控制强度的自动调整

4. **评估系统**：
   - 设计条件一致性度量
   - 实现多样性评估
   - 构建自动化测试框架

</details>

### 9.4.5 引导技术的组合策略

**1. 级联引导**
```python
def cascade_guidance(x_t, t, conditions):
    # 逐步细化
    x = x_t
    for i, (condition, strength) in enumerate(conditions):
        x = apply_guidance(x, t, condition, strength)
        # 可选：中间去噪步骤
        if i < len(conditions) - 1:
            x = denoise_step(x, t)
    return x
```

**2. 注意力引导的引导**
```python
def attention_guided_cfg(x_t, t, c, attention_maps):
    # 使用注意力图调制引导强度
    noise_c = model(x_t, t, c)
    noise_u = model(x_t, t, null_token)
    
    # 空间变化的引导权重
    w_spatial = compute_spatial_weights(attention_maps)
    
    return noise_u + w_spatial * (noise_c - noise_u)
```

**3. 元引导**
```python
class MetaGuidance:
    def __init__(self):
        self.guidance_predictor = nn.Module()  # 预测最优引导策略
    
    def apply(self, x_t, t, context):
        # 预测当前最优引导参数
        guidance_params = self.guidance_predictor(x_t, t, context)
        
        # 应用预测的引导
        return apply_guidance_with_params(x_t, t, guidance_params)
```

🔬 **研究前沿：可学习的引导**  
能否训练一个网络来学习最优的引导策略？这可能需要元学习或强化学习方法。

### 9.4.6 实际应用中的权衡

**质量 vs 多样性**：
- 强引导提高质量但降低多样性
- 需要根据应用场景平衡

**计算成本**：
- 多条件组合增加推理时间
- ControlNet需要额外内存
- 需要考虑部署限制

**用户体验**：
- 过多的控制选项可能困扰用户
- 需要合理的默认值
- 提供预设模板

🌟 **最佳实践：渐进式复杂度**  
为用户提供分层的控制：基础用户使用简单文本，高级用户可以访问所有控制选项。

## 9.5 评估与优化

### 9.5.1 条件一致性度量

评估生成内容与条件的匹配程度是关键挑战。

**1. 分类准确率**

对于类别条件：
```python
def classification_accuracy(generated_images, target_classes, classifier):
    predictions = classifier(generated_images)
    accuracy = (predictions.argmax(1) == target_classes).float().mean()
    return accuracy
```

**2. CLIP Score**

对于文本条件：
```python
def clip_score(images, texts, clip_model):
    # 编码图像和文本
    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity.mean()
```

**3. 结构相似度**

对于空间控制（如ControlNet）：
```python
def structural_similarity(generated, control_signal):
    # 提取结构特征
    gen_edges = edge_detector(generated)
    
    # 计算相似度
    ssim = structural_similarity_index(gen_edges, control_signal)
    return ssim
```

**4. 语义一致性**

使用预训练模型评估语义对齐：
```python
def semantic_consistency(images, conditions, semantic_model):
    # 提取语义特征
    image_semantics = semantic_model.extract_semantics(images)
    condition_semantics = semantic_model.encode_conditions(conditions)
    
    # 计算语义距离
    distance = semantic_distance(image_semantics, condition_semantics)
    return 1 / (1 + distance)  # 转换为相似度
```

### 9.5.2 多样性与质量权衡

**1. 多样性度量**

```python
def diversity_metrics(generated_samples):
    metrics = {}
    
    # 特征空间多样性
    features = extract_features(generated_samples)
    metrics['feature_diversity'] = compute_variance(features)
    
    # 成对距离
    pairwise_dist = pdist(features)
    metrics['avg_distance'] = pairwise_dist.mean()
    
    # 覆盖度
    metrics['coverage'] = compute_coverage(features, reference_features)
    
    return metrics
```

**2. 质量-多样性前沿**

```python
def quality_diversity_frontier(model, conditions, guidance_weights):
    results = []
    
    for w in guidance_weights:
        # 生成样本
        samples = generate_with_guidance(model, conditions, w)
        
        # 评估
        quality = compute_quality(samples)
        diversity = compute_diversity(samples)
        
        results.append({
            'guidance_weight': w,
            'quality': quality,
            'diversity': diversity
        })
    
    return results
```

**3. 自动权衡选择**

```python
def auto_select_guidance(target_quality, target_diversity):
    # 基于历史数据拟合关系
    quality_fn = fit_quality_curve(historical_data)
    diversity_fn = fit_diversity_curve(historical_data)
    
    # 优化目标
    def objective(w):
        q = quality_fn(w)
        d = diversity_fn(w)
        return abs(q - target_quality) + abs(d - target_diversity)
    
    optimal_w = minimize(objective, x0=7.5)
    return optimal_w
```

### 9.5.3 引导失效的诊断

**1. 常见失效模式**

```python
class GuidanceFailureDetector:
    def __init__(self):
        self.failure_patterns = {
            'over_guidance': self.detect_over_guidance,
            'under_guidance': self.detect_under_guidance,
            'mode_collapse': self.detect_mode_collapse,
            'semantic_drift': self.detect_semantic_drift
        }
    
    def diagnose(self, samples, conditions):
        issues = []
        for name, detector in self.failure_patterns.items():
            if detector(samples, conditions):
                issues.append(name)
        return issues
```

**2. 过度引导检测**

```python
def detect_over_guidance(samples):
    # 检查饱和度
    saturation = compute_saturation(samples)
    if saturation > threshold_high:
        return True
    
    # 检查多样性
    diversity = compute_diversity(samples)
    if diversity < threshold_low:
        return True
    
    return False
```

**3. 语义漂移检测**

```python
def detect_semantic_drift(samples, conditions, steps=10):
    # 追踪生成过程中的语义变化
    semantic_trajectory = []
    
    for step in range(steps):
        intermediate = get_intermediate_result(step)
        semantics = extract_semantics(intermediate)
        semantic_trajectory.append(semantics)
    
    # 检测异常漂移
    drift = compute_trajectory_drift(semantic_trajectory)
    return drift > drift_threshold
```

💡 **调试技巧：可视化中间结果**  
保存并可视化不同时间步的中间结果，可以帮助识别引导在哪个阶段失效。

### 9.5.4 实际应用案例

**1. 文本到图像生成**

```python
class Text2ImagePipeline:
    def __init__(self, model, cfg_scale=7.5):
        self.model = model
        self.cfg_scale = cfg_scale
        self.negative_prompts = [
            "low quality", "blurry", "distorted"
        ]
    
    def generate(self, prompt, **kwargs):
        # 编码文本
        text_emb = self.encode_text(prompt)
        neg_emb = self.encode_text(self.negative_prompts)
        
        # CFG采样
        image = self.sample_with_cfg(
            text_emb, neg_emb, 
            self.cfg_scale, **kwargs
        )
        
        return image
```

**2. 图像编辑**

```python
class ImageEditingPipeline:
    def __init__(self, model, controlnet):
        self.model = model
        self.controlnet = controlnet
    
    def edit(self, image, edit_instruction, mask=None):
        # 提取控制信号
        control = self.extract_control(image)
        
        # 编码编辑指令
        instruction_emb = self.encode_instruction(edit_instruction)
        
        # 条件生成
        if mask is not None:
            # 局部编辑
            edited = self.local_edit(image, mask, instruction_emb, control)
        else:
            # 全局编辑
            edited = self.global_edit(image, instruction_emb, control)
        
        return edited
```

**3. 多模态生成**

```python
class MultiModalGenerator:
    def __init__(self, models):
        self.models = models
        self.fusion_module = CrossModalFusion()
    
    def generate(self, conditions):
        # conditions = {"text": ..., "audio": ..., "sketch": ...}
        
        # 编码各模态
        embeddings = {}
        for modality, condition in conditions.items():
            embeddings[modality] = self.models[modality].encode(condition)
        
        # 跨模态融合
        fused_condition = self.fusion_module(embeddings)
        
        # 生成
        output = self.sample_with_fusion(fused_condition)
        return output
```

<details>
<summary>**综合练习：构建生产级条件生成系统**</summary>

设计并实现一个完整的条件生成系统。

1. **系统架构**：
   - 模块化设计，支持插件式扩展
   - 统一的API接口
   - 错误处理和恢复机制

2. **功能实现**：
   - 支持多种条件类型
   - 自动参数优化
   - 批处理和流式处理

3. **性能优化**：
   - 模型量化和剪枝
   - 缓存机制
   - 并行化策略

4. **监控与评估**：
   - 实时质量监控
   - A/B测试框架
   - 用户反馈集成

5. **部署考虑**：
   - 容器化部署
   - 负载均衡
   - 版本管理

</details>

### 9.5.5 优化策略总结

**训练阶段优化**：
1. 合理的条件dropout率（通常0.1）
2. 多任务学习平衡
3. 数据增强策略
4. 课程学习（从简单到复杂）

**推理阶段优化**：
1. 引导权重的自适应调整
2. 提前停止策略
3. 批处理优化
4. 结果缓存

**系统级优化**：
1. 模型蒸馏
2. 量化感知训练
3. 硬件加速（GPU/TPU优化）
4. 分布式推理

### 9.5.6 未来发展方向

**1. 自适应引导**
- 基于内容的动态调整
- 学习型引导策略
- 用户偏好建模

**2. 统一框架**
- 多种引导方法的统一理论
- 可组合的引导模块
- 标准化评估体系

**3. 效率提升**
- 一次前向传播的引导
- 轻量级引导网络
- 边缘设备部署

🌟 **展望：智能引导系统**  
未来的条件生成系统将更加智能，能够理解用户意图，自动选择最优引导策略，并在生成过程中动态调整，实现真正的"所想即所得"。

## 本章小结

本章深入探讨了扩散模型的条件生成与引导技术，从基础的条件信息注入到高级的ControlNet方法。我们学习了：

- **条件扩散模型的基础**：各种条件注入方式和架构设计
- **分类器引导**：使用外部分类器梯度的经典方法
- **无分类器引导**：简洁高效的CFG技术
- **高级引导技术**：多条件组合、负向提示、动态引导等
- **评估与优化**：全面的评估体系和优化策略

这些技术使扩散模型从随机生成工具转变为精确可控的创作系统。下一章，我们将探讨潜在扩散模型，学习如何在压缩的潜在空间中高效地进行扩散建模。

[← 返回目录](index.md) | 第9章 / 共14章 | [下一章 →](chapter10.md)