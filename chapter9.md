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

在条件扩散模型中，我们的目标是建模条件分布 $p(\mathbf{x}|\mathbf{c})$ ，其中 $\mathbf{x}$ 是数据（如图像）， $\mathbf{c}$ 是条件信息（如类别标签、文本描述等）。

条件扩散过程定义为：
- **前向过程**： $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$ （与条件无关）
- **反向过程**： $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, \mathbf{c}), \sigma_t^2\mathbf{I})$

关键在于如何设计和训练条件去噪网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$ 。

### 9.1.2 条件信息的注入方式

**1. 拼接（Concatenation）**

最直接的方式是将条件信息与输入拼接。对于图像条件，可以在通道维度上拼接 $[\mathbf{x}_t, \mathbf{c}_{image}]$ 。对于向量条件，先通过条件编码器得到嵌入 $\mathbf{c}_{embed}$ ，然后扩展到空间维度后拼接。这种方法简单有效，但会增加第一层的参数量。

**2. 自适应归一化（Adaptive Normalization）**

通过条件信息调制归一化参数，包括AdaIN、AdaGN、AdaLN等变体。核心思想是：

$$\mathbf{h} = \gamma(\mathbf{c}) \odot \text{Normalize}(\mathbf{h}) + \beta(\mathbf{c})$$

其中 $\gamma$ 和 $\beta$ 是通过MLP从条件嵌入预测得到的缩放和偏移参数。

**3. 交叉注意力（Cross-Attention）**

特别适合序列条件（如文本）。查询（Query）来自图像特征，键（Key）和值（Value）来自文本编码：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{Q} = \mathbf{h}_{image}W_Q$ ， $\mathbf{K} = \mathbf{h}_{text}W_K$ ， $\mathbf{V} = \mathbf{h}_{text}W_V$ 。

**4. 特征调制（Feature-wise Modulation）**

FiLM（Feature-wise Linear Modulation）层通过条件信息缩放和偏移特征：

$$\mathbf{h}_{out} = \gamma(\mathbf{c}) \odot \mathbf{h}_{in} + \beta(\mathbf{c})$$

这种方法参数效率高，且能有效控制特征的激活模式。

🔬 **研究线索：最优注入位置**  
应该在网络的哪些层注入条件信息？早期层vs后期层？所有层vs特定层？这可能依赖于条件类型和任务。

### 9.1.3 架构设计原则

**1. 条件编码器设计**

不同类型的条件需要不同的编码器：
- **类别标签**：通过嵌入层映射到高维空间，再经过MLP进一步处理
- **文本**：使用预训练语言模型（如CLIP文本编码器、T5编码器）提取语义特征
- **图像**：预训练视觉模型（如ResNet、ViT）或专门设计的卷积编码器
- **音频**：先转换为频谱图，然后使用专门的时频编码器

**2. 多尺度条件注入**

在U-Net的不同分辨率层级注入条件信息，使得：
- 高分辨率层获得细节控制（如纹理、边缘）
- 中分辨率层获得结构控制（如物体形状）
- 低分辨率层获得语义控制（如整体布局）

每个下采样块和上采样块都接收条件信息： $\mathbf{h}_i = f_i(\mathbf{h}_{i-1}, t, \mathbf{c})$

**3. 时间-条件交互**

时间步 $t$ 和条件信息 $\mathbf{c}$ 可能需要交互建模。一种常见方法是联合编码：

$$\mathbf{e}_{joint} = \text{MLP}(\mathbf{e}_t + \mathbf{e}_c)$$

其中 $\mathbf{e}_t$ 是时间嵌入， $\mathbf{e}_c$ 是条件嵌入。这种交互允许模型根据去噪阶段调整条件的影响方式。

### 9.1.4 训练策略

**1. 条件dropout**

随机丢弃条件信息，训练模型同时处理条件和无条件生成。在训练时，以概率 $p_{uncond}$ 将条件 $\mathbf{c}$ 替换为空条件 $\varnothing$ ：

$$\mathbf{c}_{train} = \begin{cases}
\mathbf{c} & \text{with probability } 1-p_{uncond} \\
\varnothing & \text{with probability } p_{uncond}
\end{cases}$$

然后正常计算去噪损失：
$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{train})\|^2\right]$$

这是无分类器引导的基础，使模型能够同时进行条件和无条件生成。

**2. 条件增强**

对条件信息进行数据增强以提高泛化能力：
- **文本条件**：同义词替换、句子改写、随机删除/添加修饰词
- **图像条件**：几何变换（旋转、缩放）、颜色扰动、随机裁剪
- **类别条件**：标签平滑、Mixup混合策略

**3. 多任务学习**

同时训练多种条件类型，总损失为各任务损失的加权和：

$$\mathcal{L}_{total} = \mathcal{L}_{uncond} + \lambda_1\mathcal{L}_{class} + \lambda_2\mathcal{L}_{text} + \lambda_3\mathcal{L}_{image}$$

其中 $\lambda_i$ 是各任务的权重系数。

💡 **实践技巧：条件缩放**  
不同条件的强度可能需要不同的缩放。使用可学习的缩放因子： $\mathbf{c}_{scaled} = s_c \cdot \mathbf{c}$ ，其中 $s_c$ 是可学习参数。

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

**内存优化策略**：
- **梯度检查点**：对计算密集但内存占用大的条件块使用 `torch.utils.checkpoint`
- **混合精度训练**：条件编码器使用FP16，关键层保持FP32
- **动态批处理**：根据条件复杂度动态调整批大小

**计算优化技巧**：
- **条件编码缓存**：对于离散条件（如类别），缓存编码结果
- **批量编码**：将相同类型的条件批量处理
- **编码器共享**：多个条件类型共享底层特征提取器

**数值稳定性保障**：
- **条件归一化**： $\mathbf{c}_{encoded} = s \cdot \mathbf{c}_{encoded} / \|\mathbf{c}_{encoded}\|_2$
- **残差缩放**：条件注入时使用小的初始权重
- **梯度裁剪**：防止条件相关的梯度爆炸

## 9.2 分类器引导（Classifier Guidance）

### 9.2.1 理论推导

分类器引导的核心思想是使用外部分类器的梯度来引导扩散模型的采样过程。我们从贝叶斯规则开始：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{c}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t)$$

第一项是无条件分数，第二项是分类器的梯度。这给出了条件采样的更新规则：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

其中 $p_\phi(\mathbf{c}|\mathbf{x}_t)$ 是在噪声数据上训练的分类器。

### 9.2.2 噪声条件分类器

关键挑战是训练一个能在所有噪声水平 $t$ 上工作的分类器。

**训练目标**：
$$\mathcal{L}_{classifier} = \mathbb{E}_{t \sim \mathcal{U}[1,T], \mathbf{x}_0 \sim p_{data}, \boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})} \left[-\log p_\phi(\mathbf{c}|\mathbf{x}_t, t)\right]$$

其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 是加噪后的样本。

**分类器架构要求**：
1. **时间条件**：通过时间嵌入了解当前噪声水平，通常使用正弦编码
2. **鲁棒性**：在高噪声下仍能提取有用特征，需要强大的特征提取能力
3. **梯度质量**：提供平滑且有意义的梯度信号用于引导

**架构设计原则**：
- 使用与扩散模型相似的骨干网络（如U-Net）
- 在多个尺度提取特征以增强鲁棒性
- 使用残差连接和归一化层稳定训练

### 9.2.3 引导强度与采样

引导强度 $s$ 控制条件的影响程度：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s\sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

- $s = 0$ ：无条件生成
- $s = 1$ ：标准条件生成
- $s > 1$ ：强化条件，可能降低多样性
- $s < 0$ ：负向引导，远离条件

**采样算法流程**：

1. 从标准高斯分布采样初始噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. 对于每个时间步 $t = T, T-1, ..., 1$ ：
   - 使用扩散模型预测无条件噪声： $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - 计算分类器对数概率的梯度： $\nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$
   - 组合得到引导后的噪声预测： $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta - s\sqrt{1-\bar{\alpha}_t}\nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$
   - 执行去噪步骤得到 $\mathbf{x}_{t-1}$

**梯度计算细节**：
- 需要对 $\mathbf{x}_t$ 启用梯度计算
- 通过自动微分计算分类器输出相对于输入的梯度
- 计算完成后关闭梯度计算以节省内存

### 9.2.4 梯度计算的实践考虑

**1. 梯度缩放**

不同时间步的梯度量级差异很大，需要自适应缩放。根据噪声水平调整：

$$\nabla_{scaled} = \frac{1}{\sqrt{1-\bar{\alpha}_t}} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

这种缩放补偿了不同噪声水平下的信号强度差异。

**2. 梯度裁剪**

防止梯度爆炸，对梯度进行归一化：

$$\nabla_{clipped} = \frac{\nabla}{\max(1, \|\nabla\|_2 / \lambda)}$$

其中 $\lambda$ 是梯度范数的阈值。

**3. 多步梯度累积**

通过对带噪声扰动的输入计算多次梯度并平均，获得更稳定的梯度估计：

$$\nabla_{stable} = \frac{1}{N} \sum_{i=1}^N \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t + \sigma\boldsymbol{\epsilon}_i)$$

其中 $\boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \mathbf{I})$ ， $\sigma$ 是小的噪声尺度。

💡 **实践技巧：温度调节**  
对分类器输出使用温度缩放可以控制引导的锐度： $p_\phi(\mathbf{c}|\mathbf{x}_t) \propto \exp(\text{logits}/\tau)$ ，其中 $\tau$ 是温度参数。

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

只在特定时间范围内应用引导，避免在噪声过大或过小时的不良影响：

$$\tilde{\boldsymbol{\epsilon}} = \begin{cases}
\boldsymbol{\epsilon}_\theta - s\sqrt{1-\bar{\alpha}_t}\nabla \log p_\phi(\mathbf{c}|\mathbf{x}_t) & \text{if } T_{start} < t < T_{end} \\
\boldsymbol{\epsilon}_\theta & \text{otherwise}
\end{cases}$$

**2. 局部引导**

使用空间掩码 $\mathbf{M}$ 只对图像的特定区域应用引导：

$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta - s\sqrt{1-\bar{\alpha}_t}(\mathbf{M} \odot \nabla \log p_\phi(\mathbf{c}|\mathbf{x}_t))$$

这允许精细的空间控制。

**3. 多分类器集成**

组合多个分类器提供更稳健的引导：

$$\nabla \log p_{ensemble}(\mathbf{c}|\mathbf{x}_t) = \sum_{i=1}^K w_i \nabla \log p_{\phi_i}(\mathbf{c}|\mathbf{x}_t)$$

其中 $w_i$ 是各分类器的权重。
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

CFG通过隐式估计 $\nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t)$ ：
$$\nabla_{\mathbf{x}_t} \log p(\mathbf{c}|\mathbf{x}_t) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{c}) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$$

### 9.3.2 训练策略：条件Dropout

关键创新是在训练时随机丢弃条件。具体过程：

1. 对于每个训练样本，以概率 $p_{uncond}$ 将条件替换为空条件 $\varnothing$
2. 使用修改后的条件进行标准扩散模型训练
3. 损失函数保持不变： $\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{masked})\|^2]$

其中：
$$\mathbf{c}_{masked} = \begin{cases}
\mathbf{c} & \text{with probability } 1-p_{uncond} \\
\varnothing & \text{with probability } p_{uncond}
\end{cases}$$

这使得单个模型能够同时学习条件分布 $p(\mathbf{x}|\mathbf{c})$ 和边缘分布 $p(\mathbf{x})$ 。

**空条件的表示**：
- 对于文本条件：使用空字符串或特殊的 `[NULL]` token
- 对于类别条件：使用额外的"无条件"类别
- 对于图像条件：使用零张量或学习的空嵌入

### 9.3.3 采样公式

CFG的采样公式：
$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1 + w)\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - w\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$$

其中：
- $w$ ：引导权重（guidance weight）
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$ ：条件预测
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$ ：无条件预测

这可以重写为：
$$\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

显示了从无条件预测出发，朝条件方向移动的解释。

### 9.3.4 实现细节

**高效采样策略**：

为了避免两次独立的模型前向传播，可以批量处理条件和无条件预测：

1. 将输入 $\mathbf{x}_t$ 复制一份： $[\mathbf{x}_t, \mathbf{x}_t]$
2. 准备条件批次： $[\mathbf{c}, \varnothing]$
3. 单次前向传播得到： $[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}), \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$
4. 应用CFG公式组合预测

**内存优化**：
- 对于大模型，可以顺序计算条件和无条件预测
- 使用梯度检查点减少激活内存
- 在低精度（FP16）下运行推理
**采样算法完整流程**：
1. 初始化： $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. 对每个时间步 $t = T, T-1, ..., 1$ ：
   - 计算条件和无条件预测
   - 应用CFG公式： $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$
   - 执行采样步骤（DDPM或DDIM）

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

可以使用时变的引导权重，例如线性插值：
$$w(t) = w_{start} \cdot (1 - t/T) + w_{end} \cdot (t/T)$$

其中早期使用较强的引导（ $w_{start}$ 较大），后期逐渐减弱（ $w_{end}$ 较小），帮助模型在保持条件忠实度的同时提高细节质量。

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
- 可能超越条件预测（当 $w > 1$ ）

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

使用负条件来避免特定内容的生成。组合公式为：
$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w_{pos} [\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{pos}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)] - w_{neg} [\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{neg}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

其中 $\mathbf{c}_{pos}$ 是期望的条件， $\mathbf{c}_{neg}$ 是要避免的条件， $w_{pos}$ 和 $w_{neg}$ 分别控制正向和负向引导的强度。

**2. 多尺度引导**

在不同时间步使用不同的引导策略。例如：
- 早期阶段（ $t > 0.8T$ ）：使用强语义引导（ $w=10$ ），确保整体结构正确
- 中期阶段（ $0.3T < t \leq 0.8T$ ）：使用平衡引导（ $w=7.5$ ）
- 后期阶段（ $t \leq 0.3T$ ）：使用较弱引导（ $w=3$ ），保留细节多样性

**3. 自适应CFG**

根据预测的不确定性调整引导强度。一种方法是基于条件和无条件预测的差异：

$$w_{adaptive} = w_{base} \cdot \exp(-\alpha \cdot ||\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)||)$$

当预测差异较大时，说明模型对条件的理解存在不确定性，此时减小引导权重可以避免过度放大误差。

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

最简单的方法是对多个条件进行线性加权：

$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + \sum_{i=1}^{n} w_i [\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_i) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

其中 $\mathbf{c}_i$ 是第 $i$ 个条件， $w_i$ 是对应的权重。权重需要满足 $\sum_i w_i = 1$ 以保持引导的整体强度。

**2. 层次化条件**

不同条件在不同尺度起作用。层次化条件策略可以将条件分为：
- 全局条件：影响整体结构和布局
- 局部条件：影响细节和纹理

在早期阶段（ $t > 0.5T$ ）应用全局条件，后期阶段（ $t \leq 0.5T$ ）应用局部条件。这种方法可以确保先建立正确的整体结构，再添加细节。

**3. 条件图结构**

使用图结构定义条件之间的依赖关系。每个条件节点可以有父节点，其影响传播遵循拓扑排序。这样可以实现复杂的条件依赖，如：“如果有人物，则添加背景”或“风格受主题影响”等。

### 9.4.2 负向提示技术

负向提示（Negative Prompting）是避免特定内容的强大工具。

**1. 基础负向提示**

组合正向和负向条件的公式：
$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w_{pos}[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{pos}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)] - w_{neg}[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{neg}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

这个公式使得生成朝着正向条件移动，同时远离负向条件。

**2. 多负向提示**

当需要避免多个不希望的属性时，可以使用多负向提示：
$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w_{pos}[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{pos}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)] - \sum_{i=1}^{n} w_{neg,i}[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{neg,i}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$$

每个负向条件可以有不同的权重 $w_{neg,i}$ 。

**3. 自适应负向强度**

根据正负向条件的相似度调整负向强度：
$$w_{neg} = w_{neg,base} \cdot (1 + \alpha \cdot \text{sim}(\mathbf{c}_{pos}, \mathbf{c}_{neg}))$$

其中 $\text{sim}(\cdot,\cdot)$ 是余弦相似度。当正负向条件相似度高时（如“高质量”与“低质量”），增强负向强度更有必要。

💡 **实践技巧：负向提示的艺术**  
好的负向提示应该具体但不过于限制。例如，"低质量"比"模糊"更通用，"过度饱和"比"太亮"更精确。

### 9.4.3 动态引导强度

固定的引导强度可能不是最优的。动态调整可以获得更好的结果。

**1. 时间相关的引导**

使用余弦调度的引导权重：
$$w(t) = w_{min} + (w_{max} - w_{min}) \cdot \frac{1 + \cos(\pi \cdot t/T)}{2}$$

这种调度在初期和末期使用较弱的引导，中期使用较强的引导，形成平滑的过渡。

**2. 内容相关的引导**

基于当前生成内容与条件的对齐度调整引导强度。可以提取中间特征并计算与条件的对齐分数：
$$w = \begin{cases}
w_{strong} & \text{if } \text{alignment}(\mathbf{x}_t, \mathbf{c}) < \tau \\
w_{normal} & \text{otherwise}
\end{cases}$$

其中 $\tau$ 是对齐阈值。

**3. 不确定性相关的引导**

通过多次采样估计模型预测的不确定性，并据此调整引导强度。当不确定性高时，减小引导强度以避免放大误差。
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

ControlNet通过复制基础模型的编码器结构，并使用零初始化的投影层将控制信号注入到基础模型中。关键设计点：
- 控制编码器：复制基础模型的编码器权重
- 零卷积：使用零初始化的卷积层确保训练初期不影响基础模型
- 特征注入：在多个层级将控制特征添加到基础特征中

**2. 多控制组合**

同时使用多个控制信号（如深度图、边缘图、姿态图）时，可以通过加权组合各个控制网络的输出：
$$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{text} + \sum_{i} w_i \cdot \boldsymbol{\epsilon}_{control_i}$$

其中 $\boldsymbol{\epsilon}_{text}$ 是文本引导的预测， $\boldsymbol{\epsilon}_{control_i}$ 是第 $i$ 个控制网络的输出， $w_i$ 是对应的权重。

**3. 适配器方法**

适配器（Adapter）是一种轻量级的条件注入方法，使用下投影-激活-上投影的结构：
$$\mathbf{h} = \mathbf{x} + \text{UP}(\text{GELU}(\text{DOWN}(\mathbf{c})))$$

其中：
- $\text{DOWN}$ ：降维投影， $\mathbb{R}^{d} \to \mathbb{R}^{d'}$ ， $d' < d$
- $\text{UP}$ ：升维投影， $\mathbb{R}^{d'} \to \mathbb{R}^{d}$ ，零初始化
- $\text{GELU}$ ：非线性激活函数

这种设计保持了参数效率，同时通过零初始化确保训练稳定性。

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

级联引导通过逐步应用不同的条件来细化生成结果。每个阶段应用一个条件，并可选择地在阶段之间执行部分去噪：
$$\mathbf{x}^{(i+1)} = \text{ApplyGuidance}(\mathbf{x}^{(i)}, t, \mathbf{c}_i, w_i)$$

这种方法特别适合处理层次化的条件，如先应用全局布局条件，再应用局部细节条件。

**2. 注意力引导的引导**

使用模型内部的注意力图来调制引导强度。在注意力集中的区域使用更强的引导，在其他区域保持较弱的引导，以保护细节和多样性。
    noise_u = model(x_t, t, null_token)
    
    # 空间变化的引导权重
    w_spatial = compute_spatial_weights(attention_maps)
    
    return noise_u + w_spatial * (noise_c - noise_u)


**3. 元引导**

元引导是一种高级技术，使用学习的模型来预测最优引导策略：
- **引导预测器**：一个神经网络，根据当前状态预测最佳引导参数
- **上下文意识**：根据不同的生成上下文调整引导策略
- **动态适应**：在生成过程中实时调整引导参数

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

对于类别条件，可以使用预训练的分类器评估生成图像的类别一致性：
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max_j p(y_j|\mathbf{x}_i) = c_i]$$

其中 $p(y_j|\mathbf{x}_i)$ 是分类器对生成图像 $\mathbf{x}_i$ 的预测概率， $c_i$ 是目标类别。

**2. CLIP Score**

对于文本条件，使用CLIP模型计算图像-文本的对齐度：
$$\text{CLIP Score} = \mathbb{E}[\cos(\mathbf{f}_I(\mathbf{x}), \mathbf{f}_T(\mathbf{c}))]$$

其中 $\mathbf{f}_I$ 和 $\mathbf{f}_T$ 分别是CLIP的图像和文本编码器， $\cos(\cdot,\cdot)$ 是余弦相似度。更高的CLIP分数表示更好的图像-文本对齐。

**3. 结构相似度**

对于空间控制（如ControlNet），可以使用结构相似性指标（SSIM）或边缘检测来评估：
$$\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

其中 $\mu$ 是均值， $\sigma$ 是标准差， $\sigma_{xy}$ 是协方差， $c_1, c_2$ 是稳定常数。

**4. 语义一致性**

使用预训练模型评估语义对齐。通过提取图像和条件的语义特征，计算它们之间的距离：
$$\text{Semantic Consistency} = \frac{1}{1 + d(\mathbf{s}_I, \mathbf{s}_C)}$$

其中 $\mathbf{s}_I$ 是图像的语义特征， $\mathbf{s}_C$ 是条件的语义特征， $d(\cdot,\cdot)$ 是距离度量（如L2距离）。

### 9.5.2 多样性与质量权衡

**1. 多样性度量**

评估生成样本的多样性可以使用多种指标：
- **特征空间多样性**：计算生成样本在特征空间中的方差
- **成对距离**：计算所有样本对之间的平均距离
- **覆盖度**：评估生成分布对参考分布的覆盖程度

**2. 质量-多样性前沿**

通过测试不同的引导权重，可以绘制质量-多样性的权衡曲线。通常：
- 低引导权重：高多样性、低质量
- 高引导权重：低多样性、高质量
- 最佳点：在两者之间找到平衡

**3. 自动权衡选择**

可以基于历史数据拟合质量和多样性与引导权重的关系，然后根据目标质量和多样性自动选择最佳引导权重：
$$w^* = \arg\min_w |Q(w) - Q_{target}| + |D(w) - D_{target}|$$

其中 $Q(w)$ 和 $D(w)$ 分别是质量和多样性关于引导权重的函数。

### 9.5.3 引导失效的诊断

**1. 常见失效模式**

条件引导可能出现的失效模式包括：
- **过度引导**：生成结果过于饫和或失真
- **引导不足**：条件与生成内容不匹配
- **模式崩塌**：所有生成结果趋同
- **语义漂移**：生成过程中偏离原始条件

可以设计一个诊断系统来自动检测这些失效模式。

**2. 过度引导检测**

检测过度引导的指标包括：
- **饱和度异常**：检查图像的颜色饱和度是否过高
- **多样性下降**：评估多个生成样本之间的差异是否过小
- **细节丢失**：检查高频信息是否被过度平滑

**3. 语义漂移检测**

[代码块已移除]

💡 **调试技巧：可视化中间结果**  
保存并可视化不同时间步的中间结果，可以帮助识别引导在哪个阶段失效。

### 9.5.4 实际应用案例

**1. 文本到图像生成**

[代码实现已转换为数学公式和文字描述]

**2. 图像编辑**

图像编辑管道的关键组件：
- **控制信号提取**：从原始图像中提取结构信息（如边缘、深度）
- **编辑指令编码**：将文本编辑指令转换为条件向量
- **局部/全局编辑**：根据是否有掩码选择编辑模式
- **条件生成**：结合ControlNet保持结构一致性

**3. 多模态生成**

多模态生成系统的核心要素：
- **模态编码器**：每个模态（文本、音频、草图等）需要专门的编码器
- **跨模态融合**：将不同模态的条件融合成统一表示
- **权重分配**：不同模态可能需要不同的影响权重
- **一致性保持**：确保多个模态条件不会产生冲突

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