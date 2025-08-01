[← 上一章](chapter1.md) | 第2章 / 共14章 | [下一章 →](chapter3.md)

# 第2章：神经网络架构：U-Net与ViT

扩散模型的成功离不开强大的神经网络架构。有趣的是，扩散模型并没有发明全新的网络结构，而是巧妙地借用了计算机视觉领域的两个里程碑式架构：U-Net和Vision Transformer (ViT)。本章将追溯这两种架构的历史发展，理解它们的设计初衷，并剖析它们为何能与扩散模型的去噪任务完美契合。这种“历史的巧合”不仅展示了深度学习领域知识迁移的魅力，也为我们设计未来更高效的生成模型提供了深刻的启示。

## 2.1 从图像分割到去噪：U-Net的历史演变

### 2.1.1 生物医学图像的挑战与U-Net的诞生

2015年，深度学习正在快速改变计算机视觉的格局。然而，在医学图像分析领域，研究者们面临着独特的挑战：标注数据极其稀缺（医学专家的时间宝贵），图像分辨率高，细节至关重要，且分割边界往往模糊不清。当时流行的全卷积网络（FCN）虽然在自然图像分割上取得了成功，但在医学图像上的表现并不理想。

正是在这样的背景下，来自弗莱堡大学的Olaf Ronneberger、Philipp Fischer和Thomas Brox提出了U-Net。他们的灵感来自一个朴素但深刻的观察：医学图像分割需要两种看似矛盾的能力——既要理解全局的语义信息（这是什么器官？），又要精确定位每个像素（边界在哪里？）。传统的编码器-解码器架构在解码过程中丢失了太多空间信息，而U-Net通过引入跳跃连接，优雅地解决了这个问题。

> **定义：历史脉络的详细时间线**
> - **2012-2014年**：全卷积网络（FCN）的兴起，Long等人证明了CNN可以进行像素级预测，但在细节保留上存在不足。
> - **2015年5月**：U-Net在ISBI细胞追踪挑战赛中首次亮相，以大幅领先的成绩震撼了医学图像界。原始论文展示了仅用30张训练图像就能达到出色性能的能力。
> - **2016-2017年**：U-Net的变体开始涌现——3D U-Net用于体积数据、V-Net引入残差连接、Attention U-Net加入注意力机制。每个变体都针对特定应用场景进行了优化。
> - **2017-2019年**：U-Net架构被广泛应用于各种像素级预测任务，从卫星图像分析到自动驾驶的道路分割，成为该领域的事实标准。其PyTorch和TensorFlow实现成为GitHub上最受欢迎的开源项目之一。
> - **2020年6月**：Ho等人发表DDPM论文，首次将U-Net用作扩散模型的去噪网络。他们的关键洞察是：去噪本质上也是一个像素到像素的映射问题。
> - **2021年**：Dhariwal和Nichol在论文《Diffusion Models Beat GANs on Image Synthesis》中提出了改进的U-Net架构（ADM），加入了自注意力层和自适应归一化，将扩散模型的生成质量推向新高度。
> - **2022年**：Stable Diffusion的发布让U-Net架构走向大众。其高效的潜在空间U-Net设计使得高质量图像生成首次可以在消费级GPU上运行。
> - **2023年至今**：U-Net继续演进，如加入更多的条件机制（ControlNet）、与Transformer混合（U-ViT）、针对视频生成的时空U-Net等。

### 2.1.2 从分割到去噪：任务的本质相似性

为什么一个为医学图像分割设计的架构能够如此完美地适用于扩散模型？答案隐藏在这两个看似不同的任务的数学本质中。

**图像分割的数学表述**：给定输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$，预测每个像素的类别标签 $\mathbf{y} \in \{0,1,...,C-1\}^{H \times W}$。这是一个确定性的映射：$f_{\text{seg}}: \mathbb{R}^{H \times W \times 3} \rightarrow \{0,1,...,C-1\}^{H \times W}$。

**扩散模型去噪的数学表述**：给定带噪声的图像 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$，预测噪声 $\boldsymbol{\epsilon} \in \mathbb{R}^{H \times W \times 3}$。这同样是一个确定性的映射：$f_{\text{denoise}}: \mathbb{R}^{H \times W \times 3} \times \mathbb{R} \rightarrow \mathbb{R}^{H \times W \times 3}$（额外的输入是时间步$t$）。

两者的共同点在于：
1. **像素级预测**：两个任务都需要为输入的每个空间位置产生一个输出。
2. **多尺度信息融合**：分割需要结合局部纹理（判断边界）和全局语义（识别对象）；去噪需要结合局部细节（保留纹理）和全局结构（理解内容）。
3. **空间对应关系**：输出的空间结构必须与输入严格对应，这正是跳跃连接所保证的。

### 2.1.3 U-Net设计哲学的深层洞察

U-Net的成功不仅仅是技术上的胜利，更体现了深刻的设计哲学。让我们深入剖析其核心设计原则：

**1. 对称性的美学与功能**
U-Net的U形结构不仅在视觉上优雅，更重要的是体现了信息处理的对称性。编码器逐步压缩空间维度、提取抽象特征的过程，与解码器逐步恢复空间维度、重建具体细节的过程，形成了完美的镜像。这种对称性在扩散模型中获得了新的诠释：编码器理解"现在有什么噪声"，解码器决定"如何去除这些噪声"。

**2. 跳跃连接：信息高速公路**
原始的编码器-解码器架构存在一个致命弱点：信息瓶颈。当特征图被压缩到最小尺寸时（如原始尺寸的1/32），大量的空间信息已经无可挽回地丢失了。U-Net的跳跃连接就像在山谷两侧架起的桥梁，让高分辨率的信息可以直接"跳过"瓶颈，到达需要它的地方。

在扩散模型的语境下，这一点尤为关键。考虑去噪过程的两个极端情况：
- 当噪声很大时（$t$接近$T$），模型主要依赖瓶颈处的全局信息来重建大致结构。
- 当噪声很小时（$t$接近0），模型主要依赖跳跃连接传递的局部信息来恢复细节。

**3. 计算效率的权衡艺术**
U-Net的金字塔结构带来了计算上的巨大优势。大部分的计算（自注意力、复杂的卷积）发生在低分辨率的特征图上，而高分辨率层只进行相对简单的操作。这种设计使得U-Net可以在有限的计算资源下处理高分辨率图像，这也是为什么Stable Diffusion能够在个人电脑上运行的关键因素之一。

### 2.1.4 U-Net变体的百花齐放

U-Net的基本思想激发了无数的变体和改进。每一个成功的变体都代表了对特定问题的深刻理解：

- **3D U-Net (2016)**：将2D卷积替换为3D卷积，用于处理CT、MRI等体积数据。关键创新是各向异性的卷积核（如3×3×1），以处理医学图像中常见的各向异性分辨率。

- **Attention U-Net (2018)**：在跳跃连接中加入注意力门控（attention gates），让模型学习"哪些跳跃连接的信息是重要的"。这在医学图像中特别有用，因为病变区域往往只占整个图像的一小部分。

- **U-Net++ (2018)**：通过密集的跳跃连接创建了一个"嵌套"的U-Net结构，让解码器可以从多个尺度的编码特征中选择信息。这种设计虽然增加了计算量，但在某些任务上显著提升了性能。

- **TransUNet (2021)**：将CNN编码器的瓶颈部分替换为Transformer，结合了CNN的局部特征提取能力和Transformer的全局建模能力。这为后来的混合架构铺平了道路。

### 2.1.5 为什么是U-Net？扩散模型的架构选择

当Ho等人在2020年为DDPM选择网络架构时，他们面临着多种选择：ResNet、VGG、甚至当时新兴的Vision Transformer。为什么最终选择了U-Net？

**1. 归纳偏置的匹配**
扩散模型的去噪任务具有特殊的性质：输出必须与输入在空间上严格对齐。U-Net的架构天然地保证了这一点，而其他架构（如将图像展平后输入全连接网络）则会破坏这种空间结构。

**2. 多时间尺度的处理能力**
在扩散过程的不同阶段，去噪的重点是不同的：
- 早期（高噪声）：需要重建全局结构和语义内容
- 中期：需要恢复中等尺度的形状和纹理  
- 后期（低噪声）：需要精修局部细节和清晰度

U-Net的多尺度特性完美匹配了这种需求，不同的层级自然地专注于不同尺度的特征。

**3. 实践中的鲁棒性**
医学图像分割领域的严苛要求（小数据集、高精度需求）锻造了U-Net的鲁棒性。这种鲁棒性在扩散模型的训练中同样重要，因为去噪网络需要处理从纯噪声到清晰图像的整个谱系。

<details>
<summary><strong>深入研究：U-Net的理论基础与未来方向</strong></summary>

**1. 信息论视角**
从信息论的角度，U-Net的跳跃连接可以被理解为创建了多个信息传输通道，每个通道具有不同的"带宽"（分辨率）。这种设计最小化了信息在网络中传输时的损失。研究方向：
- 如何定量分析不同跳跃连接的信息流量？
- 是否存在最优的跳跃连接模式？
- 能否设计自适应的跳跃连接，根据输入内容动态调整？

**2. 神经架构搜索（NAS）在U-Net的应用**
虽然U-Net的基本结构已经被证明非常有效，但其具体的配置（深度、宽度、跳跃连接的位置等）仍有优化空间。研究方向：
- 如何为特定的数据集自动搜索最优的U-Net配置？
- 能否设计一个"元U-Net"，根据输入动态调整其结构？
- 如何在保持U-Net核心思想的同时，探索更激进的架构创新？

**3. U-Net与其他范式的融合**
U-Net代表了一种特定的归纳偏置，但它并非唯一的选择。研究方向：
- 如何将U-Net与图神经网络（GNN）结合，处理非规则的空间结构？
- 能否设计一个统一的框架，在U-Net和Transformer之间平滑过渡？
- 如何将物理约束（如守恒定律）直接编码到U-Net的架构中？

</details>

## 2.2 U-Net架构详解

### 2.2.1 现代U-Net：为扩散模型重新设计

当DDPM的作者们在2020年选择U-Net作为去噪网络时，他们面临着与原始分割任务完全不同的需求。因此，一个为扩散模型“现代化”的U-Net诞生了，它融合了自2015年以来深度学习架构的诸多进展。

> **定义：扩散U-Net的关键改进**
> | 组件 | 原始U-Net (2015) | 扩散U-Net (2020+) |
> | :--- | :--- | :--- |
> | **卷积类型** | Valid卷积 (无padding) | Same卷积 (保持尺寸) |
> | **归一化** | 无 (或后期加入BatchNorm) | GroupNorm (小批量稳定) |
> | **激活函数** | ReLU | SiLU / Swish (更平滑) |
> | **残差连接** | 无 | 每个块内部都有 (类ResNet) |
> | **注意力机制** | 无 | 多分辨率自注意力 |
> | **条件机制** | 无需条件 | 时间嵌入 (必需) |

让我们深入理解几个关键改进：

#### 1. 残差块 (ResNet Block)
现代U-Net的基本构建单元不再是简单的卷积层，而是借鉴了ResNet的残差块。一个典型的块流程如下：
1.  输入 `x` 首先通过 `GroupNorm` 和 `SiLU` 激活函数。
2.  经过一个3x3的 `Conv2d` 层。
3.  再次通过 `GroupNorm` 和 `SiLU`。
4.  经过第二个3x3的 `Conv2d` 层。
5.  将处理后的结果与原始输入 `x` 相加（残差连接）。

#### 2. 时间嵌入注入 (Time Embedding)
时间步 `t` 的信息至关重要。它通常通过一个小型MLP从正弦编码转换为嵌入向量，然后通过自适应归一化层（Adaptive Group Normalization, AdaGN）注入到每个残差块中。其核心思想是调制残差块的统计特性：
`h_out = GroupNorm(h_in) * (1 + scale(t)) + shift(t)`
其中 `scale(t)` 和 `shift(t)` 是从时间嵌入向量线性变换得到的。

#### 3. 自注意力 (Self-Attention)
为了捕获长程依赖关系，自注意力机制被引入到U-Net中。但由于其计算复杂度与像素数的平方成正比，它通常只在特征图分辨率较低的层级（如16x16或8x8）使用，以在计算效率和全局建模能力之间取得平衡。

### 2.2.2 采样方式的演进：从池化到可学习的卷积

"如何正确地降低和恢复分辨率"是U-Net设计的核心问题之一，其演进过程反映了深度学习架构设计的范式转变。这个问题看似简单，实则深刻影响着模型的表现力和生成质量。

#### 下采样的哲学：信息压缩的艺术

下采样不仅仅是减少计算量的技术手段，更是一种信息抽象的过程。每次下采样，我们都在回答一个问题：如何用更少的数字表示更大的区域？

**1. 最大池化时代（2012-2015）**
最大池化（`nn.MaxPool2d`）曾是卷积神经网络的标配。其背后的假设是：在一个局部区域内，最强的激活值代表了最重要的特征。这种假设在分类任务中很合理——我们关心的是"是否存在某个特征"，而不是"特征在哪里"。

```
输入: [[1, 2],    MaxPool2d    输出: [4]
       [3, 4]]    (2x2)
```

然而，对于生成任务，这种"赢者通吃"的策略是灾难性的：
- **位置信息丢失**：我们不知道最大值来自哪个位置
- **梯度稀疏**：只有最大值位置有梯度，其他位置梯度为零
- **不可逆性**：无法从池化后的结果准确重建原始信息

**2. 步进卷积革命（2015-2018）**
DCGAN论文提出了一个革命性的想法：让网络自己学习如何下采样。步进卷积（`stride=2`的`nn.Conv2d`）将下采样和特征提取合二为一：

```python
# 传统方法：先卷积，后池化
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2)
output = pool(conv(input))  # 两步操作

# 现代方法：步进卷积一步到位
strided_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
output = strided_conv(input)  # 一步操作，可学习
```

这种方法的优势在于：
- **完全可学习**：网络可以学习最适合任务的下采样方式
- **保留更多信息**：不是简单地选择最大值，而是学习加权组合
- **梯度流畅**：所有位置都参与计算，梯度流动更健康

**3. 现代最佳实践：分而治之（2018至今）**
随着模型规模的增长，训练稳定性成为关键考虑。现代架构倾向于将"改变分辨率"和"提取特征"解耦：

```python
# 第一步：在当前分辨率提取特征
conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
# 第二步：专门的下采样层
downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
```

这种设计的智慧在于：
- **功能解耦**：每层专注于一个任务，更容易优化
- **灵活性**：可以在两步之间插入归一化、激活函数等
- **数值稳定**：避免在一个操作中进行过于剧烈的变换

#### 上采样的挑战：从低分辨率重建细节

如果说下采样是"压缩"，那么上采样就是"解压缩"。但与信息压缩不同，神经网络的上采样需要"创造"原本不存在的细节。

**1. 转置卷积的诱惑与陷阱**
转置卷积（`nn.ConvTranspose2d`）在数学上是步进卷积的精确逆操作。它通过在输入之间插入零值，然后进行常规卷积来实现上采样：

```
输入: [a, b]  →  插零: [a, 0, b]  →  卷积: 生成更大的输出
```

然而，这种方法存在一个致命问题：**棋盘效应（Checkerboard Artifacts）**。当`kernel_size`不能被`stride`整除时，输出像素接收到的"贡献"不均匀：

```
kernel_size=3, stride=2 的情况：
某些输出像素被1个输入像素影响
某些输出像素被2个输入像素影响
→ 产生棋盘状的明暗模式
```

这个问题在2016年被Odena等人系统分析后，引发了社区的广泛讨论。

**2. 插值+卷积：简单但有效的解决方案**
为了避免棋盘效应，现代架构采用了一个看似"倒退"但实际上更稳健的方法：

```python
# 方法1：最近邻插值 + 卷积
upsample = nn.Upsample(scale_factor=2, mode='nearest')
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
output = conv(upsample(input))

# 方法2：双线性插值 + 卷积
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
output = conv(upsample(input))
```

这种方法的优势：
- **无棋盘效应**：插值保证了空间均匀性
- **计算效率高**：插值操作很快，卷积是标准操作
- **易于理解和调试**：两步操作各司其职

**3. 亚像素卷积：另一种优雅的方案**
亚像素卷积（Pixel Shuffle）提供了另一种思路：先在低分辨率空间生成多个通道，然后重新排列成高分辨率输出：

```python
# 输入: [B, C, H, W]
# 先扩展通道: [B, C*r², H, W]
conv = nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=3, padding=1)
# 然后重排: [B, C, H*r, W*r]
pixel_shuffle = nn.PixelShuffle(scale_factor)
output = pixel_shuffle(conv(input))
```

这种方法在超分辨率任务中特别流行，因为它允许网络在低分辨率空间进行大部分计算。

#### 采样策略对扩散模型的特殊意义

在扩散模型中，采样方式的选择有着特殊的重要性：

**1. 信息保真度**
扩散模型需要在多个时间步之间传递信息。任何信息损失都会在迭代过程中被放大。因此，可逆或近似可逆的采样方式（如步进卷积配合适当的上采样）特别重要。

**2. 多尺度一致性**
去噪过程需要在不同尺度上保持一致性。粗糙的采样方式可能导致不同分辨率层之间的特征不匹配，影响最终的生成质量。

**3. 计算效率的关键**
U-Net的大部分计算发生在低分辨率层。高效的采样策略可以显著减少计算量，这是Stable Diffusion能够在消费级硬件上运行的关键因素之一。

### 2.2.3 归一化技术：从BatchNorm到AdaGN的演进

归一化技术的演进史，是深度学习社区对"如何让深层网络稳定训练"这一核心问题不断探索的历史。在扩散模型中，归一化不仅影响训练稳定性，更成为了注入条件信息的关键机制。

#### 归一化的本质：对抗内部协变量偏移

2015年，Ioffe和Szegedy提出BatchNorm时，他们的核心观察是：深层网络训练困难的一个重要原因是**内部协变量偏移（Internal Covariate Shift）**——即每层的输入分布在训练过程中不断变化，导致后续层需要不断适应新的输入分布。

归一化的基本思想很简单：
```
归一化输出 = γ × (输入 - 均值) / 标准差 + β
```
其中γ和β是可学习的缩放和偏移参数。关键在于：如何计算均值和标准差？

#### BatchNorm的局限：为什么它不适合扩散模型

BatchNorm在许多任务上取得了巨大成功，但在扩散模型中却遇到了前所未有的挑战：

**1. 批次依赖性带来的不一致**
BatchNorm在训练时使用当前批次的统计量，在推理时使用移动平均。这导致：
```python
# 训练时：使用批次统计
mean = x.mean(dim=[0, 2, 3])  # 跨批次维度计算
var = x.var(dim=[0, 2, 3])
x_norm = (x - mean) / sqrt(var + eps)

# 推理时：使用移动平均
x_norm = (x - running_mean) / sqrt(running_var + eps)
```

对于扩散模型，这种不一致是致命的：
- 生成时通常batch_size=1，统计量毫无意义
- 训练和推理的行为差异会累积放大

**2. 时间步混淆问题**
扩散模型的一个批次中，不同样本可能处于不同的时间步：
```
批次 = [x_t1, x_t2, x_t3, x_t4]  # t1, t2, t3, t4可能完全不同
```

BatchNorm会将这些处于不同噪声水平的样本混合计算统计量，这就像把苹果和橙子混在一起求平均——毫无意义。

**3. 小批量训练的灾难**
高分辨率的扩散模型因为内存限制，批次大小通常很小（如2或4）。在如此小的批次上估计统计量，方差极大，训练极不稳定。

#### GroupNorm：优雅的解决方案

2018年，何恺明等人提出的GroupNorm巧妙地解决了这些问题。其核心思想是：**不跨样本计算统计量，而是在每个样本内部，将通道分组后计算**。

```python
# GroupNorm的计算方式
# 假设输入 x 的形状为 [B, C, H, W]
# 将 C 个通道分成 G 组
x = x.view(B, G, C//G, H, W)
mean = x.mean(dim=[2, 3, 4])  # 在每组内计算
var = x.var(dim=[2, 3, 4])
x = (x - mean) / sqrt(var + eps)
x = x.view(B, C, H, W)
```

GroupNorm的优势：
- **批次无关**：每个样本独立计算，batch_size=1也能正常工作
- **时间步隔离**：不同时间步的样本互不影响
- **稳定性好**：不依赖批次大小，小批量训练也稳定

GroupNorm实际上是一个统一框架：
- 当 G = 1 时，退化为 LayerNorm（跨所有通道归一化）
- 当 G = C 时，退化为 InstanceNorm（每个通道独立归一化）
- 当 G = 32 时（常用设置），在两者之间取得平衡

#### 自适应归一化：从固定到动态的飞跃

传统的归一化使用固定的γ和β参数。但StyleGAN的成功启发了一个革命性的想法：**让这些参数根据外部条件动态变化**。

**AdaGN（Adaptive Group Normalization）的工作原理：**

```python
class AdaGN(nn.Module):
    def __init__(self, num_features, num_groups=32, time_emb_dim=128):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_features)
        # 从时间嵌入预测 scale 和 shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_features * 2)
        )
    
    def forward(self, x, time_emb):
        # 计算动态的 scale 和 shift
        scale_shift = self.time_mlp(time_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        
        # 应用 GroupNorm
        x = self.norm(x)
        
        # 应用动态调制
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x
```

**为什么AdaGN对扩散模型如此有效？**

1. **时间感知的去噪**：不同时间步需要不同的去噪策略。早期（高噪声）可能需要更强的归一化来稳定训练，后期（低噪声）可能需要更弱的归一化来保留细节。

2. **高效的条件注入**：相比于将条件信息拼接到特征图（增加计算量），AdaGN通过调制现有特征实现条件控制，几乎不增加计算成本。

3. **分层的控制粒度**：每一层可以根据时间步独立调整其行为，这种细粒度的控制对于处理不同尺度的噪声至关重要。

#### 归一化位置的艺术：前置还是后置？

在Transformer的发展过程中，Layer Normalization的位置引发了激烈讨论。这个讨论同样适用于U-Net：

**Post-Norm（传统方式）：**
```
x → Conv → ReLU → Norm → + → 输出
                         ↑
                         x (残差连接)
```

**Pre-Norm（现代方式）：**
```
x → Norm → Conv → ReLU → + → 输出
                         ↑
                         x (残差连接)
```

Pre-Norm的优势：
- **梯度流更稳定**：残差连接直接连接输入输出，梯度可以无障碍地流过
- **训练更容易**：特别是对于非常深的网络
- **与AdaGN配合更好**：在块的开始就进行条件调制，影响整个块的计算

这就是为什么现代扩散模型普遍采用Pre-Norm设计。

#### RMSNorm：更简单的未来？

最近，RMSNorm作为LayerNorm的简化版本引起了关注：

```python
# LayerNorm: 减均值，除标准差
x_norm = (x - mean) / std

# RMSNorm: 只除以均方根，不减均值
x_norm = x / sqrt(mean(x²))
```

RMSNorm的优势：
- **计算更简单**：少了减均值的操作
- **某些情况下效果相当**：特别是当激活函数本身有中心化效果时

虽然RMSNorm在扩散模型中的应用还不广泛，但它代表了一个重要趋势：**不断简化和优化基础组件**。

<details>
<summary><strong>练习 2.1：U-Net架构的权衡分析</strong></summary>

1.  **深度 vs. 宽度**：分析U-Net的深度（下采样次数）和宽度（基础通道数）对模型性能和计算成本的影响。对于一个固定计算预算的模型，是更深好还是更宽好？
2.  **注意力位置**：讨论在U-Net的不同层级（高、中、低分辨率）插入自注意力模块的利弊。为什么大多数模型选择在中低分辨率层插入？
3.  **跳跃连接**：标准的跳跃连接使用拼接（concatenation）。分析如果改为逐元素相加（addition）会对信息流产生什么影响。在什么情况下相加可能是更好的选择？
4.  **开放探索**：设计一种“动态U-Net”，其深度或宽度可以根据输入的时间步`t`自适应调整。例如，在噪声水平高时使用更深的网络来捕捉全局结构，在噪声水平低时使用更浅的网络来关注细节。
5.  **研究思路**：
    *   查阅有关神经架构搜索（NAS）在生成模型中应用的研究。
    *   从信息论角度分析跳跃连接，将其视为信息瓶颈的旁路。
    *   研究不同归一化层（如`RMSNorm`）与自适应调制结合的可能性。

</details>

## 2.3 从NLP到CV：Vision Transformer的跨界之旅

### 2.3.1 Transformer的视觉革命

Transformer架构由Vaswani等人在2017年的论文《Attention Is All You Need》中为自然语言处理提出。2020年，Dosovitskiy等人的ViT论文证明了纯Transformer架构在图像分类上可以达到甚至超越顶尖的CNN，开启了Transformer在计算机视觉领域的革命。

ViT的核心思想极其简洁：
1.  将输入图像分割成固定大小的patches（例如16×16像素）。
2.  将每个patch线性投影（embedding）为一个向量（token）。
3.  将这些tokens序列以及一个可学习的`[CLS]` token输入到标准的Transformer编码器中。
4.  使用Transformer输出的`[CLS]` token进行分类。

这种设计的优雅之处在于它为视觉问题引入了新的归纳偏置：**世界是由可组合的“部件”构成的**。

### 2.3.2 扩散Transformer (DiT)

2022年，Peebles和Xie在论文《Scalable Diffusion Models with Transformers》中提出了DiT，成功将ViT架构应用于扩散模型。DiT对ViT进行了关键改造以适应去噪任务：

1.  **输入处理**：输入不再是清晰图像，而是带噪声的图像patches。
2.  **无`[CLS]` Token**：生成任务需要对每个patch进行预测，因此去除了分类任务专用的`[CLS]` token。
3.  **条件注入**：时间步`t`和类别标签`c`的嵌入向量被视为额外的条件tokens，通过自适应LayerNorm（AdaLN）或交叉注意力（cross-attention）注入到模型中。
4.  **输出处理**：Transformer的输出tokens被重新排列，并通过一个线性解码器预测每个patch对应的噪声。

DiT的成功，特别是其卓越的可扩展性（scaling law），使其迅速成为SOTA文生图模型（如Sora, Stable Diffusion 3）的首选架构。

<details>
<summary><strong>练习 2.2：比较U-Net和DiT的归纳偏置与复杂度</strong></summary>

1.  **归纳偏置**：对比CNN（U-Net的基础）和Transformer（DiT的基础）的核心归纳偏置。CNN的“局部性”和“平移等变性”与Transformer的“全局关系”和“排列不变性”分别如何影响它们作为去噪网络的性能？
2.  **计算复杂度**：对于一个分辨率为`H x W`的输入，推导U-Net和DiT的主要计算瓶颈。U-Net的复杂度与什么成正比？DiT的复杂度与什么成正比？（提示：考虑卷积操作和自注意力操作的复杂度）
3.  **开放探索**：U-Net和DiT代表了两种不同的架构范式。近年来，出现了许多试图结合两者优点的混合架构（如U-ViT）。分析这种混合设计的动机，并提出一种你自己的混合块（hybrid block）设计。
4.  **研究思路**：
    *   阅读ViT和DiT的原文，关注作者关于模型扩展性（scaling）的实验部分。
    *   探索卷积操作和自注意力在数学上的联系（例如，卷积可以被看作是一种特殊的、带强位置偏置的局部注意力）。
    *   研究最新的SOTA生成模型（如Sora的技术报告），分析其架构选择。

</details>

## 2.4 性能优化与实用技巧

理论架构和实际部署之间往往存在巨大鸿沟。本节分享一些在实践中积累的优化技巧。

### 2.4.1 内存优化：在GPU上塞下更大的模型

训练扩散模型时，内存的最大消耗通常来自**激活值**，特别是U-Net中为跳跃连接而保存的各层特征图。

- **梯度检查点 (Gradient Checkpointing)**：核心思想是“用计算换内存”。通过`torch.utils.checkpoint.checkpoint`包裹模型的一部分（如一个ResBlock），在前向传播时不保存其内部的激活值，而在反向传播时重新计算它们。这可以显著降低内存占用（约30-50%），但会增加训练时间（约20-30%）。

- **混合精度训练 (Mixed Precision)**：使用`torch.cuda.amp`（自动混合精度）可以利用现代GPU的Tensor Cores，将大部分计算从FP32转为FP16或BF16，内存减半，速度翻倍。关键是使用`GradScaler`来防止FP16梯度下溢。

- **注意力优化**：标准自注意力的内存和计算复杂度与序列长度的平方成正比。对于高分辨率图像，这很快会成为瓶颈。FlashAttention等库通过融合内核操作，避免将巨大的注意力矩阵写入和读出GPU内存，从而实现显著的加速和内存节省。

### 2.4.2 训练稳定性：让大模型稳定收敛

- **初始化策略**：一个关键技巧是**将输出层的权重和偏置初始化为零**。这确保模型在训练开始时输出为零，即预测的噪声为零。这是一种“无为而治”的初始化，使得模型在学习初期不会对输入造成巨大扰动，有助于稳定训练。

- **数值稳定性**：
    - **梯度裁剪**：通过`torch.nn.utils.clip_grad_norm_`来防止梯度爆炸，是训练大模型的标配。
    - **学习率调度**：使用预热（warmup）和余弦退火（cosine decay）的学习率调度器通常比固定学习率效果更好。
    - **AdamW优化器**：AdamW通过解耦权重衰减和梯度更新，通常比标准Adam更稳定。

<details>
<summary><strong>综合练习：为特定任务设计去噪网络</strong></summary>

假设你要为以下两种不同的任务设计去噪网络架构，你会如何选择和修改U-Net或DiT？请详细说明理由。

**任务A：移动端实时人像风格化**
- **约束**：模型大小 < 50MB，在手机GPU上推理延迟 < 100ms。
- **数据**：512x512的人像图片。

**任务B：生成具有复杂物理规律的科学模拟数据（如流体动力学）**
- **约束**：追求最高的物理保真度，计算资源几乎无限。
- **数据**：256x256x256的3D体数据，需要尊重物理守恒定律。

**设计分析与研究方向：**
1.  **架构选择**：为每个任务选择基础架构（U-Net, DiT, 或混合架构），并论证你的选择。
2.  **关键修改**：针对每个任务的约束和数据特性，你会对所选架构进行哪些关键修改？（例如，对于任务A，如何修改通道数、深度、注意力机制？对于任务B，如何处理3D数据、如何引入物理约束？）
3.  **理论空白**：在任务B中，如何设计一个能内建物理不变量（如散度为零）的神经网络架构？这被称为物理信息神经网络（PINN）与生成模型的交叉领域，是一个活跃的研究方向。
4.  **研究思路**：
    *   查阅有关模型量化、剪枝和知识蒸馏的文献，以满足任务A的部署要求。
    *   研究傅里叶神经算子（Fourier Neural Operator）等将物理方程求解器与神经网络结合的工作，以获取任务B的灵感。
    *   探索等变神经网络（Equivariant Neural Networks），它们被设计用来尊重数据的内在对称性（如旋转不变性）。

</details>

## 本章小结

本章我们追溯了扩散模型中两种主流架构的历史渊源，并深入分析了它们的设计细节和演进过程。

- **U-Net**：从2015年的医学图像分割任务，到2020年成为DDPM的核心架构，其多尺度特征融合能力是成功的关键。
- **Vision Transformer (DiT)**：从2017年的NLP革命，经2020年的CV突破，到2022年成为可扩展扩散模型的主流选择，其全局关系建模能力和卓越的扩展性是其优势所在。

这两种架构能够成功应用于扩散模型并非偶然，而是经过了精心的改造和适配：
- **共同的改造**：都引入了时间嵌入作为关键的条件信息，并发展出如AdaGN/AdaLN等高效的注入机制。
- **不同的演进**：U-Net在卷积、采样和归一化等模块上不断优化；DiT则专注于如何将Transformer范式更好地应用于像素级的生成任务。

扩散模型的架构演进史启示我们：创新并不总是需要“从零开始”。善于发现和利用已有技术的潜力，通过巧妙的改造和组合，往往能产生意想不到的突破。

下一章，我们将深入DDPM的数学原理，看看这些强大的架构是如何在一个清晰的概率框架下进行训练和优化的。
