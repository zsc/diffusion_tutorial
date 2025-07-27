[← 上一章](chapter1.md)
 第2章 / 共14章
 [下一章 →](chapter3.md)



# 第2章：神经网络架构：U-Net与ViT



 扩散模型的成功离不开强大的神经网络架构。有趣的是，扩散模型并没有发明全新的网络结构，而是巧妙地借用了计算机视觉领域的两个里程碑式架构：U-Net和Vision Transformer。本章将追溯这两种架构的历史发展，理解它们的设计初衷，以及为什么它们恰好适合扩散模型的去噪任务。这种"历史的巧合"展示了深度学习领域知识迁移的魅力。



## 2.1 从图像分割到去噪：U-Net的历史演变



U-Net最初并非为扩散模型设计。2015年，Ronneberger等人提出U-Net用于生物医学图像分割，其独特的编码器-解码器结构配合跳跃连接，能够在保留细节信息的同时进行语义理解。这种架构特性恰好契合了扩散模型的需求。



> **定义**
> 历史脉络



 - **2015年**：U-Net诞生，用于医学图像分割，在ISBI细胞追踪挑战赛中获胜
 - **2017-2019年**：U-Net被广泛应用于各种密集预测任务（语义分割、深度估计等）
 - **2020年**：DDPM采用U-Net作为去噪网络，开启了U-Net在生成模型中的新篇章
 - **2021-2022年**：各种改进版U-Net成为扩散模型的标配（加入注意力机制、自适应归一化等）





为什么U-Net特别适合扩散模型？关键在于扩散模型的去噪任务本质上是一个"图像到图像"的转换问题：




 输入：噪声图像 $\mathbf{x}_t$ → 输出：预测的噪声 $\boldsymbol{\epsilon}$ 或 清晰图像 $\mathbf{x}_0$




这与U-Net最初设计的分割任务（输入：原始图像 → 输出：分割掩码）在结构上高度相似。U-Net的几个关键特性使其成为理想选择：




 - **多尺度特征提取**：编码器逐步降低分辨率，捕获从局部纹理到全局结构的各层次特征
 - **跳跃连接保留细节**：直接将编码器的特征传递给解码器，避免细节信息在下采样过程中丢失
 - **对称结构**：编码器和解码器的对称设计，天然适合"加噪"和"去噪"的可逆过程
 - **参数效率**：相比全连接网络，卷积结构的参数共享大大减少了模型参数量




## 2.2 U-Net架构详解



### 2.2.1 原始U-Net：优雅的对称设计



2015年的原始U-Net论文标题很有意思："U-Net: Convolutional Networks for Biomedical Image Segmentation"。作者们面临的挑战是：如何在训练数据极少的情况下（ISBI挑战赛只有30张训练图像！）实现精确的细胞分割？他们的解决方案展现了深度学习的一个重要原则：**好的架构设计可以弥补数据的不足**。





 原始 U-Net 架构 (2015)

 输入图像 输出分割
 572×572 388×388
 | ↑
 ↓ |
 [Conv3×3, ReLU] × 2 ←―――――――――――――――――→ [Conv3×3, ReLU] × 2
 64 channels 跳跃连接 (crop) 64 channels
 | ↑
 ↓ MaxPool 2×2 ↑ Conv2×2 (上采样)
 | |
 [Conv3×3, ReLU] × 2 ←―――――――――――――――――→ [Conv3×3, ReLU] × 2
 128 channels 128 channels
 | ↑
 ↓ MaxPool 2×2 ↑ Conv2×2 (上采样)
 | |
 ... 继续下采样 ... ... 继续上采样 ...
 | |
 ↓ ↑
 底部：1024 channels
 


 U-Net的几个关键设计决策至今仍然影响着深度学习架构设计：




 - **非对称的输入输出尺寸**：原始U-Net使用valid卷积（无padding），导致每次卷积都会缩小特征图。这是为了避免边界伪影，确保输出的每个像素都有完整的感受野。

 - **跳跃连接的crop操作**：由于尺寸不匹配，需要裁剪（crop）编码器特征再与解码器特征拼接。这个看似笨拙的设计其实确保了特征对齐的精确性。

 - **数据增强的重要性**：原文特别强调了弹性形变（elastic deformation）对小数据集的重要性——这启发了后续研究中各种数据增强技术。




### 2.2.2 现代U-Net：为扩散模型重新设计



当DDPM的作者们在2020年选择U-Net作为去噪网络时，他们面临着完全不同的需求。原始U-Net的一些设计（如valid卷积、非对称尺寸）对扩散模型来说是不必要的复杂。于是，一个"现代化"的U-Net诞生了：



> **定义**
> 扩散模型U-Net的关键改进

 
 
 组件
 原始U-Net (2015)
 扩散U-Net (2020+)
 
 
 卷积类型
 Valid卷积（无padding）
 Same卷积（保持尺寸）
 
 
 归一化
 无（后来加入BatchNorm）
 GroupNorm（稳定训练）
 
 
 激活函数
 ReLU
 SiLU/Swish（更平滑）
 
 
 残差连接
 无
 每个块都有（类ResNet）
 
 
 注意力机制
 无
 多分辨率自注意力
 
 
 条件机制
 无需条件
 时间嵌入（必需）
 
 



让我们深入理解几个关键改进：



#### 1. 残差块设计





 输入 x
 ↓
 GroupNorm → SiLU → Conv3×3
 ↓
 GroupNorm → SiLU → Conv3×3
 ↓
 + ← x （残差连接）
 ↓
 输出
 




#### 2. 时间嵌入注入

 时间信息通过自适应归一化层注入到每个残差块中：



 $h = \text{GroupNorm}(h)$

 $h = h \cdot (1 + \text{scale}(t)) + \text{shift}(t)$

 其中 $\text{scale}(t)$ 和 $\text{shift}(t)$ 是时间嵌入经过线性变换得到的




#### 3. 自注意力的引入位置


自注意力通常只在中等分辨率（如16×16、32×32）引入，原因是：



 - 高分辨率（64×64以上）：计算成本过高，且局部特征更重要
 - 低分辨率（8×8以下）：特征图太小，全局信息已经被压缩
 - 中等分辨率：平衡了计算效率和全局建模能力




### 2.2.3 下采样的演进：从池化到步进卷积



下采样（downsampling）是U-Net的核心操作之一，但"如何正确地降低分辨率"这个看似简单的问题，却经历了深度学习历史上的多次范式转变。



#### 第一代：最大池化的统治时期（2012-2015）



早期CNN几乎都使用最大池化（MaxPooling）进行下采样，包括原始U-Net：




# 经典的MaxPool下采样
self.down = nn.MaxPool2d(kernel_size=2, stride=2)

# 优点：保留最强激活，计算简单
# 缺点：丢失位置信息，不可学习


 最大池化的问题在于它是一个固定的、不可学习的操作。在每个2×2窗口中，75%的信息被直接丢弃，这对于需要精确重建的生成任务来说是灾难性的。



#### DCGAN的革命性发现（2015）



> **定义**
> DCGAN论文的关键贡献


Radford等人在"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"中提出了几条影响深远的架构指南：



 - **用步进卷积（strided convolution）替代池化**：让网络自己学习如何下采样
 - **去除全连接层**：保持全卷积架构
 - **批归一化的系统使用**：除了生成器输出层和判别器输入层
 - **ReLU vs LeakyReLU**：生成器用ReLU，判别器用LeakyReLU





DCGAN的这些发现不仅影响了GAN，也深刻改变了后续所有生成模型的设计，包括扩散模型。特别是"可学习的下采样"这一理念，成为了现代架构的标准。



#### 步进卷积：让网络决定如何采样




# DCGAN风格的可学习下采样
self.down = nn.Conv2d(in_channels, out_channels,
 kernel_size=3, stride=2, padding=1)

# 为什么是3×3而不是2×2？
# - 3×3提供更大的感受野
# - 避免棋盘效应（稍后详述）
# - 与padding=1配合，输出正好是输入的一半



#### 上采样的对称问题：棋盘效应


 既然下采样可以用步进卷积，那么上采样自然想到用转置卷积（transposed convolution）。但这带来了一个意外的问题：




##### ⚠️ 棋盘效应（Checkerboard Artifacts）


当转置卷积的kernel_size不能被stride整除时，会产生棋盘状的伪影：



 kernel_size=3, stride=2 的重叠模式：

 1 2 1 2 1
 2 4 2 4 2
 1 2 1 2 1
 2 4 2 4 2
 1 2 1 2 1

 某些像素被覆盖4次，某些只有1次！
 



#### 现代解决方案：解耦采样和特征变换


 为了避免棋盘效应，现代架构倾向于将采样和卷积分离：




# 下采样：先卷积，再降采样
class DownBlock(nn.Module):
 def __init__(self, in_channels, out_channels):
 super().__init__()
 self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
 self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

# 上采样：先插值，再卷积（推荐）
class UpBlock(nn.Module):
 def __init__(self, in_channels, out_channels):
 super().__init__()
 self.up = nn.Upsample(scale_factor=2, mode='nearest')
 self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

# 或者使用PixelShuffle（对于学习型上采样）
class PixelShuffleUp(nn.Module):
 def __init__(self, in_channels, out_channels):
 super().__init__()
 # 先增加通道数到4倍
 self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
 self.shuffle = nn.PixelShuffle(2) # 将通道重排为空间维度



#### 扩散模型中的最佳实践



> **定义**
> 下采样方式对比（扩散模型视角）

 
 
 方法
 优点
 缺点
 推荐度
 
 
 MaxPool
 简单、无参数
 信息丢失严重
 ❌ 不推荐
 
 
 步进卷积
 可学习、灵活
 可能有伪影
 ✅ 推荐
 
 
 Conv + 步进Conv
 更稳定
 计算量略大
 ✅✅ 强烈推荐
 
 
 平均池化
 平滑、抗锯齿
 模糊细节
 ⚡ 特定场景
 
 


 **实践建议**：



 - 对于扩散模型，推荐使用"Conv + 步进Conv"的组合，这在Stable Diffusion等SOTA模型中被广泛采用
 - 上采样优先使用"最近邻插值 + Conv"，避免棋盘效应
 - 如果需要学习型上采样，PixelShuffle是个不错的选择
 - 在高分辨率（>256）时，考虑使用抗锯齿下采样（blur pool）




### 2.2.4 权重共享之争：编码器与解码器是否应该绑定？



一个经常被忽视但值得深入讨论的设计决策是：U-Net的编码器和解码器是否应该共享权重（weight tying）？这个问题在自编码器领域有悠久的历史，但在扩散模型中有其独特的考量。



#### 权重共享的理论动机



权重共享的想法来源于一个优雅的对称性假设：




 如果编码器学习映射 $f: \mathcal{X} \rightarrow \mathcal{Z}$

 那么解码器应该学习逆映射 $f^{-1}: \mathcal{Z} \rightarrow \mathcal{X}$

 理想情况下，$f^{-1} = f^T$（转置关系）




在传统的去噪自编码器（Denoising Autoencoders）中，这种设计有几个吸引人的特性：



 - **参数效率**：模型参数量减少一半
 - **正则化效果**：强制编码器和解码器学习互逆的变换
 - **理论优雅**：符合某些流形学习的理论框架




#### 扩散模型的特殊性：为什么不共享？



> **定义**
> 扩散模型中编码器和解码器的不对称性

 
 
 方面
 编码器任务
 解码器任务
 
 
 主要目标
 提取多尺度特征表示
 基于特征重建细节
 
 
 信息流向
 从细节到语义（压缩）
 从语义到细节（生成）
 
 
 噪声敏感度
 需要对噪声鲁棒
 需要精确预测噪声
 
 
 时间依赖性
 提取与t相关的特征
 根据t调整去噪策略
 
 



扩散模型的去噪任务本质上是**非对称的**：



 - **不同时间步需要不同策略**：早期（高噪声）需要全局结构恢复，后期（低噪声）需要细节润色
 - **跳跃连接的作用不同**：编码器的跳跃连接保留原始信息，解码器的跳跃连接用于精确重建
 - **注意力机制的需求不同**：编码器可能需要局部注意力，解码器可能需要全局注意力




#### 实验证据：权重共享的实际效果

权重共享是一个有趣的理论问题。实验设计要点：

**权重共享U-Net：**
- 解码器使用编码器权重的转置
- 使用`F.conv_transpose2d`实现转置卷积
- 需要仔细处理维度匹配问题

**独立权重U-Net（标准）：**
- 编码器和解码器完全独立
- 参数量翻倍但表达能力更强
- 当前所有SOTA模型的选择

💡 **研究方向：自适应权重共享**  
能否设计一种"软"权重共享机制，在训练过程中自适应地调整共享程度？这可能结合`nn.Parameter`的灵活使用和正则化技术。


 根据社区的实验和论文报告，权重共享在扩散模型中的表现：




##### 实验结果汇总



 - **FID分数**：独立权重通常好2-5个点
 - **训练稳定性**：权重共享在某些配置下会导致训练不稳定
 - **参数效率**：权重共享确实减少50%参数，但性能下降不成比例
 - **泛化能力**：独立权重在out-of-distribution数据上表现更好





#### 特殊情况：部分权重共享



一些研究探索了折中方案：




部分权重共享是一种折中方案：

**设计原则：**
- 高分辨率层（细节）：独立权重
- 中间层（通用特征）：共享权重  
- 低分辨率层（语义）：独立权重

这种设计基于不同层次特征的语义差异。中间层学习的特征较为通用，适合共享；而浅层和深层的任务差异较大。

🔬 **研究线索：层级特定的归纳偏置**  
不同深度的层是否需要不同的架构设计？如何根据层的语义级别自适应调整架构？这涉及到神经架构搜索（NAS）在扩散模型中的应用。


 这种设计的直觉是：中间层学习的是较为通用的特征变换，可以共享；而浅层和深层由于任务差异较大，应该独立。



#### 现代实践：为什么主流模型都不用权重共享？



> **定义**
> 主流扩散模型的选择



 - **DDPM (2020)**：独立权重
 - **IDDPM (2021)**：独立权重 + 更深的架构
 - **ADM (2021)**：独立权重 + 自适应归一化
 - **Stable Diffusion (2022)**：独立权重 + 交叉注意力
 - **DiT (2022)**：不适用（Transformer架构）





**结论**：尽管权重共享在理论上很优雅，在扩散模型的实践中，独立权重已经成为事实标准。主要原因是：



 - 扩散模型的计算瓶颈不在参数量，而在推理步数
 - 编码器和解码器的任务确实存在本质差异
 - 现代GPU的内存已经足够大，参数效率不再是首要考虑
 - 性能提升（2-5个FID点）对于竞争激烈的生成模型领域很重要





思考练习：设计实验验证权重共享的效果

设计一个实验来验证权重共享对扩散模型性能的影响。考虑以下因素：



 - 如何确保公平比较（相同的总参数量 vs 相同的架构）？
 - 应该在哪些指标上评估（FID、IS、LPIPS、推理时间）？
 - 如何测试不同噪声水平下的表现差异？

**实验设计建议：**



 - **公平比较**：


 方案A：固定总参数量，权重共享版本可以更深
 - 方案B：固定架构深度，接受参数量差异
 - 推荐方案A，因为实际应用中参数效率很重要



 - **评估指标**：


 生成质量：FID、IS、Precision/Recall
 - 去噪能力：不同t下的MSE
 - 训练动态：损失曲线、梯度范数
 - 推理效率：内存使用、推理时间



 - **噪声水平测试**：


 将t分为三段：[0,300]、[300,700]、[700,1000]
 - 分别评估每段的去噪表现
 - 预期：高噪声段差异更大（解码器任务更难）





### 2.2.5 归一化技术FAQ：从BatchNorm到AdaGN的演进史



归一化（Normalization）可能是深度学习中最令人困惑的组件之一。让我们通过FAQ的形式，追溯归一化技术的演进历程，特别是它如何影响了生成模型的发展。



> **定义**
> Q1: 为什么原始U-Net没有使用归一化？


**A:** 2015年的U-Net发表时，BatchNorm刚刚被提出几个月（Ioffe & Szegedy, 2015）。原始U-Net依赖于良好的初始化和相对较浅的架构（只有4次下采样）来保持训练稳定。这在当时是常见做法，但限制了模型的深度。




#### 归一化技术时间线



 
 
 年份
 技术
 关键创新
 生成模型影响
 
 
 2015
 BatchNorm
 标准化批次统计
 DCGAN采用，但需要大批次
 
 
 2016
 LayerNorm
 标准化特征维度
 主要用于RNN/Transformer
 
 
 2016
 InstanceNorm
 每个样本独立标准化
 风格迁移的关键
 
 
 2018
 GroupNorm
 组内标准化
 成为扩散模型标准
 
 
 2019
 AdaIN (StyleGAN)
 自适应实例标准化
 革命性的风格控制
 
 
 2020+
 AdaGN
 自适应组标准化
 扩散模型时间调制
 
 



> **定义**
> Q2: 为什么扩散模型不用BatchNorm？


**A:** BatchNorm有几个对扩散模型致命的问题：



 - **批次依赖**：生成质量受同批次其他样本影响，导致推理时行为不一致
 - **小批次退化**：扩散模型由于内存限制常用小批次（如4或8），BatchNorm统计不稳定
 - **时间步混淆**：不同时间步的样本混在一起计算统计量，违背了扩散过程的假设





#### StyleGAN的革命性发现：归一化即风格



StyleGAN (Karras et al., 2019) 带来了一个深刻洞察：**归一化操作本质上是在操纵特征的统计量，而这些统计量恰好编码了"风格"信息**。




# StyleGAN的AdaIN实现
class AdaIN(nn.Module):
 def __init__(self, num_features):
 super().__init__()
 self.norm = nn.InstanceNorm2d(num_features, affine=False)

 def forward(self, x, style):
 # style是从潜在码w映射而来的向量
 # 分别预测每个通道的均值和标准差调制
 style = style.view(style.size(0), 2, x.size(1), 1, 1)
 gamma = style[:, 0] # 标准差调制
 beta = style[:, 1] # 均值调制

 # 先标准化，再用style调制
 normalized = self.norm(x)
 return gamma * normalized + beta

# 扩散模型的AdaGN实现
class AdaGN(nn.Module):
 def __init__(self, num_features, num_groups=32):
 super().__init__()
 self.norm = nn.GroupNorm(num_groups, num_features, affine=False)

 def forward(self, x, time_emb):
 # time_emb是时间步的嵌入
 # 通过MLP映射到scale和shift
 scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
 scale = scale.view(-1, x.size(1), 1, 1)
 shift = shift.view(-1, x.size(1), 1, 1)

 normalized = self.norm(x)
 return normalized * (1 + scale) + shift # 注意这里的(1 + scale)



> **定义**
> Q3: 各种归一化方法的直观理解是什么？




```python

BatchNorm: 在批次维度求统计 → "这批图片的平均亮度是多少？"
LayerNorm: 在特征维度求统计 → "这个位置所有通道的平均激活是多少？"
InstanceNorm: 在空间维度求统计 → "这张图片的平均纹理强度是多少？"
GroupNorm: 在通道组内求统计 → "这组相关特征的平均响应是多少？"

```






#### 扩散模型中的GroupNorm：为什么是最佳选择？




##### GroupNorm的独特优势



 - **批次无关**：每个样本独立计算，推理一致
 - **参数可调**：通过调整组数在LN和IN之间平衡


 G=1：退化为LayerNorm（全局归一化）
 - G=C：退化为InstanceNorm（逐通道归一化）
 - G=8/16/32：实践中的甜点


 
 - **语义分组**：相近的通道往往编码相似的特征，组归一化保持了这种局部性





> **定义**
> Q4: 为什么要用自适应归一化（AdaGN）而不是简单的条件拼接？

 **A:** 这涉及到归一化的本质作用：



 - **统计调制 vs 特征叠加**：


 拼接：$h' = \text{Conv}([h, \text{cond}])$ - 加法式的特征组合
 - AdaGN：$h' = \gamma(t) \cdot \text{Norm}(h) + \beta(t)$ - 乘法式的统计调制


 
 - **不同时间步需要不同的"去噪策略"**：


 早期（高噪声）：需要强归一化，关注全局结构
 - 后期（低噪声）：需要弱归一化，保留细节
 - AdaGN可以通过调整γ和β灵活控制归一化强度


 





#### 实践建议：如何选择归一化？




# 扩散模型的标准归一化配置
def get_norm_layer(norm_type, num_features, num_groups=32):
 if norm_type == 'batch':
 # ❌ 不推荐：批次依赖
 return nn.BatchNorm2d(num_features)
 elif norm_type == 'instance':
 # ⚠️ 谨慎使用：可能丢失全局信息
 return nn.InstanceNorm2d(num_features)
 elif norm_type == 'layer':
 # ⚠️ 仅用于Transformer架构
 return nn.LayerNorm(num_features)
 elif norm_type == 'group':
 # ✅ 推荐：扩散模型标准选择
 return nn.GroupNorm(num_groups, num_features)
 elif norm_type == 'ada_group':
 # ✅✅ 强烈推荐：最灵活的选择
 return AdaGroupNorm(num_groups, num_features)

# 组数选择指南
def get_optimal_groups(num_channels):
 """基于通道数选择最优组数"""
 if num_channels 



> **定义**
> Q5: LayerNorm在扩散模型中完全没用吗？

 **A:** 不是的！LayerNorm在Transformer架构（如DiT）中是标准配置。关键区别在于：



 - **CNN架构**：空间维度重要 → GroupNorm
 - **Transformer架构**：token维度重要 → LayerNorm



DiT使用的是自适应LayerNorm（AdaLN），原理与AdaGN相似，只是作用维度不同。




#### 未来趋势：无归一化网络？



最新的研究（如NFNet）展示了通过精心的初始化和激活函数设计，可以训练无归一化的深度网络。但在扩散模型中，归一化不仅是为了训练稳定性，更是条件信息注入的重要机制，因此短期内仍将是必需组件。



### 2.2.6 实现细节：魔鬼在细节中



许多看似微小的实现细节对模型性能有巨大影响。以下是一些容易被忽视但很重要的点：



> **定义**
> 实践经验：U-Net实现的"坑"



 - **上采样方式**：最近邻插值 + 卷积 比 转置卷积 更稳定，避免棋盘效应
 - **初始化策略**：零初始化最后一层卷积，使网络初始输出接近零（恒等映射）
 - **通道数设计**：通常遵循 [C, 2C, 4C, 8C] 的倍增规律，但不要超过512-1024
 - **注意力头数**：通道数除以64作为头数是个不错的经验值
 - **跳跃连接处理**：拼接（concat）比相加（add）更常用，保留更多信息






练习 2.1：构建简化的U-Net

[待完成：实现一个最小化的U-Net结构]

[待完成：答案代码和解释]





## 2.3 从NLP到CV：Vision Transformer的跨界之旅



### 2.3.1 Transformer的计算机视觉革命



Transformer架构原本是2017年为自然语言处理设计的（"Attention is All You Need"），但2020年Google的ViT论文证明了纯Transformer架构在图像分类上可以达到甚至超越CNN的性能。这个突破性发现开启了Transformer在计算机视觉领域的广泛应用。



> **定义**
> ViT发展时间线



 - **2017年**：Transformer提出，革新NLP领域
 - **2020年10月**：ViT论文发表，首次将纯Transformer应用于图像分类
 - **2021年**：Swin Transformer、DeiT等变体涌现，Transformer开始统治CV任务
 - **2022年**：Peebles和Xie提出DiT，将Transformer引入扩散模型
 - **2023年**：DiT成为大规模文生图模型的主流选择（如Stable Diffusion 3）





ViT的核心思想极其简洁：将图像分割成固定大小的patches（例如16×16），将每个patch线性投影为向量，然后像处理NLP中的词序列一样处理这些patch序列。这种设计的优雅之处在于：




 - **统一的架构**：图像和文本可以用相同的Transformer处理，促进多模态理解
 - **全局感受野**：自注意力机制让每个patch都能"看到"整张图像，不像CNN需要堆叠才能扩大感受野
 - **可扩展性**：Transformer的性能随模型规模增长呈现出色的scaling law
 - **灵活的序列建模**：容易处理不同分辨率的图像，只需调整patch数量




### 2.3.2 为什么Transformer适合扩散模型？



扩散模型采用Transformer并非偶然。DiT的作者发现，当扩散模型需要处理复杂的全局依赖关系时，Transformer的优势尤为明显：




 去噪任务的本质：理解图像的全局结构 + 恢复局部细节




这恰好是Transformer的强项：



 - **长程依赖建模**：自注意力机制天然擅长捕获远距离像素间的关系
 - **并行计算**：相比U-Net的顺序计算，Transformer可以并行处理所有patches
 - **条件信息融合**：通过cross-attention或AdaLN，轻松注入时间步、类别等条件信息
 - **训练稳定性**：LayerNorm和残差连接使深层Transformer训练更稳定




### 2.3.3 实现用于扩散的ViT


[待完成：PyTorch实现，包括patch embedding、transformer blocks]




练习 2.2：比较U-Net和ViT的计算复杂度

[待完成：分析两种架构的参数量和计算量]

[待完成：详细的复杂度分析]





## 2.4 架构改造：从原始设计到扩散模型适配



将U-Net和ViT应用于扩散模型并非简单的"拿来主义"。研究者们对这些架构进行了巧妙的改造，使其能够处理扩散模型特有的需求。



### 2.4.1 关键改造：时间信息的注入



扩散模型最独特的需求是：网络必须知道当前处于哪个时间步 $t$，因为不同时间步的去噪策略完全不同。原始的U-Net和ViT都没有考虑这一点。主要的改造方案包括：



> **定义**
> 时间嵌入技术



 - **正弦位置编码**：借鉴Transformer的位置编码，将时间步 $t$ 编码为高维向量


 $\text{PE}(t, 2i) = \sin(t/10000^{2i/d})$, $\text{PE}(t, 2i+1) = \cos(t/10000^{2i/d})$



 - **可学习嵌入**：通过MLP将时间步映射到高维空间，更加灵活
 - **自适应归一化（AdaGN）**：用时间嵌入调制归一化层的scale和shift参数
 - **注意力机制**：将时间嵌入作为额外的token加入序列





### 2.4.2 从分类到生成：架构哲学的转变



原始U-Net和ViT都是为判别任务（分割、分类）设计的，而扩散模型是生成任务。这带来了设计哲学的根本转变：


 
 
 方面
 判别任务（原始设计）
 生成任务（扩散模型）
 
 
 输出要求
 语义准确性
 像素级精确度
 
 
 特征重要性
 高层语义特征
 所有层次特征同等重要
 
 
 归一化策略
 BatchNorm（依赖批统计）
 GroupNorm（独立于批大小）
 
 
 注意力使用
 主要在高层
 多个分辨率都需要
 
 


## 2.5 架构演进对比：一图看懂发展脉络



 
 
 U-Net 演进路线
 
 
 
 **原始U-Net (2015)**

 • 用途：医学图像分割

 • 特点：编码器-解码器 + 跳跃连接

 • 归一化：BatchNorm

 • 注意力：无
 
 
 **Diffusion U-Net (2020+)**

 • 用途：扩散模型去噪

 • 新增：时间嵌入 (AdaGN)

 • 归一化：GroupNorm

 • 注意力：多尺度自注意力
 
 
 
 Transformer 演进路线
 
 
 
 **ViT (2020)**

 • 用途：图像分类

 • 特点：Patch嵌入 + 位置编码

 • 输出：分类logits

 • 条件：无
 
 
 **DiT (2022)**

 • 用途：扩散模型去噪

 • 新增：时间/类别条件 (AdaLN)

 • 输出：噪声预测/速度预测

 • 条件：灵活的条件机制
 
 
 




思考题：架构选择

假设你要为以下任务选择去噪网络架构，你会选择U-Net还是Transformer？说明理由：



 - 生成32×32的低分辨率图标
 - 生成1024×1024的高分辨率人脸
 - 生成需要强全局一致性的建筑设计图
 - 在计算资源受限的边缘设备上部署

**参考答案：**



 - **32×32图标**：U-Net。低分辨率下卷积的归纳偏置更有效，计算开销小。
 - **1024×1024人脸**：两者皆可。U-Net内存效率更高，但DiT在超大模型下质量可能更好。
 - **建筑设计图**：Transformer。需要强全局一致性，自注意力机制优势明显。
 - **边缘设备**：U-Net。参数量更少，计算效率更高，更适合部署。



实际选择还需考虑：训练数据量、具体质量要求、推理延迟限制等因素。








`# 待完成：完整的U-Net代码实现
import torch
import torch.nn as nn

class UNet(nn.Module):
 def __init__(self):
 super().__init__()
 # TODO: 实现完整的U-Net
 pass`




## 2.6 性能优化与实用技巧


 理论架构和实际部署之间往往存在巨大鸿沟。本节分享一些在实践中积累的优化技巧，这些技巧往往决定了模型能否真正落地。



### 2.6.1 内存优化：在GPU上塞下更大的模型



> **定义**
> 内存瓶颈分析


训练扩散模型时，内存主要消耗在：



 - **激活值**：U-Net的跳跃连接需要保存所有中间特征（占用最大）
 - **梯度**：反向传播需要的梯度存储
 - **优化器状态**：Adam需要存储一阶和二阶动量
 - **模型参数**：相对较小，但FP32下也不容忽视





#### 技巧1：梯度检查点（Gradient Checkpointing）


核心思想：用计算换内存。不保存中间激活值，反向传播时重新计算。




import torch.utils.checkpoint as checkpoint

class CheckpointedResBlock(nn.Module):
 def forward(self, x, time_emb):
 # 使用checkpoint包装计算密集但内存友好的部分
 def _forward(x):
 h = self.norm1(x)
 h = self.act(h)
 h = self.conv1(h)
 # ... 更多计算
 return h

 # 只在训练时使用checkpoint
 if self.training:
 return checkpoint.checkpoint(_forward, x)
 else:
 return _forward(x)


 **经验法则**：在U-Net的每个分辨率级别使用1-2个checkpoint，可减少约40%内存，训练时间增加约20%。



#### 技巧2：混合精度训练（Mixed Precision）


使用FP16计算，FP32累积，充分利用现代GPU的Tensor Core：




from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
 optimizer.zero_grad()

 with autocast(): # 自动将合适的操作转为FP16
 noise_pred = model(noisy_images, timesteps)
 loss = F.mse_loss(noise_pred, noise)

 # 梯度缩放，防止FP16下溢
 scaler.scale(loss).backward()
 scaler.step(optimizer)
 scaler.update()


 **注意事项**：



 - LayerNorm和注意力机制的softmax保持FP32精度
 - 损失缩放（loss scaling）对稳定性至关重要
 - 某些操作（如上采样）可能需要显式转回FP32




#### 技巧3：注意力优化


自注意力是内存消耗大户，特别是在高分辨率特征图上：




 标准注意力内存复杂度：$O(N^2 \cdot d)$，其中$N = H \times W$




优化方案：



 - **Flash Attention**：融合计算，减少内存读写
 - **分块注意力**：将特征图分块，只在块内计算注意力
 - **线性注意力近似**：用Performer或Linformer降低复杂度




### 2.6.2 训练稳定性：让10亿参数模型稳定收敛



#### 初始化的艺术



> **定义**
> 扩散模型特殊的初始化需求


目标：让模型初始预测接近零均值高斯噪声，实现"恒等映射"



 - **最后一层零初始化**：

nn.init.zeros_(self.final_conv.weight)
nn.init.zeros_(self.final_conv.bias)

 - **残差分支缩放**：

```python
self.residual_scale = nn.Parameter(torch.zeros(1))
output = x + self.residual_scale * residual
```


 - **注意力输出零初始化**：初始时注意力不起作用





#### 数值稳定性技巧


 
 
 问题
 解决方案
 
 
 FP16训练NaN
 
 • 使用FP32 LayerNorm

 • 梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

 • 调整损失缩放因子
 
 
 
 注意力爆炸
 
 • 缩放点积注意力（必须！）

 • 注意力dropout

 • QK归一化：`q = F.normalize(q, dim=-1)`
 
 
 
 训练后期不稳定
 
 • EMA权重平均

 • 学习率预热和余弦退火

 • AdamW权重衰减（通常0.01）
 
 
 


#### 调试技巧：如何定位问题



> **定义**
> 扩散模型调试检查清单



 - **单步测试**：固定 $t=500$，检查模型是否能学会去噪中等噪声
 - **可视化中间激活**：

```python
# 在forward中加入
if hasattr(self, 'debug_mode') and self.debug_mode:
 self.activations[f'layer_{i}'] = h.detach()
```


 - **监控关键指标**：


 各层激活值的均值和方差
 - 梯度范数（应该随深度递减）
 - 注意力权重分布（不应过度集中）


 
 - **噪声调度验证**：确认 $t=T$ 时 $x_T \approx \mathcal{N}(0,I)$





### 2.6.3 架构搜索：如何选择超参数


 没有一种架构适合所有任务。以下是一些经过验证的经验规则：




#### U-Net超参数选择指南

 
 
 图像分辨率
 基础通道数
 下采样次数
 注意力分辨率
 
 
 32×32
 128
 2
 16×16
 
 
 64×64
 128-256
 3
 16×16, 32×32
 
 
 256×256
 256-320
 4-5
 16×16, 32×32
 
 
 512×512+
 320-512
 5-6
 考虑使用DiT
 
 



**最后的建议**：从小模型开始，逐步扩大。过早使用大模型只会让调试变得困难。记住，Stable Diffusion的成功很大程度上归功于其在64×64潜在空间而非512×512像素空间上操作的设计决策。




综合练习：设计适合特定数据的去噪网络

[待完成：给定数据特性，选择和修改网络架构]

[待完成：设计思路和实现]






## 本章小结


本章我们追溯了扩散模型中两种主流架构的历史渊源：



 - **U-Net**：从2015年的医学图像分割任务，到2020年成为DDPM的核心架构
 - **Vision Transformer**：从2017年的NLP革命，经2020年的CV突破，到2022年DiT的提出




这两种架构能够成功应用于扩散模型并非偶然：



 - U-Net的编码器-解码器结构天然适合"加噪-去噪"的对称过程
 - ViT的全局注意力机制正好满足扩散模型对长程依赖建模的需求
 - 两种架构都经过改造：加入时间嵌入、调整归一化策略、优化注意力机制




扩散模型的成功启示我们：创新并不总是需要"从零开始"，善于发现和利用已有技术的潜力，通过巧妙的改造和组合，往往能产生意想不到的突破。



下一章，我们将深入DDPM的数学原理，看看这些架构是如何在具体的扩散模型训练中发挥作用的。