[← 返回目录](index.md) | 第7章 / 共14章 | [下一章 →](chapter8.md)

# 第7章：扩散Transformer (DiT)

扩散Transformer（Diffusion Transformer, DiT）标志着扩散模型架构的范式转变。本章将深入探讨DiT如何将Transformer的强大表达能力和优秀的扩展性引入扩散模型，实现了从卷积架构到注意力架构的飞跃。您将理解DiT的核心设计原则，学习其与传统U-Net的关键差异，并掌握如何利用Transformer的缩放定律来构建更强大的生成模型。通过本章的学习，您将获得设计和训练大规模扩散模型的关键洞察，为理解Sora、Stable Diffusion 3等前沿模型打下基础。

## 章节大纲

### 7.1 DiT架构详解
- 从Vision Transformer到Diffusion Transformer
- DiT的核心组件：patchify、位置编码、时间条件
- 架构变体：DiT-S/B/L/XL的设计选择

### 7.2 与U-Net的对比分析
- 归纳偏置：卷积vs注意力
- 计算复杂度与内存效率
- 特征表示的差异

### 7.3 可扩展性分析
- 缩放定律在扩散模型中的体现
- 模型大小、数据量与性能的关系
- 训练效率与推理优化

### 7.4 条件机制与灵活性
- 自适应层归一化（AdaLN）
- 交叉注意力vs AdaLN-Zero
- 多模态条件的统一处理

### 7.5 实践考虑与未来方向
- 训练策略与超参数选择
- 混合精度训练与分布式训练
- 架构创新的研究方向

## 7.1 DiT架构详解