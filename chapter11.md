[← 返回目录](index.md) | 第11章 / 共14章 | [下一章 →](chapter12.md)

# 第11章：视频扩散模型

视频生成是扩散模型面临的最具挑战性的任务之一。与静态图像不同，视频需要在时间维度上保持连贯性，同时处理更高维度的数据。本章将深入探讨视频扩散模型的核心技术，从时序建模的基本原理到3D架构设计，再到运动动力学的建模。您将学习如何处理时间一致性、运动模糊、长程依赖等视频特有的挑战，并掌握设计高效视频生成系统的关键技术。通过本章的学习，您将理解Sora、Runway等前沿视频生成模型背后的技术原理。

## 章节大纲

### 11.1 视频生成的挑战与机遇
- 时序一致性要求
- 计算和内存瓶颈
- 运动表示与建模
- 数据集与评估指标

### 11.2 时序扩散模型架构
- 3D U-Net与因子化卷积
- 时空注意力机制
- 帧间信息传播
- 分层时序建模

### 11.3 条件控制与运动引导
- 文本到视频生成
- 图像动画化
- 运动轨迹控制
- 风格与内容解耦

### 11.4 高效训练与推理策略
- 视频压缩与潜在空间
- 级联生成框架
- 帧插值与超分辨率
- 分布式训练技术

### 11.5 应用案例与未来方向
- 视频编辑与修复
- 虚拟现实内容生成
- 实时视频合成
- 多模态视频理解

## 11.1 视频生成的挑战与机遇

### 11.1.1 时序一致性要求

视频生成的核心挑战是保持时间上的连贯性：

**1. 对象持续性**：
- 物体身份在帧间保持一致
- 外观特征（颜色、纹理）稳定
- 形状变化符合物理规律

**2. 运动连续性**：
- 轨迹平滑自然
- 速度和加速度合理
- 遮挡关系正确

**3. 光照一致性**：
- 阴影随物体移动
- 反射和高光稳定
- 环境光照渐变

```python
class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失"""
    def __init__(self, flow_model=None):
        super().__init__()
        self.flow_model = flow_model  # 光流估计模型
        self.perceptual = lpips.LPIPS(net='vgg')
        
    def forward(self, video_pred, video_gt=None):
        """
        video_pred: [B, T, C, H, W]
        """
        losses = {}
        
        # 1. 短程一致性（相邻帧）
        frame_diff = video_pred[:, 1:] - video_pred[:, :-1]
        losses['short_term'] = torch.abs(frame_diff).mean()
        
        # 2. 光流一致性
        if self.flow_model is not None:
            for t in range(video_pred.shape[1] - 1):
                flow = self.flow_model(video_pred[:, t], video_pred[:, t+1])
                warped = self.warp_frame(video_pred[:, t], flow)
                losses[f'flow_{t}'] = F.l1_loss(warped, video_pred[:, t+1])
        
        # 3. 感知一致性
        perceptual_loss = 0
        for t in range(video_pred.shape[1] - 1):
            perceptual_loss += self.perceptual(
                video_pred[:, t], 
                video_pred[:, t+1]
            ).mean()
        losses['perceptual'] = perceptual_loss
        
        return losses
```

💡 **关键洞察：时序正则化的重要性**  
单纯的帧级损失会导致闪烁。必须显式地鼓励时序平滑性，但过度平滑会失去运动细节。平衡是关键。

### 11.1.2 计算和内存瓶颈

视频数据的高维特性带来巨大挑战：

**维度爆炸**：
- 图像：`[B, C, H, W]` → 4D张量
- 视频：`[B, T, C, H, W]` → 5D张量
- 内存需求：T倍增长

**计算复杂度分析**：
```python
def estimate_video_memory(batch_size, frames, height, width, channels=3):
    """估算视频模型内存需求"""
    # 输入视频
    input_size = batch_size * frames * channels * height * width * 4  # float32
    
    # 3D卷积特征（假设最大通道数1024）
    feature_size = batch_size * frames * 1024 * (height//8) * (width//8) * 4
    
    # 时空注意力
    seq_len = frames * (height//16) * (width//16)
    attention_size = batch_size * seq_len * seq_len * 4
    
    total_gb = (input_size + feature_size + attention_size) / (1024**3)
    
    print(f"输入: {input_size / 1024**3:.2f} GB")
    print(f"特征: {feature_size / 1024**3:.2f} GB")
    print(f"注意力: {attention_size / 1024**3:.2f} GB")
    print(f"总计: {total_gb:.2f} GB")
    
    return total_gb

# 示例：16帧 512x512 视频
estimate_video_memory(1, 16, 512, 512)
# 输出：总计约 20GB！
```

🔬 **研究线索：高效时空表示**  
如何设计更高效的时空表示？分解方法（空间+时间）、稀疏表示、或是全新的架构？这是活跃的研究领域。

### 11.1.3 运动表示与建模

**运动的多尺度特性**：
1. **像素级运动**：光流、形变场
2. **对象级运动**：轨迹、旋转、缩放
3. **场景级运动**：相机运动、视角变化

**运动表示方法**：
```python
class MotionRepresentation:
    """不同的运动表示方法"""
    
    @staticmethod
    def optical_flow(frame1, frame2):
        """光流表示"""
        # 使用预训练的光流模型
        flow = flow_model(frame1, frame2)
        return flow  # [B, 2, H, W]
    
    @staticmethod
    def motion_patches(video):
        """运动块表示"""
        # 计算时空梯度
        dt = video[:, 1:] - video[:, :-1]  # 时间导数
        dx = video[:, :, :, :, 1:] - video[:, :, :, :, :-1]  # 空间导数x
        dy = video[:, :, :, 1:] - video[:, :, :, :-1]  # 空间导数y
        
        # 运动能量
        motion_energy = torch.sqrt(dt**2 + dx**2 + dy**2)
        return motion_energy
    
    @staticmethod
    def trajectory_encoding(keypoints_sequence):
        """轨迹编码"""
        # 将关键点序列编码为轨迹特征
        velocities = keypoints_sequence[1:] - keypoints_sequence[:-1]
        accelerations = velocities[1:] - velocities[:-1]
        
        return {
            'positions': keypoints_sequence,
            'velocities': velocities,
            'accelerations': accelerations
        }
```

### 11.1.4 数据集与评估指标

**主要数据集**：

| 数据集 | 规模 | 分辨率 | 特点 |
|--------|------|---------|------|
| UCF-101 | 13K videos | 240p | 人类动作 |
| Kinetics | 650K videos | 变化 | 多样动作 |
| WebVid-10M | 10M videos | 360p | 文本配对 |
| HD-VILA-100M | 100M videos | 720p | 高质量 |

**评估指标**：
```python
class VideoMetrics:
    """视频生成质量评估"""
    
    def __init__(self):
        self.i3d_model = load_i3d_model()  # 用于FVD
        self.flow_model = load_flow_model()  # 用于时序指标
        
    def fvd(self, real_videos, fake_videos):
        """Fréchet Video Distance"""
        # 提取I3D特征
        real_features = self.i3d_model(real_videos)
        fake_features = self.i3d_model(fake_videos)
        
        # 计算FVD（类似FID）
        mu_real = real_features.mean(0)
        mu_fake = fake_features.mean(0)
        sigma_real = torch.cov(real_features.T)
        sigma_fake = torch.cov(fake_features.T)
        
        return calculate_frechet_distance(
            mu_real, sigma_real, mu_fake, sigma_fake
        )
    
    def temporal_consistency(self, video):
        """时序一致性评分"""
        consistency_scores = []
        
        for t in range(len(video) - 1):
            # 计算光流
            flow = self.flow_model(video[t], video[t+1])
            
            # warp误差
            warped = warp_frame(video[t], flow)
            error = torch.abs(warped - video[t+1]).mean()
            
            consistency_scores.append(1.0 / (1.0 + error))
            
        return torch.tensor(consistency_scores).mean()
```

<details>
<summary>**练习 11.1：分析视频生成的挑战**</summary>

深入理解视频生成的独特挑战。

1. **时序建模实验**：
   - 实现简单的帧插值基线
   - 测试不同的时序一致性损失
   - 分析失败案例（闪烁、漂移等）

2. **内存优化探索**：
   - 比较不同的视频表示（RGB vs 光流）
   - 实现梯度检查点减少内存
   - 测试混合精度训练效果

3. **运动分析**：
   - 可视化不同类型的运动模式
   - 实现运动分解（全局vs局部）
   - 研究运动先验的作用

4. **数据集构建**：
   - 设计视频质量筛选pipeline
   - 实现高效的视频预处理
   - 创建专门的评测基准

</details>

### 11.1.5 视频扩散的独特机遇

**1. 强大的时序先验**：
- 物理规律（重力、惯性）
- 因果关系
- 周期性模式

**2. 多模态信息**：
- 视觉+音频同步
- 文本描述的时序结构
- 动作标签序列

**3. 分层表示**：
```python
class HierarchicalVideoRepresentation:
    """分层视频表示"""
    
    def __init__(self):
        self.scene_encoder = SceneEncoder()      # 场景级
        self.object_encoder = ObjectEncoder()    # 对象级
        self.motion_encoder = MotionEncoder()    # 运动级
        self.texture_encoder = TextureEncoder()  # 纹理级
        
    def encode(self, video):
        # 分层编码
        scene_features = self.scene_encoder(video)
        object_features = self.object_encoder(video)
        motion_features = self.motion_encoder(video)
        texture_features = self.texture_encoder(video)
        
        return {
            'scene': scene_features,      # 全局场景信息
            'objects': object_features,   # 对象身份和位置
            'motion': motion_features,    # 运动模式
            'texture': texture_features   # 精细纹理细节
        }
```

🌟 **前沿思考：视频理解与生成的统一**  
视频理解模型（如VideoMAE）的表示能否直接用于生成？如何设计既能理解又能生成的统一架构？

### 11.1.6 技术路线选择

**主要技术路线对比**：

1. **直接3D扩散**：
   - 优点：端到端建模
   - 缺点：计算量巨大

2. **级联生成**：
   - 优点：分而治之，易于控制
   - 缺点：误差累积

3. **潜在空间扩散**：
   - 优点：高效
   - 缺点：需要好的视频编码器

4. **混合方法**：
   ```python
   class HybridVideoGeneration:
       def __init__(self):
           # 关键帧生成器（2D扩散）
           self.keyframe_generator = ImageDiffusion()
           
           # 运动生成器（轻量3D）
           self.motion_generator = MotionDiffusion()
           
           # 细节填充器
           self.detail_filler = DetailRefinement()
       
       def generate(self, text_prompt, num_frames):
           # 1. 生成关键帧
           keyframes = self.keyframe_generator(
               text_prompt, 
               num_keyframes=num_frames // 4
           )
           
           # 2. 生成运动
           motion = self.motion_generator(
               keyframes, 
               text_prompt
           )
           
           # 3. 填充细节
           full_video = self.detail_filler(
               keyframes, 
               motion
           )
           
           return full_video
   ```

接下来，我们将深入探讨具体的模型架构设计...

## 11.2 时序扩散模型架构

### 11.2.1 3D U-Net与因子化卷积

将2D U-Net扩展到3D是最直接的方法，但需要仔细设计以控制参数量：

**完整3D卷积**：
```python
class Full3DConv(nn.Module):
    """完整的3D卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            padding=kernel_size//2
        )
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        return self.conv(x)
```

**因子化卷积（更高效）**：
```python
class FactorizedConv3D(nn.Module):
    """(2+1)D因子化卷积：空间卷积 + 时间卷积"""
    def __init__(self, in_channels, out_channels, spatial_kernel=3, temporal_kernel=3):
        super().__init__()
        # 空间卷积
        self.spatial_conv = nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel//2, spatial_kernel//2)
        )
        
        # 时间卷积
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel//2, 0, 0)
        )
        
        self.norm = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        # 先空间后时间
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
```

**伪3D卷积（Pseudo-3D）**：
```python
class Pseudo3DBlock(nn.Module):
    """交替进行2D空间和1D时间处理"""
    def __init__(self, channels, num_frames):
        super().__init__()
        # 空间处理（跨帧共享）
        self.spatial_block = ResBlock2D(channels)
        
        # 时间混合
        self.temporal_mixer = nn.Conv1d(
            channels * num_frames,
            channels * num_frames,
            kernel_size=3,
            padding=1,
            groups=channels  # 按通道分组
        )
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 空间处理：合并batch和time维度
        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        x_2d = self.spatial_block(x_2d)
        x = rearrange(x_2d, '(b t) c h w -> b c t h w', b=B, t=T)
        
        # 时间混合：将空间维度展平
        x_1d = rearrange(x, 'b c t h w -> b (c h w) t')
        x_1d = self.temporal_mixer(x_1d)
        x = rearrange(x_1d, 'b (c h w) t -> b c t h w', c=C, h=H, w=W)
        
        return x
```

💡 **设计权衡：计算效率 vs 表达能力**  
- 完整3D：最强表达力，计算量 O(k³)
- 因子化：平衡选择，计算量 O(k² + k)
- 伪3D：最高效，但时空交互受限

### 11.2.2 时空注意力机制

注意力在视频模型中至关重要，但需要精心设计以控制复杂度：

**全时空注意力（计算密集）**：
```python
class FullSpatioTemporalAttention(nn.Module):
    """对所有时空位置计算注意力"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 展平为序列
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        
        # 自注意力（复杂度：O(T²H²W²)）
        x, _ = self.attention(x, x, x)
        
        # 恢复形状
        x = rearrange(x, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
        return x
```

**分解的时空注意力（高效）**：
```python
class FactorizedSpatioTemporalAttention(nn.Module):
    """先空间注意力，后时间注意力"""
    def __init__(self, dim, spatial_heads=8, temporal_heads=4):
        super().__init__()
        self.spatial_attn = SpatialAttention(dim, spatial_heads)
        self.temporal_attn = TemporalAttention(dim, temporal_heads)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        
        # 空间注意力（在每帧内）
        x = self.spatial_attn(x)
        
        # 时间注意力（跨帧）
        x = self.temporal_attn(x)
        
        return x

class SpatialAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 对每帧独立计算空间注意力
        outputs = []
        for t in range(T):
            frame = x[:, :, t, :, :]  # [B, C, H, W]
            frame = rearrange(frame, 'b c h w -> b (h w) c')
            frame_out, _ = self.attention(frame, frame, frame)
            frame_out = rearrange(frame_out, 'b (h w) c -> b c h w', h=H, w=W)
            outputs.append(frame_out)
            
        return torch.stack(outputs, dim=2)

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 对每个空间位置计算时间注意力
        x = rearrange(x, 'b c t h w -> (b h w) t c')
        x, _ = self.attention(x, x, x)
        x = rearrange(x, '(b h w) t c -> b c t h w', b=B, h=H, w=W)
        
        return x
```

**分块时空注意力（内存友好）**：
```python
class WindowedSpatioTemporalAttention(nn.Module):
    """窗口内的时空注意力，类似Video Swin Transformer"""
    def __init__(self, dim, window_size=(4, 7, 7), heads=8):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(dim, heads)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        Tw, Hw, Ww = self.window_size
        
        # 划分窗口
        x = rearrange(x, 'b c (tw nw) (hw ph) (ww pw) -> b (nw ph pw) (tw hw ww) c',
                     tw=Tw, hw=Hw, ww=Ww)
        
        # 窗口内注意力
        x, _ = self.attention(x, x, x)
        
        # 恢复形状
        x = rearrange(x, 'b (nw ph pw) (tw hw ww) c -> b c (tw nw) (hw ph) (ww pw)',
                     tw=Tw, hw=Hw, ww=Ww, nw=T//Tw, ph=H//Hw, pw=W//Ww)
        
        return x
```

🔬 **研究方向：自适应注意力模式**  
能否学习数据相关的注意力模式？例如，快速运动区域使用密集时间注意力，静态区域使用稀疏注意力。

### 11.2.3 帧间信息传播

确保信息在帧间有效流动是关键：

**循环连接**：
```python
class RecurrentPropagation(nn.Module):
    """使用循环结构传播时序信息"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 空间pooling获得帧级特征
        frame_features = x.mean(dim=(3, 4))  # [B, C, T]
        frame_features = frame_features.transpose(1, 2)  # [B, T, C]
        
        # GRU处理
        hidden_states, _ = self.gru(frame_features)
        
        # 广播回空间维度
        hidden_states = hidden_states.transpose(1, 2)  # [B, C, T]
        hidden_states = hidden_states.unsqueeze(3).unsqueeze(4)
        hidden_states = hidden_states.expand(-1, -1, -1, H, W)
        
        return x + hidden_states
```

**双向传播**：
```python
class BidirectionalPropagation(nn.Module):
    """双向传播确保前后文信息"""
    def __init__(self, channels):
        super().__init__()
        self.forward_conv = nn.Conv3d(
            channels, channels, 
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        self.backward_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0)
        )
        
    def forward(self, x):
        # 前向传播
        forward_out = self.forward_conv(x)
        
        # 后向传播（翻转时间维度）
        x_flip = torch.flip(x, dims=[2])
        backward_out = self.backward_conv(x_flip)
        backward_out = torch.flip(backward_out, dims=[2])
        
        # 融合双向信息
        return x + 0.5 * (forward_out + backward_out)
```

<details>
<summary>**练习 11.2：设计高效的视频架构**</summary>

探索不同的架构设计选择。

1. **架构比较**：
   - 实现3种不同的3D卷积变体
   - 比较参数量、FLOPs和内存使用
   - 在小数据集上测试性能

2. **注意力优化**：
   - 实现稀疏注意力模式
   - 测试不同的分解策略
   - 分析注意力图的时空模式

3. **信息流分析**：
   - 可视化特征在时间维度的传播
   - 测量有效感受野
   - 识别信息瓶颈

4. **混合架构设计**：
   - 结合CNN和Transformer的优势
   - 设计自适应的计算分配
   - 探索早期融合vs晚期融合

</details>

### 11.2.4 分层时序建模

不同时间尺度需要不同的处理策略：

```python
class HierarchicalTemporalModeling(nn.Module):
    """多尺度时序建模"""
    def __init__(self, base_channels=256):
        super().__init__()
        
        # 短程建模（1-2帧）
        self.short_term = ShortTermModule(base_channels)
        
        # 中程建模（4-8帧）  
        self.mid_term = MidTermModule(base_channels * 2)
        
        # 长程建模（16+帧）
        self.long_term = LongTermModule(base_channels * 4)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        
        # 短程特征
        short_features = self.short_term(x)
        
        # 中程特征（降采样时间维度）
        x_mid = F.avg_pool3d(x, kernel_size=(2, 1, 1))
        mid_features = self.mid_term(x_mid)
        mid_features = F.interpolate(mid_features, size=x.shape[2:])
        
        # 长程特征（更激进的降采样）
        x_long = F.avg_pool3d(x, kernel_size=(4, 2, 2))
        long_features = self.long_term(x_long)
        long_features = F.interpolate(long_features, size=x.shape[2:])
        
        # 多尺度融合
        return short_features + mid_features + long_features

class ShortTermModule(nn.Module):
    """处理快速运动和细节变化"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
    def forward(self, x):
        return self.conv(x)

class MidTermModule(nn.Module):
    """处理对象运动和交互"""
    def __init__(self, channels):
        super().__init__()
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0)
        )
        self.spatial_attn = SpatialAttention(channels, heads=8)
        
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_attn(x)
        return x

class LongTermModule(nn.Module):
    """处理场景变化和全局模式"""
    def __init__(self, channels):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(channels, num_heads=8)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        # 全局池化空间维度
        x_pooled = x.mean(dim=(3, 4))  # [B, C, T]
        x_pooled = x_pooled.permute(2, 0, 1)  # [T, B, C]
        
        # 时间注意力
        attn_out, _ = self.temporal_attn(x_pooled, x_pooled, x_pooled)
        attn_out = attn_out.permute(1, 2, 0)  # [B, C, T]
        
        # 广播回原始形状
        attn_out = attn_out.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, H, W)
        
        return attn_out
```

### 11.2.5 Video DiT架构

将DiT扩展到视频领域：

```python
class VideoDiT(nn.Module):
    """Video Diffusion Transformer"""
    def __init__(
        self,
        spatial_patch_size=8,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        # 时空patchify
        self.patchify = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 输出投影
        self.final_layer = FinalLayer(hidden_size, spatial_patch_size, temporal_patch_size)
        
    def forward(self, x, t, y=None):
        # x: [B, C, T, H, W]
        # t: 时间步
        # y: 条件（如类别标签）
        
        # Patchify
        x = self.patchify(x)  # [B, hidden_size, T', H', W']
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 时间和条件编码
        t_emb = timestep_embedding(t, self.hidden_size)
        if y is not None:
            y_emb = self.class_embedder(y)
            c = t_emb + y_emb
        else:
            c = t_emb
            
        # Transformer处理
        for block in self.blocks:
            x = block(x, c)
            
        # 输出
        x = self.final_layer(x, c)
        
        # Unpatchify
        x = rearrange(x, 'b (t h w) c -> b c t h w', 
                     t=T//self.temporal_patch_size,
                     h=H//self.spatial_patch_size,
                     w=W//self.spatial_patch_size)
        
        return x
```

🌟 **前沿探索：视频生成的扩展定律**  
DiT证明了图像生成的扩展定律。视频生成是否有类似规律？时间维度如何影响扩展？这是开放的研究问题。

### 11.2.6 轻量级视频架构

对于实时或移动应用，需要更轻量的设计：

```python
class LightweightVideoGenerator(nn.Module):
    """轻量级视频生成架构"""
    def __init__(self, base_channels=64):
        super().__init__()
        
        # 共享的2D backbone
        self.shared_backbone = MobileNetV3()
        
        # 轻量时间建模
        self.temporal_module = DepthwiseSeparableConv3D(
            base_channels, base_channels
        )
        
        # 运动预测头
        self.motion_head = nn.Conv3d(
            base_channels, 2,  # 2D motion vectors
            kernel_size=1
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 使用共享backbone处理每帧
        features = []
        for t in range(T):
            feat = self.shared_backbone(x[:, :, t])
            features.append(feat)
        features = torch.stack(features, dim=2)
        
        # 轻量时间处理
        temporal_features = self.temporal_module(features)
        
        # 预测运动
        motion = self.motion_head(temporal_features)
        
        # 基于运动的高效生成
        output = self.motion_based_synthesis(x[:, :, 0], motion)
        
        return output

class DepthwiseSeparableConv3D(nn.Module):
    """深度可分离3D卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels
        )
        # 逐点卷积
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

💡 **实践建议：架构选择指南**  
- 高质量离线生成：使用完整3D架构
- 实时应用：使用因子化或伪3D
- 移动设备：使用共享backbone + 轻量时间模块
- 长视频：使用分层架构避免内存爆炸

## 11.3 条件控制与运动引导

### 11.3.1 文本到视频生成

文本条件是视频生成最重要的控制方式：

**时序感知的文本编码**：
```python
class TemporalTextEncoder(nn.Module):
    """编码包含时序信息的文本描述"""
    def __init__(self, base_encoder='clip', temporal_layers=4):
        super().__init__()
        # 基础文本编码器
        if base_encoder == 'clip':
            self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        elif base_encoder == 't5':
            self.text_encoder = T5EncoderModel.from_pretrained('t5-large')
        
        # 时序增强层
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=3072
            ) for _ in range(temporal_layers)
        ])
        
        # 时序标记
        self.temporal_tokens = nn.Parameter(torch.randn(1, 16, 768))  # 16个时序位置
        
    def forward(self, text, frame_indices=None):
        # 基础文本编码
        text_features = self.text_encoder(text).last_hidden_state
        
        # 添加时序标记
        if frame_indices is not None:
            # 根据帧索引选择时序标记
            temporal_embeds = self.temporal_tokens[:, frame_indices]
            text_features = text_features + temporal_embeds
        
        # 时序增强
        for layer in self.temporal_layers:
            text_features = layer(text_features)
            
        return text_features
```

**动作词提取与对齐**：
```python
class ActionAlignment(nn.Module):
    """对齐文本中的动作描述与视频时序"""
    def __init__(self):
        super().__init__()
        # 动作词检测
        self.action_detector = ActionWordDetector()
        
        # 时序分配网络
        self.temporal_allocator = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 预测时间位置
            nn.Sigmoid()
        )
        
    def forward(self, text_features, num_frames):
        # 检测动作词
        action_masks = self.action_detector(text_features)
        action_features = text_features * action_masks.unsqueeze(-1)
        
        # 分配到时间轴
        temporal_positions = self.temporal_allocator(action_features)
        temporal_positions = temporal_positions * num_frames
        
        # 生成时序attention mask
        attention_weights = self.gaussian_attention(
            temporal_positions, 
            num_frames, 
            sigma=2.0
        )
        
        return attention_weights
    
    def gaussian_attention(self, centers, length, sigma):
        """生成高斯形状的注意力权重"""
        positions = torch.arange(length).float()
        weights = []
        
        for center in centers:
            weight = torch.exp(-(positions - center)**2 / (2 * sigma**2))
            weights.append(weight)
            
        return torch.stack(weights)
```

💡 **关键技巧：时序提示工程**  
有效的视频生成提示需要包含：
- 明确的时序词汇（"首先"、"然后"、"最后"）
- 动作的持续时间（"缓慢地"、"快速地"）
- 运动方向（"从左到右"、"向上"）

### 11.3.2 图像动画化

将静态图像转换为动态视频：

**图像编码与运动预测**：
```python
class Image2Video(nn.Module):
    """图像到视频生成"""
    def __init__(self, image_encoder, motion_predictor, video_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.motion_predictor = motion_predictor
        self.video_decoder = video_decoder
        
    def forward(self, image, text_prompt=None, num_frames=16):
        # 编码输入图像
        image_features = self.image_encoder(image)
        
        # 预测运动模式
        if text_prompt is not None:
            # 文本引导的运动
            motion_params = self.motion_predictor(
                image_features, 
                text_prompt
            )
        else:
            # 自动运动预测
            motion_params = self.motion_predictor(image_features)
        
        # 生成视频帧
        frames = [image]  # 第一帧是原始图像
        current_features = image_features
        
        for t in range(1, num_frames):
            # 应用运动
            current_features = self.apply_motion(
                current_features, 
                motion_params, 
                t
            )
            
            # 解码为图像
            frame = self.video_decoder(current_features)
            frames.append(frame)
            
        return torch.stack(frames, dim=1)
    
    def apply_motion(self, features, motion_params, t):
        """应用运动变换"""
        # 提取运动参数
        translation = motion_params['translation'] * t
        rotation = motion_params['rotation'] * t
        scaling = motion_params['scaling'] ** t
        
        # 应用空间变换
        transformed = self.spatial_transform(
            features, 
            translation, 
            rotation, 
            scaling
        )
        
        return transformed
```

**运动类型分解**：
```python
class MotionDecomposition(nn.Module):
    """将复杂运动分解为基本组件"""
    def __init__(self):
        super().__init__()
        # 全局运动（相机运动）
        self.global_motion = GlobalMotionPredictor()
        
        # 局部运动（对象运动）
        self.local_motion = LocalMotionPredictor()
        
        # 形变运动（非刚性变形）
        self.deformation = DeformationPredictor()
        
    def forward(self, image_features, motion_type='mixed'):
        motions = {}
        
        if motion_type in ['global', 'mixed']:
            # 预测相机运动
            motions['camera'] = self.global_motion(image_features)
            
        if motion_type in ['local', 'mixed']:
            # 预测对象运动
            motions['objects'] = self.local_motion(image_features)
            
        if motion_type in ['deform', 'mixed']:
            # 预测形变
            motions['deformation'] = self.deformation(image_features)
            
        return motions

class GlobalMotionPredictor(nn.Module):
    """预测相机运动参数"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)  # 6-DOF相机运动
        )
        
    def forward(self, features):
        encoded = self.encoder(features).squeeze(-1).squeeze(-1)
        motion_params = self.predictor(encoded)
        
        return {
            'translation': motion_params[:, :3],
            'rotation': motion_params[:, 3:]
        }
```

🔬 **研究挑战：运动的歧义性**  
同一张图像可能对应多种合理的运动。如何处理这种多模态性？可以使用变分方法或条件流匹配来建模运动分布。

### 11.3.3 运动轨迹控制

精确控制视频中的运动路径：

**轨迹表示与编码**：
```python
class TrajectoryEncoder(nn.Module):
    """编码用户指定的运动轨迹"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        # 轨迹点编码
        self.point_encoder = nn.Sequential(
            nn.Linear(2, 64),  # 2D坐标
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # 时序编码
        self.temporal_encoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=2,
            bidirectional=True
        )
        
        # 轨迹到特征图的映射
        self.trajectory_to_feature = nn.Conv2d(
            hidden_dim * 2,  # 双向LSTM
            256,
            kernel_size=1
        )
        
    def forward(self, trajectory_points, num_frames, feature_size):
        # trajectory_points: [B, N, 2] N个轨迹点的坐标
        
        # 编码轨迹点
        point_features = self.point_encoder(trajectory_points)
        
        # 时序编码
        lstm_out, _ = self.temporal_encoder(point_features)
        
        # 插值到目标帧数
        trajectory_features = F.interpolate(
            lstm_out.transpose(1, 2),
            size=num_frames,
            mode='linear'
        ).transpose(1, 2)
        
        # 转换为空间特征图
        spatial_features = self.render_trajectory_map(
            trajectory_features, 
            trajectory_points,
            feature_size
        )
        
        return spatial_features
    
    def render_trajectory_map(self, features, points, size):
        """将轨迹渲染为特征图"""
        H, W = size
        feature_map = torch.zeros(features.shape[0], features.shape[-1], H, W)
        
        for b in range(features.shape[0]):
            for t, (x, y) in enumerate(points[b]):
                # 将轨迹点转换为特征图坐标
                x_idx = int(x * W)
                y_idx = int(y * H)
                
                # 高斯散布
                self.gaussian_splat(
                    feature_map[b], 
                    features[b, t], 
                    x_idx, 
                    y_idx,
                    sigma=3.0
                )
                
        return feature_map
```

**稀疏控制点插值**：
```python
class SparseControlInterpolation(nn.Module):
    """从稀疏控制点生成密集运动场"""
    def __init__(self):
        super().__init__()
        self.flow_basis = FlowBasisNetwork()
        self.interpolator = BilinearInterpolator()
        
    def forward(self, control_points, control_flows, image_size):
        """
        control_points: [B, K, 2] K个控制点位置
        control_flows: [B, K, T, 2] 每个控制点的运动向量
        """
        B, K, T, _ = control_flows.shape
        H, W = image_size
        
        # 生成流基函数
        basis_flows = self.flow_basis(control_points, image_size)
        
        # 稀疏到密集插值
        dense_flows = []
        for t in range(T):
            # 当前时刻的控制流
            flows_t = control_flows[:, :, t, :]
            
            # 加权组合基函数
            weights = self.compute_weights(control_points, (H, W))
            dense_flow = torch.sum(
                basis_flows * weights.unsqueeze(-1).unsqueeze(-1),
                dim=1
            )
            
            dense_flows.append(dense_flow)
            
        return torch.stack(dense_flows, dim=1)

class DragGAN(nn.Module):
    """基于拖拽的视频编辑"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.motion_predictor = MotionPredictor()
        self.warping_module = WarpingModule()
        
    def forward(self, video, source_points, target_points):
        """
        source_points: 用户选择的源点
        target_points: 用户拖拽的目标点
        """
        # 提取特征
        features = self.feature_extractor(video)
        
        # 预测整体运动场
        motion_field = self.motion_predictor(
            features, 
            source_points, 
            target_points
        )
        
        # 应用运动
        edited_video = self.warping_module(video, motion_field)
        
        return edited_video
```

<details>
<summary>**练习 11.3：实现交互式视频控制**</summary>

设计和实现各种视频控制机制。

1. **文本控制实验**：
   - 实现时序感知的文本编码器
   - 测试不同的动作词对齐策略
   - 评估生成视频与文本的一致性

2. **运动轨迹设计**：
   - 实现基于贝塞尔曲线的轨迹
   - 支持多对象独立轨迹
   - 处理轨迹冲突和遮挡

3. **交互式编辑**：
   - 实现拖拽式视频编辑
   - 支持局部区域的运动控制
   - 保持未编辑区域的稳定性

4. **多模态控制**：
   - 结合文本、轨迹、参考视频
   - 设计控制信号的融合策略
   - 处理冲突的控制指令

</details>

### 11.3.4 风格与内容解耦

分离视频的内容（什么）和风格（如何）：

```python
class StyleContentDiSffusion(nn.Module):
    """风格-内容解耦的视频扩散模型"""
    def __init__(self):
        super().__init__()
        # 内容编码器（提取语义和结构）
        self.content_encoder = ContentEncoder()
        
        # 风格编码器（提取视觉风格）
        self.style_encoder = StyleEncoder()
        
        # 运动编码器（提取运动模式）
        self.motion_encoder = MotionEncoder()
        
        # 解耦的去噪网络
        self.denoiser = DisentangledDenoiser()
        
    def forward(self, x_t, t, content_ref=None, style_ref=None, motion_ref=None):
        # 提取各种参考特征
        if content_ref is not None:
            content_features = self.content_encoder(content_ref)
        else:
            content_features = None
            
        if style_ref is not None:
            style_features = self.style_encoder(style_ref)
        else:
            style_features = None
            
        if motion_ref is not None:
            motion_features = self.motion_encoder(motion_ref)
        else:
            motion_features = None
            
        # 解耦去噪
        noise_pred = self.denoiser(
            x_t, t,
            content=content_features,
            style=style_features,
            motion=motion_features
        )
        
        return noise_pred

class DisentangledDenoiser(nn.Module):
    """支持风格-内容-运动解耦的去噪器"""
    def __init__(self, base_channels=256):
        super().__init__()
        # 基础U-Net
        self.unet = UNet3D(base_channels)
        
        # 条件注入模块
        self.content_inject = FiLMLayer(base_channels)
        self.style_inject = AdaIN(base_channels)
        self.motion_inject = SpatioTemporalModulation(base_channels)
        
    def forward(self, x_t, t, content=None, style=None, motion=None):
        # 基础特征提取
        features = self.unet.encoder(x_t, t)
        
        # 注入各种条件
        if content is not None:
            features = self.content_inject(features, content)
            
        if style is not None:
            features = self.style_inject(features, style)
            
        if motion is not None:
            features = self.motion_inject(features, motion)
            
        # 解码
        output = self.unet.decoder(features)
        
        return output
```

**时序一致的风格迁移**：
```python
class TemporalStyleTransfer(nn.Module):
    """保持时序一致性的视频风格迁移"""
    def __init__(self):
        super().__init__()
        self.style_extractor = VGGStyleExtractor()
        self.temporal_consistency = TemporalConsistencyModule()
        
    def forward(self, content_video, style_image):
        B, T, C, H, W = content_video.shape
        
        # 提取风格特征
        style_features = self.style_extractor(style_image)
        
        # 逐帧风格化
        stylized_frames = []
        prev_frame = None
        
        for t in range(T):
            frame = content_video[:, t]
            
            # 应用风格
            stylized = self.apply_style(frame, style_features)
            
            # 保持时序一致性
            if prev_frame is not None:
                stylized = self.temporal_consistency(
                    stylized, 
                    prev_frame,
                    content_video[:, t-1],
                    content_video[:, t]
                )
            
            stylized_frames.append(stylized)
            prev_frame = stylized
            
        return torch.stack(stylized_frames, dim=1)
```

### 11.3.5 细粒度属性控制

控制视频的特定属性：

```python
class AttributeController(nn.Module):
    """细粒度属性控制"""
    def __init__(self, attributes=['speed', 'direction', 'intensity']):
        super().__init__()
        self.attributes = attributes
        
        # 每个属性的编码器
        self.encoders = nn.ModuleDict({
            attr: self.build_encoder(attr) 
            for attr in attributes
        })
        
        # 属性融合
        self.fusion = AttributeFusion(len(attributes))
        
    def build_encoder(self, attribute):
        if attribute == 'speed':
            return SpeedEncoder()
        elif attribute == 'direction':
            return DirectionEncoder()
        elif attribute == 'intensity':
            return IntensityEncoder()
        else:
            return GenericAttributeEncoder()
            
    def forward(self, attribute_values):
        encoded_attrs = {}
        
        for attr, value in attribute_values.items():
            if attr in self.encoders:
                encoded_attrs[attr] = self.encoders[attr](value)
                
        # 融合所有属性
        control_signal = self.fusion(encoded_attrs)
        
        return control_signal

class SpeedController(nn.Module):
    """控制视频播放速度和运动速度"""
    def __init__(self):
        super().__init__()
        self.speed_embedding = nn.Embedding(10, 256)  # 10个速度级别
        self.temporal_modulator = TemporalModulator()
        
    def forward(self, video_features, speed_level):
        # 获取速度嵌入
        speed_emb = self.speed_embedding(speed_level)
        
        # 调制时间维度
        modulated = self.temporal_modulator(video_features, speed_emb)
        
        return modulated
```

🌟 **前沿方向：可组合的视频控制**  
如何设计一个统一框架，支持任意组合的控制信号（文本+轨迹+风格+属性）？这需要解决控制信号的对齐、融合和冲突解决。

### 11.3.6 物理约束与真实感

确保生成的运动符合物理规律：

```python
class PhysicsAwareGeneration(nn.Module):
    """物理感知的视频生成"""
    def __init__(self):
        super().__init__()
        # 物理模拟器
        self.physics_engine = DifferentiablePhysics()
        
        # 物理参数预测
        self.physics_predictor = nn.Sequential(
            nn.Conv3d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(64, 5, 1)  # 质量、摩擦、弹性等
        )
        
    def forward(self, initial_state, num_frames):
        # 预测物理参数
        physics_params = self.physics_predictor(initial_state)
        
        # 物理模拟
        states = [initial_state]
        for t in range(1, num_frames):
            # 应用物理规律
            next_state = self.physics_engine.step(
                states[-1], 
                physics_params,
                dt=1.0/30  # 30 FPS
            )
            states.append(next_state)
            
        return torch.stack(states, dim=1)

class DifferentiablePhysics(nn.Module):
    """可微分的物理模拟"""
    def __init__(self):
        super().__init__()
        
    def step(self, state, params, dt):
        # 提取物理量
        position = state['position']
        velocity = state['velocity']
        mass = params['mass']
        
        # 计算力（重力、摩擦等）
        forces = self.compute_forces(state, params)
        
        # 更新速度和位置（欧拉积分）
        acceleration = forces / mass
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        
        # 处理碰撞
        new_position, new_velocity = self.handle_collisions(
            new_position, 
            new_velocity,
            params
        )
        
        return {
            'position': new_position,
            'velocity': new_velocity
        }
```

通过这些条件控制机制，视频扩散模型可以生成高度可控和真实的动态内容。下一节将探讨如何高效地训练和部署这些模型。