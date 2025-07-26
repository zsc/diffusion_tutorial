"""
扩散模型教程 - 核心库
这个库将随着教程章节逐步构建，提供扩散模型的基础组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============= 第1章：基础工具函数 =============

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    线性噪声调度
    
    Args:
        timesteps: 扩散步数
        beta_start: 起始beta值
        beta_end: 结束beta值
    
    Returns:
        betas: shape (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦噪声调度 (改进的调度策略)
    
    Args:
        timesteps: 扩散步数
        s: 偏移量，防止beta过小
    
    Returns:
        betas: shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def compute_alpha_schedule(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从beta计算alpha相关的值
    
    Args:
        betas: 噪声调度
    
    Returns:
        alphas: 1 - betas
        alphas_cumprod: 累积乘积
        sqrt_alphas_cumprod: 平方根
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


class DiffusionUtils:
    """扩散过程的工具类"""
    
    def __init__(self, timesteps: int = 1000, beta_schedule: str = 'linear'):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 预计算所有需要的值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于后验分布的计算
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod.roll(1)) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = self.posterior_variance[1]  # 第0步特殊处理
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程：从x_0采样x_t
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_start: 初始数据 x_0
            t: 时间步
            noise: 可选的噪声，如果不提供则采样
        
        Returns:
            x_t: 扩散后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 重参数化技巧
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def visualize_forward_process(self, x_start: torch.Tensor, steps: List[int] = None):
        """
        可视化前向扩散过程
        
        Args:
            x_start: 初始数据
            steps: 要展示的时间步
        """
        if steps is None:
            steps = [0, 250, 500, 750, 999]
        
        fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))
        
        for i, t in enumerate(steps):
            t_tensor = torch.tensor([t])
            x_t = self.q_sample(x_start, t_tensor)
            
            # 假设是图像数据，展示第一个样本
            if len(x_t.shape) == 4:  # B, C, H, W
                img = x_t[0].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
                axes[i].imshow(img)
            else:  # 其他类型的数据用散点图
                axes[i].scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu() if x_t.shape[1] > 1 else torch.zeros_like(x_t[:, 0]), alpha=0.5)
            
            axes[i].set_title(f't = {t}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============= 第2章：神经网络架构（待实现） =============
# U-Net 和 ViT 架构将在第2章中实现

class SimpleUNet(nn.Module):
    """简化版U-Net（占位符，将在第2章详细实现）"""
    def __init__(self, in_channels: int = 3, out_channels: int = 3, time_emb_dim: int = 32):
        super().__init__()
        # TODO: 在第2章实现完整的U-Net
        self.placeholder = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 占位实现
        return self.placeholder(x)


# ============= 第3章：DDPM（待实现） =============
# DDPM 训练和采样将在第3章中实现

class DDPM:
    """DDPM模型（占位符，将在第3章详细实现）"""
    def __init__(self, model: nn.Module, diffusion_utils: DiffusionUtils):
        self.model = model
        self.diffusion = diffusion_utils
    
    def train_step(self, x_0: torch.Tensor) -> torch.Tensor:
        """训练步骤（待实现）"""
        pass
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """采样（待实现）"""
        pass


# ============= 工具函数 =============

def plot_samples(samples: torch.Tensor, title: str = "Samples"):
    """绘制生成的样本"""
    if len(samples.shape) == 4:  # 图像
        n_samples = min(16, samples.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        
        for i in range(n_samples):
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[i].imshow(img)
            axes[i].axis('off')
    else:  # 2D数据
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试基础功能
    print("扩散模型核心库 - 第1章功能测试")
    
    # 测试噪声调度
    utils = DiffusionUtils(timesteps=1000, beta_schedule='cosine')
    print(f"Beta范围: {utils.betas[0]:.4f} - {utils.betas[-1]:.4f}")
    print(f"Alpha_bar_T: {utils.alphas_cumprod[-1]:.4f}")
    
    # 测试前向扩散
    x_0 = torch.randn(100, 2)  # 100个2D点
    x_t = utils.q_sample(x_0, torch.tensor([999]))
    print(f"x_0 均值: {x_0.mean():.4f}, 标准差: {x_0.std():.4f}")
    print(f"x_T 均值: {x_t.mean():.4f}, 标准差: {x_t.std():.4f}")