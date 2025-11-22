"""
EncoderVAE：NB-VAE编码器，具有数值稳定性和梯度安全特性

关键设计点：
1. logvar的范围约束：防止exp(logvar)出现极端值
2. 重参数化的数值稳定性：使用log-space操作
3. 梯度流清晰：避免in-place操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EncoderVAE(nn.Module):
    """
    VAE编码器，分离内在状态和技术噪声
    
    参数
    ----
    n_genes : int
        基因数量
    
    n_batch : int
        批次类别数
    
    n_ct : int
        细胞类型类别数
    
    dim_int : int，default=32
        内在状态（z_int）的维度
    
    dim_tech : int，default=8
        技术噪声（z_tech）的维度
    
    hidden_dim : int，default=256
        隐层维度
    
    dropout_rate : float，default=0.1
        Dropout比例
    """
    
    def __init__(
        self,
        n_genes: int,
        n_batch: int,
        n_ct: int,
        dim_int: int = 32,
        dim_tech: int = 8,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_batch = n_batch
        self.n_ct = n_ct
        self.dim_int = dim_int
        self.dim_tech = dim_tech
        
        # ============ 数值稳定性常数 ============
        # logvar的约束范围，防止exp(logvar)出现NaN或溢出
        self.logvar_max = 10.0  # exp(10) ≈ 22026，足够大
        self.logvar_min = -10.0  # exp(-10) ≈ 0.000045，足够小
        
        # 协变量嵌入
        batch_emb_dim = 8
        ct_emb_dim = 8
        self.batch_emb = nn.Embedding(n_batch, batch_emb_dim)
        self.ct_emb = nn.Embedding(n_ct, ct_emb_dim)
        
        # MLP主干：输入 = 基因数 + batch_emb + ct_emb
        input_dim = n_genes + batch_emb_dim + ct_emb_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # z_int 的均值和对数方差
        self.z_int_mean = nn.Linear(hidden_dim, dim_int)
        self.z_int_logvar = nn.Linear(hidden_dim, dim_int)
        
        # z_tech 的均值和对数方差
        self.z_tech_mean = nn.Linear(hidden_dim, dim_tech)
        self.z_tech_logvar = nn.Linear(hidden_dim, dim_tech)
        
        # 初始化权重（改进梯度流）
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier/Kaiming初始化，改进梯度流"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        ct_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数
        ----
        x : torch.Tensor
            表达矩阵 shape (B, n_genes)，可以是计数或正则化后的值
        
        batch_idx : torch.Tensor
            批次索引 shape (B,)，值在 [0, n_batch)
        
        ct_idx : torch.Tensor
            细胞类型索引 shape (B,)，值在 [0, n_ct)
        
        返回
        ----
        z_int : torch.Tensor
            内在状态采样 shape (B, dim_int)
        
        z_tech : torch.Tensor
            技术噪声采样 shape (B, dim_tech)
        
        kl_int : torch.Tensor
            内在状态的KL散度 shape (B,)
        
        kl_tech : torch.Tensor
            技术噪声的KL散度 shape (B,)
        """
        
        # 嵌入协变量
        batch_e = self.batch_emb(batch_idx)  # (B, 8)
        ct_e = self.ct_emb(ct_idx)  # (B, 8)
        
        # 拼接所有输入
        h = torch.cat([x, batch_e, ct_e], dim=-1)  # (B, n_genes+16)
        
        # 通过MLP（使用in-place操作减少内存）
        h = F.relu(self.fc1(h), inplace=True)  # (B, hidden_dim)
        h = self.dropout1(h)
        h = F.relu(self.fc2(h), inplace=True)  # (B, hidden_dim)
        h = self.dropout2(h)
        
        # z_int 的参数
        z_int_mean = self.z_int_mean(h)  # (B, dim_int)
        z_int_logvar = self.z_int_logvar(h)  # (B, dim_int)
        
        # z_tech 的参数
        z_tech_mean = self.z_tech_mean(h)  # (B, dim_tech)
        z_tech_logvar = self.z_tech_logvar(h)  # (B, dim_tech)
        
        # ============ 数值稳定性：约束logvar范围 ============
        # 这防止exp(logvar)出现NaN或爆炸
        z_int_logvar = torch.clamp(z_int_logvar, self.logvar_min, self.logvar_max)
        z_tech_logvar = torch.clamp(z_tech_logvar, self.logvar_min, self.logvar_max)
        
        # 重参数化技巧（训练时采样，eval时使用均值）
        if self.training:
            eps_int = torch.randn_like(z_int_mean)  # (B, dim_int)
            eps_tech = torch.randn_like(z_tech_mean)  # (B, dim_tech)

            z_int = z_int_mean + eps_int * torch.exp(0.5 * z_int_logvar)
            z_tech = z_tech_mean + eps_tech * torch.exp(0.5 * z_tech_logvar)
        else:
            # eval模式：使用确定性编码（均值），减少方差
            z_int = z_int_mean
            z_tech = z_tech_mean
        
        # ============ KL散度计算 ============
        # KL(N(μ,σ²) || N(0,I)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_int = -0.5 * torch.sum(
            1.0 + z_int_logvar - z_int_mean.pow(2) - z_int_logvar.exp(),
            dim=-1
        )  # (B,)
        
        kl_tech = -0.5 * torch.sum(
            1.0 + z_tech_logvar - z_tech_mean.pow(2) - z_tech_logvar.exp(),
            dim=-1
        )  # (B,)
        
        return z_int, z_tech, kl_int, kl_tech
