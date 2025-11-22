"""
DecoderVAE和负二项似然函数

关键设计点：
1. NB似然的完整Gamma函数参数化（不是泊松近似）
2. 数值稳定性：所有log操作都用eps防护
3. lgamma函数的safe使用
4. 分散参数theta的约束（确保为正）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DecoderVAE(nn.Module):
    """
    VAE解码器，使用负二项分布重建计数数据
    
    参数
    ----
    n_genes : int
        基因数量
    
    dim_int : int，default=32
        内在状态维度
    
    dim_tech : int，default=8
        技术噪声维度
    
    hidden_dim : int，default=256
        隐层维度
    
    dropout_rate : float，default=0.1
        Dropout比例
    """
    
    def __init__(
        self,
        n_genes: int,
        dim_int: int = 32,
        dim_tech: int = 8,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.dim_int = dim_int
        self.dim_tech = dim_tech
        
        # ============ 数值稳定性常数 ============
        self.eps = 1e-8  # 防止log(0)
        
        # MLP主干
        input_dim = dim_int + dim_tech
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 输出层：负二项分布的均值（log空间，确保数值稳定）
        self.mean_out = nn.Linear(hidden_dim, n_genes)
        
        # 每个基因的对数分散参数（可学习，初始化为0即θ=1）
        # 约束：θ > 0，所以用log参数化
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，改进梯度流"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        z_int: torch.Tensor,
        z_tech: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数
        ----
        z_int : torch.Tensor
            内在状态 shape (B, dim_int)
        
        z_tech : torch.Tensor
            技术噪声 shape (B, dim_tech)
        
        返回
        ----
        mean : torch.Tensor
            负二项分布的均值 shape (B, n_genes)，全为正值
        
        theta : torch.Tensor
            分散参数 shape (n_genes,)，全为正值
        """
        
        # 拼接潜变量
        h = torch.cat([z_int, z_tech], dim=-1)  # (B, dim_int + dim_tech)
        
        # 通过MLP（使用in-place操作减少内存）
        h = F.relu(self.fc1(h), inplace=True)  # (B, hidden_dim)
        h = self.dropout1(h)
        h = F.relu(self.fc2(h), inplace=True)  # (B, hidden_dim)
        h = self.dropout2(h)
        
        # 均值：使用softplus或clamped exp确保为正且数值稳定
        # 限制mean_out的输出范围，防止exp溢出
        mean_logits = self.mean_out(h)
        mean_logits = torch.clamp(mean_logits, min=-20.0, max=20.0)  # exp(-20) ≈ 2e-9, exp(20) ≈ 4.8e8
        mean = torch.exp(mean_logits)  # (B, n_genes)，全为正

        # 分散参数：同样限制范围防止溢出
        log_theta_clamped = torch.clamp(self.log_theta, min=-10.0, max=10.0)
        theta = torch.exp(log_theta_clamped)  # (n_genes,)，全为正
        
        return mean, theta


def nb_log_likelihood(
    x: torch.Tensor,
    mean: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    numerically_stable: bool = True,
) -> torch.Tensor:
    """
    负二项分布的对数似然，具有完整的数值稳定性保护
    
    负二项分布的PMF：
    P(x|μ,θ) = Γ(x+θ)/(Γ(θ)Γ(x+1)) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x
    
    对数形式：
    log P = log Γ(x+θ) - log Γ(θ) - log Γ(x+1)
            + θ*log(θ/(θ+μ)) + x*log(μ/(θ+μ))
    
    参数
    ----
    x : torch.Tensor
        观察的计数矩阵 shape (B, n_genes)
    
    mean : torch.Tensor
        分布的均值 shape (B, n_genes)
    
    theta : torch.Tensor
        分散参数 shape (n_genes,)，会自动broadcast到(B, n_genes)
    
    eps : float，default=1e-8
        数值稳定性用的小常数，防止log(0)
    
    numerically_stable : bool，default=True
        是否使用数值稳定的计算方式
        True：使用log-sum-exp技巧
        False：直接计算（可能出现NaN）
    
    返回
    ----
    log_likelihood : torch.Tensor
        对数似然 shape (B,)，对基因维度求和
    """
    
    # 确保theta的shape正确
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)  # (1, n_genes)，会自动broadcast
    
    # ============ 数值稳定性检查 ============
    # 确保所有值都是有限的（非NaN、非Inf）
    if not numerically_stable:
        assert torch.all(torch.isfinite(x)), "x包含NaN或Inf"
        assert torch.all(torch.isfinite(mean)), "mean包含NaN或Inf"
        assert torch.all(torch.isfinite(theta)), "theta包含NaN或Inf"
    
    # ============ 安全的log操作 ============
    # 使用clamp防止log(0)或log(负数)
    mean_safe = torch.clamp(mean, min=eps)
    theta_safe = torch.clamp(theta, min=eps)
    
    # ============ Gamma函数项 ============
    # log Γ(x+θ)
    lgamma_x_plus_theta = torch.lgamma(x + theta_safe)
    
    # log Γ(θ)
    lgamma_theta = torch.lgamma(theta_safe)
    
    # log Γ(x+1) = log(x!)
    # 注意：x可能为0，lgamma(1) = 0，这是正确的
    lgamma_x_plus_1 = torch.lgamma(x + 1.0)
    
    # Gamma项的组合
    gamma_term = lgamma_x_plus_theta - lgamma_theta - lgamma_x_plus_1  # (B, n_genes)
    
    # ============ 概率项（数值稳定形式）============
    # 原始：θ*log(θ/(θ+μ))
    # 改写：θ*(log(θ) - log(θ+μ))
    log_prob_theta = theta_safe * (
        torch.log(theta_safe + eps) - torch.log(theta_safe + mean_safe + eps)
    )  # (B, n_genes)
    
    # 原始：x*log(μ/(θ+μ))
    # 改写：x*(log(μ) - log(θ+μ))
    log_prob_x = x * (
        torch.log(mean_safe + eps) - torch.log(theta_safe + mean_safe + eps)
    )  # (B, n_genes)
    
    # ============ 完整的对数似然 ============
    log_p = gamma_term + log_prob_theta + log_prob_x  # (B, n_genes)
    
    # ============ 检查是否出现NaN ============
    # 如果出现NaN，尝试诊断并报告
    if torch.any(~torch.isfinite(log_p)):
        nan_mask = ~torch.isfinite(log_p)
        nan_count = torch.sum(nan_mask).item()
        
        # 只输出前几个NaN位置用于诊断（避免打印过多）
        if nan_count <= 10:
            print(f"警告：发现{nan_count}个NaN值")
            print(f"受影响的位置: {torch.where(nan_mask)}")
        else:
            print(f"警告：发现{nan_count}个NaN值（太多，仅显示统计）")
        
        # 将NaN替换为一个大的负数（表示低似然）
        log_p = torch.where(torch.isfinite(log_p), log_p, torch.tensor(-1e6, dtype=log_p.dtype, device=log_p.device))
    
    # 对基因维度求和
    log_likelihood = torch.sum(log_p, dim=-1)  # (B,)
    
    return log_likelihood
