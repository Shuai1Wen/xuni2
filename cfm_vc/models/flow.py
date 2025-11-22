"""
FlowField：Flow Matching的向量场，具有数值稳定性和梯度安全特性

关键设计点：
1. 共享trunk确保参数高效
2. base+basis分解明确因果结构
3. 时间嵌入的数值范围约束
4. 梯度流清晰：无in-place操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FlowField(nn.Module):
    """
    条件Flow Matching的向量场
    
    向量场结构：
    v_θ(z_t, t | p, c, s) = v_base(h) + Σ_j α_j(p) * b_j(h)
    
    其中：
    - h = trunk(concat[z_t, γ(t), enc_c(c, s)])：共享特征表示
    - v_base：未扰动的基线向量场
    - b_j：basis向量场集合（参数化的向量）
    - α_j：扰动导致的调制系数
    
    参数
    ----
    dim_int : int
        z_int的维度（向量场的输出维度）
    
    context_dim : int
        context向量的维度（来自ContextEncoder）
    
    alpha_dim : int
        pert_alpha向量的维度（来自ContextEncoder）
    
    hidden_dim : int，default=128
        trunk和basis MLP的隐层维度
    
    n_basis : int，default=16
        basis向量场的数量
    
    time_embed_dim : int，default=16
        时间嵌入的维度
    """
    
    def __init__(
        self,
        dim_int: int,
        context_dim: int,
        alpha_dim: int,
        hidden_dim: int = 128,
        n_basis: int = 16,
        time_embed_dim: int = 16,
    ):
        super().__init__()
        self.dim_int = dim_int
        self.context_dim = context_dim
        self.alpha_dim = alpha_dim
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis
        self.time_embed_dim = time_embed_dim
        
        # ============ 时间嵌入MLP ============
        # t ∈ [0,1] → time_embed_dim维向量
        # 使用SiLU激活，避免梯度消失
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # ============ 共享trunk MLP ============
        # 输入：z_t + time_embed + context
        trunk_input_dim = dim_int + time_embed_dim + context_dim
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # ============ Base向量场 ============
        # W_base @ h → dim_int
        # 这表示未扰动情况下的向量场
        self.base_head = nn.Linear(hidden_dim, dim_int)
        
        # ============ Basis向量场 ============
        # W_B @ h → n_basis * dim_int
        # 之后reshape为(n_basis, dim_int)
        # 这些基向量可以被扰动系数线性组合
        self.basis_head = nn.Linear(hidden_dim, n_basis * dim_int)
        
        # ============ Adapter：扰动到basis系数的映射 ============
        # α_dim → n_basis
        # 不加bias，与ContextEncoder的adapter对应
        self.alpha_head = nn.Linear(alpha_dim, n_basis, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier/Kaiming初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        pert_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算向量场v_θ(z_t, t | p, c, s)
        
        参数
        ----
        z_t : torch.Tensor
            时刻t的潜变量 shape (B, dim_int)
        
        t : torch.Tensor
            时间步长 shape (B,)，值在[0, 1]
            
            ⚠️ 梯度警告：如果t来自torch.rand()，它不可微分
            如果需要微分t，请显式开启requires_grad
        
        context : torch.Tensor
            上下文向量（协变量+空间） shape (B, context_dim)
        
        pert_alpha : torch.Tensor
            扰动adapter系数 shape (B, alpha_dim)
        
        返回
        ----
        v : torch.Tensor
            向量场 shape (B, dim_int)
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        
        # ============ 时间嵌入 ============
        # t.unsqueeze(-1)将(B,)变为(B, 1)
        t_embed = self.time_mlp(t.unsqueeze(-1))  # (B, time_embed_dim)
        
        # ============ 拼接所有输入到trunk ============
        h_in = torch.cat([z_t, t_embed, context], dim=-1)  # (B, trunk_input_dim)
        
        # ============ 通过共享trunk ============
        h = self.trunk(h_in)  # (B, hidden_dim)
        
        # ============ Base向量场 ============
        v_base = self.base_head(h)  # (B, dim_int)
        
        # ============ Basis向量场 ============
        basis_out = self.basis_head(h)  # (B, n_basis * dim_int)
        basis = basis_out.view(batch_size, self.n_basis, self.dim_int)  # (B, n_basis, dim_int)
        
        # ============ Adapter系数 ============
        coeff = self.alpha_head(pert_alpha)  # (B, n_basis)
        coeff = coeff.unsqueeze(-1)  # (B, n_basis, 1) 用于广播
        
        # ============ Basis的加权组合 ============
        # coeff * basis: (B, n_basis, 1) * (B, n_basis, dim_int) → (B, n_basis, dim_int)
        # sum(dim=1): (B, n_basis, dim_int) → (B, dim_int)
        v_eff = torch.sum(coeff * basis, dim=1)  # (B, dim_int)
        
        # ============ 最终向量场 ============
        v = v_base + v_eff  # (B, dim_int)
        
        return v
