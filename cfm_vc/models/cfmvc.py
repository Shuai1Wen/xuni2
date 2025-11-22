"""
CFMVCModel：完整的CFM-VC模型封装

关键梯度流设计：
1. Stage 1 (VAE)：encoder和decoder都需要梯度
2. Stage 2 (Flow)：z_int.detach()防止梯度反传到VAE
3. 可选联合微调：显式控制哪些部分被冻结

数值稳定性设计：
1. NB似然已在decoder中处理
2. Flow的velocity matching在合理范围内
3. 采样时的ODE积分使用Euler方法（简单稳定）
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .encoder import EncoderVAE
from .decoder import DecoderVAE, nb_log_likelihood
from .context import ContextEncoder
from .flow import FlowField


class CFMVCModel(nn.Module):
    """
    CFM-VC 2.x 完整模型
    
    参数
    ----
    n_genes : int
        基因数量
    
    n_batch : int
        批次类别数
    
    n_ct : int
        细胞类型类别数
    
    n_perts : int
        扰动类别数（one-hot向量长度）
    
    dim_int : int，default=32
        内在状态维度
    
    dim_tech : int，default=8
        技术噪声维度
    
    hidden_vae : int，default=256
        VAE编码/解码MLP的隐层维度
    
    hidden_ctx : int，default=64
        ContextEncoder MLP的隐层维度
    
    hidden_flow : int，default=128
        FlowField的隐层维度
    
    n_basis : int，default=16
        Flow Matching中basis向量场的数量
    
    spatial_dim : int，optional
        空间坐标维度。若为None则不处理空间信息
    """
    
    def __init__(
        self,
        n_genes: int,
        n_batch: int,
        n_ct: int,
        n_perts: int,
        dim_int: int = 32,
        dim_tech: int = 8,
        hidden_vae: int = 256,
        hidden_ctx: int = 64,
        hidden_flow: int = 128,
        n_basis: int = 16,
        spatial_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_batch = n_batch
        self.n_ct = n_ct
        self.n_perts = n_perts
        self.dim_int = dim_int
        self.dim_tech = dim_tech
        self.spatial_dim = spatial_dim
        
        # ============ 模块A：表示层（VAE）============
        self.encoder = EncoderVAE(
            n_genes=n_genes,
            n_batch=n_batch,
            n_ct=n_ct,
            dim_int=dim_int,
            dim_tech=dim_tech,
            hidden_dim=hidden_vae,
        )
        
        self.decoder = DecoderVAE(
            n_genes=n_genes,
            dim_int=dim_int,
            dim_tech=dim_tech,
            hidden_dim=hidden_vae,
        )
        
        # ============ Context编码器 ============
        self.context_encoder = ContextEncoder(
            n_batch=n_batch,
            n_ct=n_ct,
            p_dim=n_perts,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_ctx,
        )
        
        # 获取context和pert_alpha的维度
        context_dim = self.context_encoder.context_dim
        alpha_dim = self.context_encoder.pert_alpha_dim
        
        # ============ 模块B：向量场（Flow Matching）============
        self.flow = FlowField(
            dim_int=dim_int,
            context_dim=context_dim,
            alpha_dim=alpha_dim,
            hidden_dim=hidden_flow,
            n_basis=n_basis,
        )
    
    # ========== VAE部分：模块A ==========
    
    def vae_forward(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        ct_idx: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE前向传播（Stage 1训练用）
        
        参数
        ----
        x : torch.Tensor
            计数矩阵 shape (B, n_genes)
        
        batch_idx : torch.Tensor
            批次索引 shape (B,)
        
        ct_idx : torch.Tensor
            细胞类型索引 shape (B,)
        
        beta : float，default=1.0
            KL项的权重（β-VAE变体）
            - beta=1.0：标准VAE
            - beta<1.0：降低KL权重，得到更好的重建
        
        返回
        ----
        loss_vae : torch.Tensor
            VAE损失（标量，用于反向传播）
        
        z_int : torch.Tensor
            内在状态 shape (B, dim_int)（用于后续的Flow训练）
        
        z_tech : torch.Tensor
            技术噪声 shape (B, dim_tech)
        """
        
        # 编码
        z_int, z_tech, kl_int, kl_tech = self.encoder(x, batch_idx, ct_idx)
        
        # 解码
        mean, theta = self.decoder(z_int, z_tech)
        
        # 重建损失（负对数似然）
        nb_ll = nb_log_likelihood(x, mean, theta)  # (B,)
        recon_loss = -torch.mean(nb_ll)
        
        # KL损失（使用beta加权）
        kl_loss = torch.mean(kl_int + kl_tech) * beta
        
        # ============ 数值稳定性检查 ============
        # 检查是否出现NaN或Inf，这通常表示数值问题
        if not torch.isfinite(recon_loss):
            print("警告：重建损失不是有限值")
            print(f"  nb_ll: min={nb_ll.min()}, max={nb_ll.max()}")
            # 用一个合理的替代值
            recon_loss = torch.tensor(1e3, dtype=recon_loss.dtype, device=recon_loss.device)
        
        if not torch.isfinite(kl_loss):
            print("警告：KL损失不是有限值")
            print(f"  kl_int: min={kl_int.min()}, max={kl_int.max()}")
            print(f"  kl_tech: min={kl_tech.min()}, max={kl_tech.max()}")
            kl_loss = torch.tensor(1e3, dtype=kl_loss.dtype, device=kl_loss.device)
        
        # VAE总损失
        loss_vae = recon_loss + kl_loss
        
        return loss_vae, z_int, z_tech
    
    # ========== Flow Matching部分：模块B ==========
    
    def flow_step(
        self,
        z_int: torch.Tensor,
        p: torch.Tensor,
        batch_idx: torch.Tensor,
        ct_idx: torch.Tensor,
        spatial: Optional[torch.Tensor] = None,
        lambda_dist: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flow Matching训练步（Stage 2训练用）
        
        使用OT直线插值路径：
        - z_0 ~ N(0, I)
        - z_t = (1-t) * z_0 + t * z_1
        - u_t = z_1 - z_0（目标速度）
        
        参数
        ----
        z_int : torch.Tensor
            VAE编码的内在状态（作为z_1） shape (B, dim_int)
            ⚠️ 关键：在Stage 2中应该传入已detach的z_int，防止梯度反传到VAE
        
        p : torch.Tensor
            one-hot扰动向量 shape (B, n_perts)
        
        batch_idx : torch.Tensor
            批次索引 shape (B,)
        
        ct_idx : torch.Tensor
            细胞类型索引 shape (B,)
        
        spatial : torch.Tensor，optional
            空间坐标 shape (B, spatial_dim)
        
        lambda_dist : float，default=0.0
            分布匹配损失的权重。为0时不使用分布损失
        
        返回
        ----
        total_loss : torch.Tensor
            总损失（标量）
        
        fm_loss : torch.Tensor
            Flow Matching损失（标量）
        
        dist_loss : torch.Tensor
            分布匹配损失（标量）
        """
        
        batch_size, dim = z_int.shape
        device = z_int.device
        
        # ============ OT路径构建 ============
        # z_1 = VAE编码的z_int
        # ⚠️ 梯度关键：如果要冻结VAE，z_int应该已经detach
        z1 = z_int  # 假设已在外部detach
        
        # z_0 ~ N(0, I)
        z0 = torch.randn_like(z1)
        
        # 均匀采样时间
        t = torch.rand(batch_size, device=device)  # (B,)
        
        # OT直线插值
        z_t = (1.0 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z1  # (B, dim_int)
        
        # 目标速度
        u_t = z1 - z0  # (B, dim_int)
        
        # ============ 向量场预测 ============
        # 获取context和pert_alpha
        context, pert_alpha = self.context_encoder(p, batch_idx, ct_idx, spatial)
        
        # 向量场预测
        v_pred = self.flow(z_t, t, context, pert_alpha)  # (B, dim_int)
        
        # ============ Flow Matching损失 ============
        fm_loss = torch.mean((v_pred - u_t) ** 2)
        
        # 检查是否出现NaN
        if not torch.isfinite(fm_loss):
            print("警告：FM损失不是有限值")
            print(f"  v_pred range: [{v_pred.min()}, {v_pred.max()}]")
            print(f"  u_t range: [{u_t.min()}, {u_t.max()}]")
            fm_loss = torch.tensor(1e3, dtype=fm_loss.dtype, device=fm_loss.device)
        
        # ============ 分布匹配损失（可选）============
        dist_loss = torch.tensor(0.0, device=device)
        if lambda_dist > 0:
            # 简化版本：用z1的统计信息
            # 完整版本应该从flow采样z1_hat，计算MMD或ED
            z1_mean = torch.mean(z1, dim=0, keepdim=True)
            z1_std = torch.std(z1, dim=0, keepdim=True) + 1e-8
            
            # 标准化z1
            z1_norm = (z1 - z1_mean) / z1_std
            
            # 与N(0,I)的距离（简化的MMD）
            dist_loss = torch.mean(z1_norm ** 2) - 1.0
            dist_loss = torch.clamp(dist_loss, min=0.0)  # 避免负值
        
        # ============ 总损失 ============
        total_loss = fm_loss + lambda_dist * dist_loss
        
        return total_loss, fm_loss, dist_loss
    
    # ========== 采样与生成 ==========
    
    @torch.no_grad()
    def sample_z1_from_flow(
        self,
        p: torch.Tensor,
        batch_idx: torch.Tensor,
        ct_idx: torch.Tensor,
        spatial: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        从Flow积分z_0到z_1
        
        使用Euler方法ODE积分：
        dz/dt = v_θ(z, t | p, c, s)
        
        参数
        ----
        p : torch.Tensor
            one-hot扰动向量 shape (B, n_perts)
        
        batch_idx : torch.Tensor
            批次索引 shape (B,)
        
        ct_idx : torch.Tensor
            细胞类型索引 shape (B,)
        
        spatial : torch.Tensor，optional
            空间坐标 shape (B, spatial_dim)
        
        n_steps : int，default=20
            积分步数（增加n_steps可以提高精度，但会变慢）
        
        返回
        ----
        z1 : torch.Tensor
            采样的z_1 shape (B, dim_int)
        """
        
        batch_size = p.shape[0]
        device = p.device
        
        # z_0 ~ N(0, I)
        z = torch.randn(batch_size, self.dim_int, device=device)
        
        # 时间从0到1
        t = torch.zeros(batch_size, device=device)
        dt = 1.0 / n_steps
        
        # ============ Euler积分 ============
        for step_idx in range(n_steps):
            # 获取context和pert_alpha
            context, pert_alpha = self.context_encoder(p, batch_idx, ct_idx, spatial)
            
            # 向量场
            v = self.flow(z, t, context, pert_alpha)
            
            # 检查是否出现NaN（早期检测）
            if not torch.all(torch.isfinite(v)):
                print(f"警告：在ODE积分第{step_idx}步出现NaN")
                print(f"  v的范围: [{v.min()}, {v.max()}]")
                # 用0替换NaN
                v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
            
            # 更新
            z = z + dt * v
            t = t + dt
        
        return z
    
    @torch.no_grad()
    def generate_expression(
        self,
        p: torch.Tensor,
        batch_idx: torch.Tensor,
        ct_idx: torch.Tensor,
        spatial: Optional[torch.Tensor] = None,
        n_steps: int = 20,
        use_mean: bool = True,
    ) -> torch.Tensor:
        """
        从Flow生成表达谱
        
        参数
        ----
        p : torch.Tensor
            one-hot扰动向量 shape (B, n_perts)
        
        batch_idx : torch.Tensor
            批次索引 shape (B,)
        
        ct_idx : torch.Tensor
            细胞类型索引 shape (B,)
        
        spatial : torch.Tensor，optional
            空间坐标 shape (B, spatial_dim)
        
        n_steps : int，default=20
            ODE积分步数
        
        use_mean : bool，default=True
            若为True，返回NB分布的均值
            若为False，从NB分布采样（当前未实现，仍返回均值）
        
        返回
        ----
        X_hat : torch.Tensor
            生成的表达矩阵 shape (B, n_genes)，全为正值
        """
        
        # 从flow采样z_1
        z_int = self.sample_z1_from_flow(p, batch_idx, ct_idx, spatial, n_steps)
        
        # z_tech用0初始化（或可以从先验采样）
        z_tech = torch.zeros(z_int.shape[0], self.dim_tech, device=z_int.device)
        
        # 解码得到NB参数
        mean, theta = self.decoder(z_int, z_tech)
        
        # 检查是否全正
        if not torch.all(mean > 0):
            print("警告：生成的表达均值包含非正值")
            mean = torch.clamp(mean, min=1e-8)
        
        if use_mean:
            return mean
        else:
            # TODO: 从NB分布真实采样
            return mean
