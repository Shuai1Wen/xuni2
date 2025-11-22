"""
Stage 1：VAE预训练

只训练Encoder/Decoder，冻结Flow和Context模块。
重点关注梯度流的正确性和数值稳定性。
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import logging


logger = logging.getLogger(__name__)


def train_vae_stage(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    beta: float = 1.0,
    grad_clip_max_norm: float = 1.0,
    nan_check_interval: int = 10,  # 每10个batch检查一次NaN
) -> Dict[str, list]:
    """
    VAE预训练阶段
    
    梯度流设计：
    1. 只优化Encoder和Decoder
    2. 冻结Flow和ContextEncoder
    3. 每个batch检查梯度是否正常
    
    数值稳定性设计：
    1. 定期检查Loss是否为NaN
    2. 梯度裁剪防止爆炸
    3. 详细的日志输出
    
    参数
    ----
    model : CFMVCModel
        CFM-VC模型
    
    train_loader : DataLoader
        训练数据加载器
        batch应该有键：x, p, batch, cell_type, spatial (optional)
    
    val_loader : DataLoader，optional
        验证数据加载器
    
    n_epochs : int，default=50
        训练轮数
    
    learning_rate : float，default=1e-3
        学习率（AdamW）
    
    weight_decay : float，default=1e-5
        L2正则化权重
    
    device : str，default="cpu"
        计算设备
    
    beta : float，default=1.0
        KL损失的权重（β-VAE）
    
    grad_clip_max_norm : float，default=1.0
        梯度裁剪的最大范数
    
    nan_check_interval : int，default=10
        每n个batch检查一次NaN
    
    返回
    ----
    history : dict
        训练历史，包含键：
        - "train_loss"：训练损失列表
        - "val_loss"：验证损失列表（如果提供val_loader）
        - "nan_step"：首次出现NaN的step（如果有）
    """
    
    # ============ 冻结Flow和Context模块，只训练Encoder/Decoder ============
    model.flow.eval()
    model.context_encoder.eval()
    for param in model.flow.parameters():
        param.requires_grad = False
    for param in model.context_encoder.parameters():
        param.requires_grad = False
    
    logger.info("✓ Flow和ContextEncoder已冻结，仅训练Encoder/Decoder")
    
    # Encoder和Decoder的参数
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    
    # ============ 统计可优化参数数量 ============
    n_trainable = sum(p.numel() for p in vae_params if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"可优化参数: {n_trainable:,} / {n_total:,}")
    
    optimizer = AdamW(vae_params, lr=learning_rate, weight_decay=weight_decay)
    
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "nan_step": None,
    }
    
    # ============ 训练循环 ============
    global_step = 0
    
    for epoch in range(n_epochs):
        # ===== 训练阶段 =====
        model.train()
        total_loss = 0.0
        num_samples = 0
        epoch_nan_occurred = False
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            x = batch["x"].to(device)  # (B, n_genes)
            batch_idx_tensor = batch["batch"].to(device)  # (B,)
            ct_idx = batch["cell_type"].to(device)  # (B,)
            
            # VAE前向传播
            loss_vae, z_int, z_tech = model.vae_forward(
                x, batch_idx_tensor, ct_idx, beta=beta
            )
            
            # ============ 梯度反向传播 ============
            optimizer.zero_grad()
            loss_vae.backward()
            
            # ============ 梯度裁剪（防止梯度爆炸）============
            torch.nn.utils.clip_grad_norm_(vae_params, max_norm=grad_clip_max_norm)
            
            # ============ NaN检查（定期进行）============
            if global_step % nan_check_interval == 0:
                if not torch.isfinite(loss_vae):
                    logger.error(f"❌ NaN in loss at step {global_step}")
                    history["nan_step"] = global_step
                    epoch_nan_occurred = True
                    break
                
                # 检查梯度
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                        logger.warning(f"⚠️ NaN in gradient of {name} at step {global_step}")
            
            optimizer.step()
            
            # ============ 日志更新 ============
            total_loss += loss_vae.item() * x.shape[0]
            num_samples += x.shape[0]
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss_vae.item():.4f}",
                "step": global_step,
            })
        
        if epoch_nan_occurred:
            logger.error("训练中止：检测到NaN")
            break
        
        avg_loss = total_loss / num_samples
        history["train_loss"].append(avg_loss)
        
        # ===== 验证阶段 =====
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                    x = batch["x"].to(device)
                    batch_idx_tensor = batch["batch"].to(device)
                    ct_idx = batch["cell_type"].to(device)
                    
                    loss_vae, _, _ = model.vae_forward(
                        x, batch_idx_tensor, ct_idx, beta=beta
                    )
                    val_loss += loss_vae.item() * x.shape[0]
                    val_samples += x.shape[0]
            
            avg_val_loss = val_loss / val_samples
            history["val_loss"].append(avg_val_loss)
            
            logger.info(
                f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")
    
    # ============ 解冻所有参数（为Stage 2做准备）============
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("✓ Stage 1完成，所有参数已解冻")
    
    return history
