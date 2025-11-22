"""
Stage 2：Flow Matching训练

关键梯度流设计：
1. z_int.detach()：防止梯度反传到VAE
2. 冻结或不冻结VAE参数
3. 只优化Flow和ContextEncoder

数值稳定性：
1. Velocity matching的损失在合理范围
2. 及时检查NaN
3. ODE积分的稳定性监测
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


def train_flow_stage(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_dist: float = 0.0,
    freeze_vae: bool = True,
    device: str = "cpu",
    grad_clip_max_norm: float = 1.0,
    nan_check_interval: int = 10,
) -> Dict[str, list]:
    """
    Flow Matching训练阶段
    
    梯度流设计：
    1. z_int.detach()确保梯度不反传到VAE
    2. 可选冻结VAE参数
    3. 优化Flow和ContextEncoder
    
    数值稳定性：
    1. Velocity matching的target (u_t)通常在-5到5范围
    2. 向量场输出也应该在类似范围
    3. 检查是否出现远离期望范围的值
    
    参数
    ----
    model : CFMVCModel
        CFM-VC模型（应该已经经过Stage 1预训练）
    
    train_loader : DataLoader
        训练数据加载器
    
    val_loader : DataLoader，optional
        验证数据加载器
    
    n_epochs : int，default=50
        训练轮数
    
    learning_rate : float，default=1e-3
        学习率
    
    weight_decay : float，default=1e-5
        L2正则化权重
    
    lambda_dist : float，default=0.0
        分布匹配损失的权重。为0时不使用
    
    freeze_vae : bool，default=True
        是否冻结VAE参数。
        True：只训练Flow/Context（标准Stage 2）
        False：联合微调所有参数（可选的Stage 3）
    
    device : str，default="cpu"
        计算设备
    
    grad_clip_max_norm : float，default=1.0
        梯度裁剪的最大范数
    
    nan_check_interval : int，default=10
        每n个batch检查一次NaN
    
    返回
    ----
    history : dict
        训练历史
    """
    
    # ============ 梯度流设置 ============
    if freeze_vae:
        # 冻结VAE参数
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        logger.info("✓ VAE参数已冻结（Stage 2标准模式）")
    else:
        logger.info("⚠️ VAE参数未冻结（联合微调模式）")
    
    # Flow和Context的参数
    trainable_params = (
        list(model.flow.parameters()) + 
        list(model.context_encoder.parameters())
    )
    if not freeze_vae:
        trainable_params += (
            list(model.encoder.parameters()) + 
            list(model.decoder.parameters())
        )
    
    # ============ 统计可优化参数数量 ============
    n_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"可优化参数: {n_trainable:,} / {n_total:,}")
    
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    model.to(device)
    history = {
        "train_loss": [],
        "train_fm_loss": [],
        "train_dist_loss": [],
        "val_loss": [],
        "val_fm_loss": [],
        "val_dist_loss": [],
        "nan_step": None,
    }
    
    # ============ 训练循环 ============
    global_step = 0
    
    for epoch in range(n_epochs):
        # ===== 训练阶段 =====
        model.train()
        total_loss = 0.0
        total_fm_loss = 0.0
        total_dist_loss = 0.0
        num_samples = 0
        epoch_nan_occurred = False
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            x = batch["x"].to(device)
            p = batch["p"].to(device)
            batch_idx_tensor = batch["batch"].to(device)
            ct_idx = batch["cell_type"].to(device)
            spatial = batch["spatial"]
            if spatial is not None:
                spatial = spatial.to(device)
            
            # ============ 编码x得到z_int ============
            # 关键：这里计算z_int但立即detach，防止Flow的梯度反传到VAE
            with torch.no_grad():
                z_int, _, _, _ = model.encoder(x, batch_idx_tensor, ct_idx)
            
            # z_int现在是一个无梯度的张量，Flow训练不会影响VAE
            z_int_detached = z_int.detach()
            
            # ============ Flow训练步 ============
            loss, fm_loss, dist_loss = model.flow_step(
                z_int_detached, p, batch_idx_tensor, ct_idx,
                spatial=spatial,
                lambda_dist=lambda_dist,
            )
            
            # ============ 反向传播 ============
            optimizer.zero_grad()
            loss.backward()
            
            # ============ 梯度裁剪 ============
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_max_norm)
            
            # ============ NaN检查 ============
            if global_step % nan_check_interval == 0:
                if not torch.isfinite(loss):
                    logger.error(f"❌ NaN in loss at step {global_step}")
                    logger.error(f"  FM loss: {fm_loss.item()}")
                    logger.error(f"  Dist loss: {dist_loss.item()}")
                    history["nan_step"] = global_step
                    epoch_nan_occurred = True
                    break
                
                # 检查梯度
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                        logger.warning(f"⚠️ NaN in gradient of {name}")
                        has_nan_grad = True
                
                if has_nan_grad:
                    epoch_nan_occurred = True
                    break
            
            optimizer.step()
            
            # ============ 日志更新 ============
            total_loss += loss.item() * x.shape[0]
            total_fm_loss += fm_loss.item() * x.shape[0]
            total_dist_loss += dist_loss.item() * x.shape[0]
            num_samples += x.shape[0]
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "fm": f"{fm_loss.item():.4f}",
                "dist": f"{dist_loss.item():.4f}",
            })
        
        if epoch_nan_occurred:
            logger.error("训练中止：检测到NaN")
            break
        
        avg_loss = total_loss / num_samples
        avg_fm_loss = total_fm_loss / num_samples
        avg_dist_loss = total_dist_loss / num_samples
        
        history["train_loss"].append(avg_loss)
        history["train_fm_loss"].append(avg_fm_loss)
        history["train_dist_loss"].append(avg_dist_loss)
        
        # ===== 验证阶段 =====
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_fm_loss = 0.0
            val_dist_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                    x = batch["x"].to(device)
                    p = batch["p"].to(device)
                    batch_idx_tensor = batch["batch"].to(device)
                    ct_idx = batch["cell_type"].to(device)
                    spatial = batch["spatial"]
                    if spatial is not None:
                        spatial = spatial.to(device)
                    
                    z_int, _, _, _ = model.encoder(x, batch_idx_tensor, ct_idx)
                    
                    loss, fm_loss, dist_loss = model.flow_step(
                        z_int, p, batch_idx_tensor, ct_idx,
                        spatial=spatial,
                        lambda_dist=lambda_dist,
                    )
                    
                    val_loss += loss.item() * x.shape[0]
                    val_fm_loss += fm_loss.item() * x.shape[0]
                    val_dist_loss += dist_loss.item() * x.shape[0]
                    val_samples += x.shape[0]
            
            avg_val_loss = val_loss / val_samples
            avg_val_fm_loss = val_fm_loss / val_samples
            avg_val_dist_loss = val_dist_loss / val_samples
            
            history["val_loss"].append(avg_val_loss)
            history["val_fm_loss"].append(avg_val_fm_loss)
            history["val_dist_loss"].append(avg_val_dist_loss)
            
            logger.info(
                f"Epoch {epoch+1}: train={avg_loss:.4f} | "
                f"val={avg_val_loss:.4f}, fm={avg_val_fm_loss:.4f}, dist={avg_val_dist_loss:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}: train={avg_loss:.4f}, fm={avg_fm_loss:.4f}, dist={avg_dist_loss:.4f}"
            )
    
    logger.info("✓ Stage 2完成")
    
    return history
