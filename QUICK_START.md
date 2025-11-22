# CFM-VC 2.x 快速开始指南

**版本**：2.0.1-optimized  
**难度**：初级  
**时间**：5分钟

---

## 🎯 30秒钟了解CFM-VC

CFM-VC是一个用**Flow Matching**学习单细胞扰动响应的深度生成模型。

```
输入数据：单细胞RNA计数 + 扰动标签 + 协变量
       ↓
    VAE编码 (Stage 1)
       ↓
  Flow Matching (Stage 2)
       ↓
生成虚拟细胞表达 + 虚拟组织变化
```

---

## ⚡ 最小化训练脚本（10行代码）

```python
import torch
from torch.utils.data import DataLoader
from cfm_vc.models import CFMVCModel
from cfm_vc.training import train_vae_stage, train_flow_stage
from cfm_vc.data import SingleCellDataset, collate_fn_cfm

# 1. 数据 (AnnData必须有: counts, perturbation, batch, cell_type)
dataset = SingleCellDataset(adata)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_cfm, pin_memory=True)

# 2. 模型
model = CFMVCModel(
    n_genes=adata.n_vars,
    n_batch=len(adata.obs["batch"].unique()),
    n_ct=len(adata.obs["cell_type"].unique()),
    n_perts=len(adata.obs["perturbation"].unique()),
).cuda()

# 3. Stage 1: VAE预训练 (25分钟 @ V100)
train_vae_stage(model, loader, n_epochs=50)

# 4. Stage 2: Flow训练 (35分钟 @ V100)
train_flow_stage(model, loader, n_epochs=50)

# 5. 生成虚拟细胞
model.eval()
with torch.no_grad():
    p_ctrl = torch.zeros(10, model.n_perts).cuda()
    batch_idx = torch.zeros(10, dtype=torch.long).cuda()
    ct_idx = torch.zeros(10, dtype=torch.long).cuda()
    X_ctrl = model.generate_expression(p_ctrl, batch_idx, ct_idx)
```

---

## 📋 数据检查清单

在运行前，确保数据满足以下要求：

```python
import anndata as ad

adata = ad.read_h5ad("your_data.h5ad")

# ✅ 必须检查
assert "counts" in adata.layers, "缺少counts层"
assert "perturbation" in adata.obs, "缺少perturbation列"
assert "batch" in adata.obs, "缺少batch列"
assert "cell_type" in adata.obs, "缺少cell_type列"

# ✅ 数据质量
assert adata.X.shape[0] > 0, "无样本"
assert adata.X.shape[1] > 0, "无基因"
assert adata.layers["counts"].min() >= 0, "计数值不能为负"
assert len(adata.obs["perturbation"].unique()) >= 2, "扰动类别太少"

# ✅ 可选（空间数据）
if "spatial" in adata.obsm:
    assert adata.obsm["spatial"].shape[1] in [2, 3], "空间坐标维度应为2或3"

print("✅ 数据检查通过！")
```

---

## 🚨 常见问题速解

### Q: 出现NaN
**A**: 检查数据中是否有NaN或无穷大
```python
import numpy as np
assert np.all(np.isfinite(adata.layers["counts"])), "数据中有NaN"
```

### Q: 显存溢出 (OOM)
**A**: 减小batch_size或dim
```python
# 改为：
loader = DataLoader(dataset, batch_size=32, ...)  # 从64改为32
model = CFMVCModel(..., dim_int=16, hidden_vae=128, ...)  # 减小隐层
```

### Q: 训练速度慢
**A**: 确保使用GPU和pin_memory
```python
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_cfm, 
                   pin_memory=True)  # ← 这很重要
model.cuda()  # ← 移到GPU
```

### Q: 生成的表达不合理
**A**: 检查VAE预训练质量
```python
# 检查VAE重建loss是否下降
if history_vae['train_loss'][-1] > history_vae['train_loss'][0] * 0.5:
    print("⚠️ VAE预训练不充分，增加轮数")
    # 再训练50轮
    train_vae_stage(model, loader, n_epochs=50)
```

---

## 📊 预期性能

### 资源需求

| 资源 | 最小 | 推荐 | 充足 |
|------|------|------|------|
| GPU显存 | 3GB | 4GB | 8GB+ |
| CPU核心 | 4 | 8 | 16 |
| 样本数 | 1K | 10K | 100K |
| 基因数 | 500 | 2000 | 5000 |

### 时间预期

| 数据规模 | Stage 1 | Stage 2 | 推断 |
|---------|---------|---------|------|
| 1K细胞 | 5分钟 | 7分钟 | 10秒 |
| 10K细胞 | 25分钟 | 35分钟 | 2分钟 |
| 100K细胞 | 4小时 | 6小时 | 20分钟 |

---

## 🔍 调试模式

如果训练不稳定，使用详细日志：

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 运行训练（会输出详细信息）
history = train_vae_stage(
    model, loader, 
    n_epochs=5,  # 先用5轮测试
    nan_check_interval=1,  # 每个batch检查NaN
)

# 检查是否出现NaN
if history["nan_step"] is not None:
    print(f"❌ 在step {history['nan_step']}出现NaN")
    print("→ 检查数据质量或减小学习率")
else:
    print(f"✅ 训练稳定，最终loss: {history['train_loss'][-1]:.4f}")
```

---

## 📚 深度学习资源

- **梯度流问题**：见 README-OPTIMIZATION.md
- **NaN防护**：见 README-OPTIMIZATION.md 第2部分
- **内存优化**：见 README-OPTIMIZATION.md 第3部分
- **代码审计**：见 AUDIT_REPORT.md
- **完整文档**：见 IMPLEMENTATION_SUMMARY.md

---

## ✨ 下一步

### 训练完成后

1. **评估模型质量**
   ```python
   # 在验证集上计算loss
   val_loss = evaluate_model(model, val_loader)
   ```

2. **生成虚拟细胞**
   ```python
   # 见快速开始的第5步
   ```

3. **分析扰动效应**
   ```python
   # 比较control vs perturbed
   effect = X_pert - X_ctrl
   print(f"平均效应: {effect.mean():.4f}")
   ```

### 保存和加载模型

```python
# 保存
torch.save(model.state_dict(), "cfm_vc_model.pth")

# 加载
model = CFMVCModel(...)
model.load_state_dict(torch.load("cfm_vc_model.pth"))
```

---

## 💡 实用技巧

### Tip 1: 使用验证集监测过拟合
```python
history = train_vae_stage(
    model, train_loader,
    val_loader=val_loader,  # ← 传入验证loader
    n_epochs=50,
)
```

### Tip 2: 调整beta增加KL约束
```python
history = train_vae_stage(
    model, loader,
    beta=0.1,  # 降低KL权重，改进重建
    # 或
    beta=10.0,  # 增加KL权重，改进表示
)
```

### Tip 3: 使用lambda_dist加入分布匹配
```python
history = train_flow_stage(
    model, loader,
    lambda_dist=0.05,  # 可选的分布匹配（需要更复杂）
)
```

---

## 🆘 获取帮助

如果还有问题：

1. **检查数据**：99%的问题来自数据质量
   ```python
   # 运行数据检查
   exec(open("check_data.py").read())
   ```

2. **查看日志**：启用DEBUG日志找到具体问题位置

3. **运行测试**：确认代码本身没问题
   ```bash
   pytest cfm_vc/tests/test_models_comprehensive.py -v
   ```

4. **参考文档**：
   - 梯度问题 → README-OPTIMIZATION.md
   - NaN问题 → README-OPTIMIZATION.md Q1
   - 性能问题 → README-OPTIMIZATION.md Q3

---

## 📞 联系信息

遇到特定问题，参考相应文档：

| 问题类型 | 参考文档 |
|---------|---------|
| 梯度失效 | README-OPTIMIZATION.md #1 |
| NaN问题 | README-OPTIMIZATION.md #2 |
| 内存溢出 | README-OPTIMIZATION.md #3 |
| 代码审计 | AUDIT_REPORT.md |
| 实现细节 | IMPLEMENTATION_SUMMARY.md |

---

## ✅ 完成检查

开始训练前，确认：

- [ ] 已读本指南
- [ ] 已检查数据质量
- [ ] 已确认GPU/显存充足
- [ ] 已运行测试: `pytest cfm_vc/tests/`
- [ ] 理解Stage 1和Stage 2的区别

---

**祝训练顺利！** 🚀

最后更新：2024-12-19  
版本：CFM-VC 2.0.1-optimized
