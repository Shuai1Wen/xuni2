好，那我直接给你一套「可以照着写代码」的 pseudo-code，按 **PyTorch + scanpy/scvi-tools 的习惯**来组织：

* 先是 **数据管线接口**（AnnData → PyTorch Dataset/DataLoader）
* 再是 **模型结构**：

  * `EncoderVAE` / `DecoderVAE`
  * `ContextEncoder`（包含协变量 + 空间）
  * `FlowField`（base + perturbation adapter）
  * `CFMVCModel`（封装 VAE + Flow + 采样逻辑）
* 最后是 **训练流程：Stage 1 VAE 预训练 → Stage 2 Flow 训练 → 可选 joint finetune**
* 每个关键算法（Flow Matching 的 OT 路径、velocity 回归）都会单独写清楚。

我会用接近 PyTorch 的伪代码，但不纠结所有细节（比如维度数字），重点是 **结构和调用顺序清晰**。
一些设计与 scVI 的官方实现、Flow Matching 原论文和 CellFlow 的实现形式保持一致，这样你代码时可以对照([GitHub][1])。

---

## 一、数据管线接口（scanpy / AnnData 风格）

### 1.1 使用 AnnData 作为输入

```python
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
```

#### 1.1.1 预处理（scanpy 部分伪代码）

```python
# 读取 & 基本预处理
adata = sc.read_h5ad("perturbation_dataset.h5ad")

# 常规预处理（示例，可根据数据集调整）
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 这里 adata.X 是 log-normalized，计数可放在 adata.layers["counts"] 里
adata.layers["counts"] = adata.raw.X if adata.raw is not None else adata.X.copy()

# 扰动标签 p、协变量 c、空间坐标
# 假设:
#   adata.obs["perturbation"]   -> categorical perturbation id
#   adata.obs["batch"]          -> batch / donor
#   adata.obs["cell_type"]      -> cell type
#   adata.obsm["spatial"]       -> (x, y) 或 (x, y, z)
```

### 1.2 建立 PyTorch Dataset

```python
class SingleCellDataset(Dataset):
    def __init__(self, adata, gene_key="counts", device="cpu"):
        self.adata = adata
        self.X = np.asarray(adata.layers[gene_key])  # raw counts
        # 构造扰动 one-hot / multi-hot 向量
        self.pert_categories = adata.obs["perturbation"].astype("category")
        self.pert_codes = self.pert_categories.cat.codes.values  # [0..K-1]
        self.num_perts = len(self.pert_categories.cat.categories)
        
        # 协变量示例：batch, cell_type (可以拼接 one-hot 或 index)
        self.batch_codes = adata.obs["batch"].astype("category").cat.codes.values
        self.ct_codes    = adata.obs["cell_type"].astype("category").cat.codes.values
        
        # 空间，如果没有则用 None
        self.spatial = adata.obsm.get("spatial", None)
        
        self.device = device
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x_counts = torch.tensor(self.X[idx], dtype=torch.float32)
        
        p_idx = self.pert_codes[idx]
        p = torch.zeros(self.num_perts, dtype=torch.float32)
        p[p_idx] = 1.0
        
        batch = torch.tensor(self.batch_codes[idx], dtype=torch.long)
        ct    = torch.tensor(self.ct_codes[idx], dtype=torch.long)
        
        if self.spatial is not None:
            s = torch.tensor(self.spatial[idx], dtype=torch.float32)  # (2,) or (3,)
        else:
            s = None
        
        return {
            "x": x_counts,
            "p": p,
            "batch": batch,
            "cell_type": ct,
            "spatial": s,
        }
```

**DataLoader：**

```python
train_dataset = SingleCellDataset(adata[adata.obs["split"] == "train"])
val_dataset   = SingleCellDataset(adata[adata.obs["split"] == "val"])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=512, shuffle=False)
```

---

## 二、模型结构 pseudo-code

### 2.1 VAE：Encoder + Decoder（scVI 风格 NB-VAE）

这里跟 scVI 的思想保持一致([Project name not set][2])：

```python
import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderVAE(nn.Module):
    def __init__(self, n_genes, n_batch, n_ct,
                 dim_int=32, dim_tech=8, hidden_dim=256):
        super().__init__()
        self.n_genes = n_genes
        self.n_batch = n_batch
        self.n_ct    = n_ct
        
        # 协变量嵌入（batch, cell_type）
        self.batch_emb = nn.Embedding(n_batch, 8)
        self.ct_emb    = nn.Embedding(n_ct, 8)
        
        input_dim = n_genes + 8 + 8
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # z_int, z_tech 两块 latent
        self.z_int_mean  = nn.Linear(hidden_dim, dim_int)
        self.z_int_logvar= nn.Linear(hidden_dim, dim_int)
        
        self.z_tech_mean  = nn.Linear(hidden_dim, dim_tech)
        self.z_tech_logvar= nn.Linear(hidden_dim, dim_tech)
    
    def forward(self, x, batch_idx, ct_idx):
        """
        x: (B, G) raw counts or normalized values
        batch_idx: (B,)
        ct_idx: (B,)
        """
        batch_e = self.batch_emb(batch_idx)
        ct_e    = self.ct_emb(ct_idx)
        
        h = torch.cat([x, batch_e, ct_e], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        
        z_int_mean   = self.z_int_mean(h)
        z_int_logvar = self.z_int_logvar(h)
        
        z_tech_mean   = self.z_tech_mean(h)
        z_tech_logvar = self.z_tech_logvar(h)
        
        # reparameterize
        eps_int  = torch.randn_like(z_int_mean)
        eps_tech = torch.randn_like(z_tech_mean)
        
        z_int  = z_int_mean  + eps_int  * torch.exp(0.5 * z_int_logvar)
        z_tech = z_tech_mean + eps_tech * torch.exp(0.5 * z_tech_logvar)
        
        kl_int  = -0.5 * torch.sum(1 + z_int_logvar  - z_int_mean.pow(2)  - z_int_logvar.exp(),  dim=-1)
        kl_tech = -0.5 * torch.sum(1 + z_tech_logvar - z_tech_mean.pow(2) - z_tech_logvar.exp(), dim=-1)
        
        return z_int, z_tech, kl_int, kl_tech
```

Decoder：简化 NB（真正实现时可参考 scVI 的 negative binomial parameterization）([Project name not set][2])：

```python
class DecoderVAE(nn.Module):
    def __init__(self, n_genes, dim_int=32, dim_tech=8, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim_int + dim_tech, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_out = nn.Linear(hidden_dim, n_genes)
        # log-dispersion per gene
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
    
    def forward(self, z_int, z_tech):
        h = torch.cat([z_int, z_tech], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # mean of NB
        mean = torch.exp(self.mean_out(h))  # ensure positivity
        
        theta = torch.exp(self.log_theta)
        return mean, theta

def nb_log_likelihood(x, mean, theta, eps=1e-8):
    """
    x: counts (B, G)
    mean, theta: (B, G) or (G,) broadcast
    """
    # NB log-likelihood: log Gamma(x+theta) - log Gamma(theta) - log(x+1)
    #                    + theta*log(theta/(theta+mean)) + x*log(mean/(theta+mean))
    lgamma_term = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    log_p = (lgamma_term
             + theta * (torch.log(theta + eps) - torch.log(theta + mean + eps))
             + x * (torch.log(mean + eps) - torch.log(theta + mean + eps)))
    return torch.sum(log_p, dim=-1)  # (B,)
```

### 2.2 Context encoder（协变量 + 空间）

```python
class ContextEncoder(nn.Module):
    def __init__(self, n_batch, n_ct, p_dim, spatial_dim=None, hidden_dim=64):
        super().__init__()
        self.batch_emb = nn.Embedding(n_batch, 8)
        self.ct_emb    = nn.Embedding(n_ct, 8)
        
        self.spatial_dim = spatial_dim
        if spatial_dim is not None:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(spatial_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
            )
        
        # simple MLP for perturbation adapter coefficients α(p)
        self.pert_mlp = nn.Sequential(
            nn.Linear(p_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 输出 L 维 coeff，由 FlowField 再映射
        )
    
    def forward(self, p, batch_idx, ct_idx, spatial=None):
        """
        p: (B, K) one-hot / multi-hot
        batch_idx, ct_idx: (B,)
        spatial: (B, d_s) or None
        """
        batch_e = self.batch_emb(batch_idx)
        ct_e    = self.ct_emb(ct_idx)
        
        if spatial is not None:
            s_e = self.spatial_mlp(spatial)
        else:
            s_e = torch.zeros_like(batch_e)
        
        # coarse context can用于 h 的一部分
        context = torch.cat([batch_e, ct_e, s_e], dim=-1)
        
        # 这里为了简化，我们用 cell_type embedding + p 做 adapter 输入
        pert_input = torch.cat([p, ct_e], dim=-1)
        pert_alpha = self.pert_mlp(pert_input)  # (B, L')，后面可以线性映射到 L
        
        return context, pert_alpha
```

### 2.3 FlowField：base + perturbation adapter + Flow Matching 算法核心

我们用 **OT 直线插值路径**，目标速度就是 (u_t = z_1 - z_0)([arXiv][3])。

```python
class FlowField(nn.Module):
    def __init__(self, dim_int, context_dim, alpha_dim, 
                 hidden_dim=128, n_basis=16):
        """
        dim_int: dimension of z_int
        context_dim: dim of context from ContextEncoder
        alpha_dim: dim of pert_alpha (from ContextEncoder)
        n_basis: number of basis vector fields
        """
        super().__init__()
        
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
        )
        
        # trunk
        self.trunk = nn.Sequential(
            nn.Linear(dim_int + 16 + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # base field W_base
        self.base_head = nn.Linear(hidden_dim, dim_int)
        
        # basis fields W_B
        self.basis_head = nn.Linear(hidden_dim, n_basis * dim_int)
        
        # adapter: map alpha_dim to n_basis coefficients
        self.alpha_head = nn.Linear(alpha_dim, n_basis, bias=False)
    
    def forward(self, z_t, t, context, pert_alpha):
        """
        z_t: (B, dim_int)
        t: (B,) in [0,1]
        context: (B, context_dim)
        pert_alpha: (B, alpha_dim)  # from ContextEncoder
        """
        t_embed = self.time_mlp(t.unsqueeze(-1))  # (B, 16)
        h_in = torch.cat([z_t, t_embed, context], dim=-1)
        h = self.trunk(h_in)  # (B, hidden_dim)
        
        v_base = self.base_head(h)  # (B, dim_int)
        
        basis = self.basis_head(h)  # (B, n_basis * dim_int)
        B = basis.view(-1, self.n_basis, self.dim_int)  # (B, n_basis, dim_int)
        
        # 计算每个 basis 的权重
        coeff = self.alpha_head(pert_alpha)  # (B, n_basis)
        coeff = coeff.unsqueeze(-1)          # (B, n_basis, 1)
        
        v_eff = torch.sum(coeff * B, dim=1)  # (B, dim_int)
        
        v = v_base + v_eff
        return v
```

### 2.4 整体 CFM-VC 模型封装

```python
class CFMVCModel(nn.Module):
    def __init__(self, n_genes, n_batch, n_ct, n_perts,
                 dim_int=32, dim_tech=8,
                 hidden_vae=256, hidden_ctx=64,
                 hidden_flow=128, n_basis=16,
                 spatial_dim=None):
        super().__init__()
        
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
        
        self.context_encoder = ContextEncoder(
            n_batch=n_batch,
            n_ct=n_ct,
            p_dim=n_perts,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_ctx,
        )
        
        # context_dim = batch_emb(8) + ct_emb(8) + spatial_embed(16 or 0)
        context_dim = 8 + 8 + (16 if spatial_dim is not None else 8)  # 示例
        alpha_dim = hidden_ctx  # from ContextEncoder.pert_mlp output
        
        self.flow = FlowField(
            dim_int=dim_int,
            context_dim=context_dim,
            alpha_dim=alpha_dim,
            hidden_dim=hidden_flow,
            n_basis=n_basis,
        )
    
    # ---------- VAE part ----------
    def vae_forward(self, x, batch_idx, ct_idx):
        # encode
        z_int, z_tech, kl_int, kl_tech = self.encoder(x, batch_idx, ct_idx)
        # decode
        mean, theta = self.decoder(z_int, z_tech)
        nb_ll = nb_log_likelihood(x, mean, theta)
        
        # VAE loss: -log p + KL
        loss_vae = -torch.mean(nb_ll) + torch.mean(kl_int + kl_tech)
        return loss_vae, z_int, z_tech
    
    # ---------- Flow Matching step ----------
    def flow_step(self, z_int, p, batch_idx, ct_idx, spatial=None,
                  lambda_dist=0.0):
        """
        z_int: (B, dim_int) (treated as z_1)
        FM with OT path:
          z0 ~ N(0, I)
          z_t = (1-t) * z0 + t * z1
          u_t = z1 - z0
        """
        B, d = z_int.shape
        z1 = z_int.detach()  # stop grad from VAE during flow training (stage 2)
        z0 = torch.randn_like(z1)
        t = torch.rand(B, device=z1.device)  # Uniform[0,1]
        
        z_t = (1.0 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z1
        u_t = z1 - z0  # (B, d)
        
        context, pert_alpha = self.context_encoder(p, batch_idx, ct_idx, spatial)
        v_pred = self.flow(z_t, t, context, pert_alpha)  # (B, d)
        
        fm_loss = torch.mean((v_pred - u_t) ** 2)
        
        # 可选分布匹配 (MMD/ED)，这里只是占位符
        dist_loss = torch.tensor(0.0, device=z1.device)
        if lambda_dist > 0:
            # 例如: 对每个 perturbation 采 z0 → z1_hat, 和 encoder 产生的 z1 做 MMD
            # 这里 pseudo-code，不展开实现细节
            dist_loss = compute_mmd_loss(z1, self.sample_z1_from_flow(p, batch_idx, ct_idx, spatial))
        
        total_loss = fm_loss + lambda_dist * dist_loss
        return total_loss, fm_loss, dist_loss
    
    # ---------- Sampling from flow ----------
    @torch.no_grad()
    def sample_z1_from_flow(self, p, batch_idx, ct_idx, spatial=None, n_steps=20):
        """
        从 z0 ~ N(0,I) 积分 ODE 得到 z1 (Euler 或更好 solver)
        """
        B = p.size(0)
        z = torch.randn(B, self.encoder.z_int_mean.out_features, device=p.device)
        t = torch.zeros(B, device=p.device)
        dt = 1.0 / n_steps
        
        for _ in range(n_steps):
            context, pert_alpha = self.context_encoder(p, batch_idx, ct_idx, spatial)
            v = self.flow(z, t, context, pert_alpha)
            z = z + dt * v
            t = t + dt
        
        return z
    
    @torch.no_grad()
    def generate_expression(self, p, batch_idx, ct_idx, spatial=None, n_steps=20):
        """
        从 noise 生成表达谱
        """
        z_int = self.sample_z1_from_flow(p, batch_idx, ct_idx, spatial, n_steps)
        # 默认用 z_tech = 0 (或从 prior 采样)
        z_tech = torch.zeros(z_int.size(0), self.encoder.z_tech_mean.out_features,
                             device=z_int.device)
        mean, theta = self.decoder(z_int, z_tech)
        # 可以根据 NB 随机采样，也可以直接返回 mean 作为期望表达
        return mean
```

---

## 三、训练流程 pseudo-code

### 3.1 Stage 1：VAE 预训练（只训练 Encoder/Decoder）

```python
from torch.optim import AdamW

model = CFMVCModel(
    n_genes=adata.n_vars,
    n_batch=train_dataset.pert_categories.cat.categories.size,  # 实际需重写
    n_ct=adata.obs["cell_type"].astype("category").cat.categories.size,
    n_perts=train_dataset.num_perts,
    dim_int=32,
    dim_tech=8,
).to(device)

optimizer_vae = AdamW(
    list(model.encoder.parameters()) + list(model.decoder.parameters()),
    lr=1e-3
)

for epoch in range(n_epochs_vae):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch["x"].to(device)
        batch_idx = batch["batch"].to(device)
        ct_idx    = batch["cell_type"].to(device)
        
        optimizer_vae.zero_grad()
        loss_vae, z_int, z_tech = model.vae_forward(x, batch_idx, ct_idx)
        loss_vae.backward()
        optimizer_vae.step()
        
        total_loss += loss_vae.item() * x.size(0)
    
    total_loss /= len(train_dataset)
    print(f"Epoch {epoch}: VAE loss = {total_loss:.4f}")
    
    # 可选：在 val_loader 上评估重建误差
```

> 实际上可以直接用 scvi-tools 的 `SCVI` 模型预训练 latent，然后导出 `z_int`，再接 Flow。这里写成自定义是为了完整性，但实现时你完全可以复用 `scvi-tools` 的 Encoder/Decoder([GitHub][1])。

---

### 3.2 Stage 2：Flow Matching 训练（只训练 Flow + ContextEncoder）

```python
optimizer_flow = AdamW(
    list(model.context_encoder.parameters()) + list(model.flow.parameters()),
    lr=1e-4
)

for epoch in range(n_epochs_flow):
    model.train()
    total_loss_epoch = 0
    for batch in train_loader:
        x = batch["x"].to(device)
        p = batch["p"].to(device)
        batch_idx = batch["batch"].to(device)
        ct_idx    = batch["cell_type"].to(device)
        spatial   = batch["spatial"]
        if spatial is not None:
            spatial = spatial.to(device)
        
        # 先通过 encoder 得到 z_int (z_tech 可忽略)
        with torch.no_grad():
            z_int, z_tech, _, _ = model.encoder(x, batch_idx, ct_idx)
        
        optimizer_flow.zero_grad()
        total_loss, fm_loss, dist_loss = model.flow_step(
            z_int, p, batch_idx, ct_idx, spatial,
            lambda_dist=0.0,  # 初始阶段可设为 0
        )
        total_loss.backward()
        optimizer_flow.step()
        
        total_loss_epoch += total_loss.item() * x.size(0)
    
    total_loss_epoch /= len(train_dataset)
    print(f"Epoch {epoch}: Flow total loss = {total_loss_epoch:.4f}")
    
    # 可选：每隔若干 epoch 在 val 上评估 R^2 / corr，通过 generate_expression 做预测
```

### 3.3 可选 Stage 3：joint finetune（小学习率）

如果你想进一步统一 latent 与 flow，可以在一个 joint 阶段微调全部参数（lr 小很多）：

```python
optimizer_joint = AdamW(model.parameters(), lr=5e-5)

for epoch in range(n_epochs_joint):
    model.train()
    for batch in train_loader:
        x = batch["x"].to(device)
        p = batch["p"].to(device)
        batch_idx = batch["batch"].to(device)
        ct_idx    = batch["cell_type"].to(device)
        spatial   = batch["spatial"]
        if spatial is not None:
            spatial = spatial.to(device)
        
        optimizer_joint.zero_grad()
        
        # 1) VAE loss
        loss_vae, z_int, z_tech = model.vae_forward(x, batch_idx, ct_idx)
        
        # 2) Flow loss (这次 z_int 不 detach，让 encoder 也受 flow 约束)
        flow_loss, fm_loss, dist_loss = model.flow_step(
            z_int, p, batch_idx, ct_idx, spatial,
            lambda_dist=0.01
        )
        
        loss = loss_vae + flow_loss
        loss.backward()
        optimizer_joint.step()
```

> 这一步是 optional，如果你发现 joint 阶段不稳定，可以完全跳过，论文里写清楚“我们采用 two-stage 训练”，这在很多 Flow + VAE 模型里是常见做法([arXiv][3])。

---

## 四、核心算法逐点总结（方便你写 Methods）

1. **VAE（scVI 风格 NB-VAE）：**

   * 生成模型：
     [
     z_{\text{int}},z_{\text{tech}} \sim \mathcal{N}(0,I)
     ]
     [
     X \sim \text{NB}(\mu_\psi(z_{\text{int}},z_{\text{tech}},c),\theta_\psi)
     ]
   * 损失：
     [
     \mathcal{L}*{VAE} =
     -\mathbb{E}*{q_\phi} \log p_\psi(X\mid z,c)

     * \beta ,\text{KL}\big(q_\phi(z\mid X,c)|N(0,I)\big)
       ]

2. **Flow Matching（OT 路径）：**（Lipman 2023）([arXiv][3])

   * 对每个样本：

     * encode 得 (z_1 = z_{\text{int}})
     * 采 (z_0 \sim \mathcal{N}(0,I))，(t \sim \mathcal{U}[0,1])
     * (z_t = (1-t)z_0 + t z_1)
     * 目标速度 (u_t = z_1 - z_0)
   * 向量场：
     [
     v_\theta(z_t,t\mid p,c,s) = v_{base}(h) + v_{eff}(h,p)
     ]
     其中
     [
     h = \text{MLP}*{trunk}([z_t,\gamma(t),enc_c(c,s)])
     ]
     [
     v*{base} = W_{base}h,\quad v_{eff} = \sum_j \alpha_j(p,ct), b_j(h)
     ]
   * 损失：
     [
     \mathcal{L}*{FM} =
     \mathbb{E}|v*\theta(z_t,t\mid p,c,s) - (z_1 - z_0)|^2
     ]

3. **空间 context（可选）：**

   * 构造邻域图 (G)，用简单 GNN/MLP 计算空间嵌入 (\mathbf{s})；
   * 将 (\mathbf{s}) 拼入 context encoder `enc_c` 中，Flow 不变，这一点类似现有的空间深度生成模型（DestVI, scVIVA 等）。

4. **生成 / 预测：**

   * 分布级预测：采样 (z_0)，在目标 ((p,c,s)) 下积分 ODE 至 t=1 得 (z_1)，通过 NB decoder 得 (\hat{X})；
   * 细胞级近似 counterfactual：对选定细胞，将其 latent state/种子固定，改变 p 重复上一步。



[1]: https://github.com/scverse/scvi-tools?utm_source=chatgpt.com "GitHub - scverse/scvi-tools: Deep probabilistic analysis of single-cell ..."
[2]: https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html?utm_source=chatgpt.com "scVI — scvi-tools"
[3]: https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "[2210.02747] Flow Matching for Generative Modeling - arXiv.org"
