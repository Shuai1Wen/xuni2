直接给你结论：下面这个版本我会当成 **“CFM‑VC 2.x 的最小完整、可实现、又有因果味道的方案”**，
只针对 **scRNA‑seq + 空间转录组**，loss 数量控制在「少而有用」，模型结构也是一条清晰主线，而不是一堆 patch。

我分两大块：

1. **模型完整设计方案（CFM‑VC 2.x）**
2. **实验设计方案（怎么证明它值得一区）**

中间会穿插说明：每个设计点是 **为什么这么干**、风险在哪里、怎么避免“理论好看但跑不出来”。

---

## 一、CFM‑VC 2.x：统一、可实现的模型设计（只针对 scRNA + 空间）

### 1.1 问题和数据设定（只做一件事：扰动 → 表达）

**数据类型：**

1. **单细胞扰动 scRNA‑seq**

   * 输入：计数矩阵 (\mathbf{X}*i \in \mathbb{R}^{G}*{\ge0})
   * 扰动：(\mathbf{p}_i \in {0,1}^K)（多基因 KO / 多药组合，可扩展到 dose）
   * 协变量：(\mathbf{c}_i)：batch、donor、time、cell type、实验条件等

2. **空间转录组（单细胞或经 deconvolution 到单细胞）**

   * 额外有空间坐标 ((x_i,y_i)) 或邻接图 (G=(V,E))([Project name not set][1])

**任务：**

* 给定 ((\mathbf{X},\mathbf{p},\mathbf{c}))，学到一个生成模型，可以：

  1. 预测已见扰动条件下的新细胞表达分布（in‑distribution）；
  2. 预测未见扰动/组合/细胞类型（OOD）；
  3. 在空间场景下，预测局部扰动对组织表达格局的影响。

**因果视角（温和版）：**

* 每个细胞 i 的潜在结局 (Y_i(\mathbf{p}))：在协变量 (\mathbf{c}_i) 给定下，对扰动组合 (\mathbf{p}) 的转录结果。
* 我们不做“严格定理”，而是采用和 CellFlow / CFM‑GP 类似的假设：**在给定协变量和嵌入空间后，扰动 assignment 近似可忽略**，Flow 学到的是 (P(Y(\mathbf{p})\mid \mathbf{c})) 的近似生成过程([bioRxiv][2])。
* 也就是说：我们更强调 **“分布级 counterfactual + cell‑state aware”**，而不是严格 cell‑pair 精确反事实。

---

### 1.2 模型总览（一句话版）

> **先用 NB‑VAE / scVI 风格建立一个“内在细胞状态 latent 空间”，
> 再在这个 latent 上用 Flow Matching 学一个向量场
> (,v_\theta(z,t\mid \mathbf{p},\mathbf{c},\mathbf{s}))，
> 把「未扰动分布」推到「给定扰动分布」，
> 其中空间信息 (\mathbf{s}) 只是 context 的一部分。**

主干就三块：

1. **表示层：RNA NB‑VAE（scVI 类）**
2. **向量场：Flow Matching 上的 base + perturbation‑adapter 向量场**
3. **空间 context：把空间信息编码进 context，不搞额外的空间 loss**

---

### 1.3 表示层：因果友好的 NB‑VAE（但保持简单）

我们尽量靠近 scVI / DestVI / scVIVA 这条成熟路线([Project name not set][1])：

#### (1) 编码器（encoder）

* 输入：((\mathbf{X}_i,\mathbf{c}_i))
* 输出 latent：
  [
  \mathbf{z}*i = [\mathbf{z}^{(i)}*{\text{int}},; \mathbf{z}^{(i)}_{\text{tech}}]
  ]

  * (\mathbf{z}*{\text{int}} \in \mathbb{R}^{d*{int}})：内在（“basal”）细胞状态
  * (\mathbf{z}*{\text{tech}} \in \mathbb{R}^{d*{tech}})：技术/批次/环境噪声

参数化：
[
q_\phi(\mathbf{z}\mid \mathbf{X},\mathbf{c}) = \mathcal{N}(\mu_\phi(\mathbf{X},\mathbf{c}), \text{diag}(\sigma^2_\phi(\mathbf{X},\mathbf{c})))
]

#### (2) 解码器（decoder）

* 采用负二项似然（scVI 风格）：([suyccc.github.io][3])
  [
  p_\psi(\mathbf{X}\mid \mathbf{z}*{\text{int}},\mathbf{z}*{\text{tech}},\mathbf{c})
  = \text{NB}\big(\mu_\psi(\mathbf{z}*{\text{int}},\mathbf{z}*{\text{tech}},\mathbf{c}),; \theta_\psi(\mathbf{z}*{\text{int}},\mathbf{z}*{\text{tech}},\mathbf{c})\big)
  ]

#### (3) VAE 损失（模块 A）

[
\mathcal{L}*{VAE}
= -\mathbb{E}*{q_\phi} \log p_\psi(\mathbf{X}\mid \mathbf{z},\mathbf{c})

* \beta ,\text{KL}\big(q_\phi(\mathbf{z}\mid \mathbf{X},\mathbf{c});|;\mathcal{N}(0,I)\big)
  ]

**可选的轻量对抗（但不是必须）：**

* 一个小分类器 D 试图从 (\mathbf{z}_{\text{int}}) 预测扰动 label (\mathbf{p})，VAE 通过 gradient reversal 让 D 的准确率接近随机（CPA / scVIVA 有类似思路）([cpa-tools.readthedocs.io][4])。
* 这只是一个 **弱约束**：让「扰动信息」更倾向通过 Flow 学，而不是直接塞进 latent。
* 实现时可以先不开（(\lambda_{adv}=0)），等主模型稳定后再尝试打开作为增益。

> 这里的优化是**针对性**的：
> 用最成熟的 NB‑VAE 路线保证 latent 稳定，同时只加一个非常轻的“去 p 信息”门槛，而不是堆很多 disentanglement 正则。

---

### 1.4 向量场：一个 trunk + 一个 perturbation adapter，而不是一堆 g_k MLP

这是整个 CFM‑VC 的“灵魂”，也是我们最小化复杂度又保留因果结构的关键。

#### (1) Flow Matching 的概率路径

采用 Lipman 等提出的 **直线插值路径**([arXiv][5])，也是 scDFM / CFM‑GP 等常用的选择([OpenReview][6])：

* 对每个样本 ((\mathbf{X},\mathbf{p},\mathbf{c}))：

  1. 编码得到 (z_1 = z_{\text{int}})；
  2. 采样 (z_0 \sim \mathcal{N}(0,I))；
  3. 取 (t \sim \mathcal{U}[0,1])，插值：
     [
     z_t = (1-t)z_0 + t z_1
     ]
  4. 目标速度（analytical）：
     [
     u_t(z_0,z_1) = z_1 - z_0
     ]

Flow Matching loss：
[
\mathcal{L}*{FM}
= \mathbb{E}\big[\big| v*\theta(z_t,t\mid \mathbf{p},\mathbf{c},\mathbf{s}) - (z_1-z_0)\big|^2\big]
]

其中 (\mathbf{s}) 是空间相关信息（后面讲）。

#### (2) 向量场结构：**一个 trunk + 低秩 perturbation adapter**

我们不用“每个基因一个小 MLP”，那样太重、易过拟合。
改成下面这种 **紧凑 + 融洽** 的形式：

1. 先算一个共享 trunk 表征：
   [
   h = \text{MLP}_{\text{trunk}}\big([z_t;,\gamma(t);;\text{enc}_c(\mathbf{c},\mathbf{s})]\big)
   ]

   * (\gamma(t))：时间 embedding（正弦或小 MLP）
   * (\text{enc}_c(\mathbf{c},\mathbf{s}))：协变量 + 空间的信息编码（例如 cell type, batch, (x,y)，邻域平均表达等）

2. **基线向量场（未扰动 dynamic）**：
   [
   v_{\text{base}} = W_{\text{base}} h
   \quad (\text{shape：} d_{int} \times d_h)
   ]

3. **有限个“效应基”向量场（basis fields）**：

   * 用一组 basis 来表示各种扰动效应，而不是每个 k 一套全新网络。
     [
     B(h) = \text{reshape}(W_B h) \in \mathbb{R}^{L \times d_{int}}
     ]
     这里 (L) 是 basis 数（例如 8–16），相当于 (L) 个候选 effect 向量场 (b_j(h))。

4. **扰动 adapter（FiLM 风格）**：

   * 用一个小 MLP 把 (\mathbf{p})（必要时加 coarse cell type）映射到系数 (\alpha(\mathbf{p}) \in \mathbb{R}^L)：
     [
     \alpha = \text{MLP}*p(\mathbf{p}, c*{\text{coarse}})
     ]
   * 效应向量场：
     [
     v_{\text{eff}} = \sum_{j=1}^L \alpha_j b_j(h)
     ]

最终向量场：

[
v_\theta(z_t,t\mid \mathbf{p},\mathbf{c},\mathbf{s})
= v_{\text{base}}(h) + v_{\text{eff}}(h,\mathbf{p})
]

**这样设计的好处：**

* **不是拼接**：所有 field 共用同一个 trunk (h)，结构紧凑而统一；
* vs CFM‑GP/scDFM：我们显式拆出了 base vs effect，但没有为每个 perturbation开一整个子网络，只是通过 adapter 改 basis 的线性组合，既可解释又不爆参([OpenReview][6])；
* 和 CPA/CellFlow 的“条件模块化”思想类似：perturbation 换成了「调制向量场的 adapter」，而不是直接在 latent 上做线性叠加([GitHub][7])。

#### (3) “因果”约束尽量靠结构，而不是堆 loss

* 对于 **非靶向对照（p=0）**：可以把 MLP_p 设计成满足
  [
  \text{MLP}*p(\mathbf{0},c*{\text{coarse}}) = \mathbf{0}
  ]
  例如去掉 bias，只保留线性层和非线性，使 (\mathbf{p}=0) 时自然输出全 0 → effect 自动为 0。
  无需额外 (\mathcal{L}_{zero})。
* 对于 **跨 batch / cell type**，我们尽量让 confounder 进 encoder / (\text{enc}_c) 而不是进 (\text{MLP}_p)，即 perturbation adapter 只用 (\mathbf{p}) 或 coarse cell type，而不直接看 batch。
  通过结构减少把 batch 学成“伪 effect”的机会（这一点和 CFM‑GP 学 cell‑type‑agnostic  perturbation 的思路类似）([arXiv][8])。

> 这就是有针对性的“简化”：
> 很多原本可以作为 loss 的内容（p=0 effect≈0、batch‑invariance），尽量通过 **网络结构硬编码**，避免再加一堆调不动的 λ。

---

### 1.5 空间信息如何进入（只做 context，不搞新 loss）

我们只考虑两种常见 ST 场景：([Project name not set][1])

1. **近似单细胞分辨率（如 MERFISH/seqFISH / 单细胞 ST）**
2. **spot‑level ST + scRNA deconvolution 到单细胞（SpatialScope / DestVI 等做过）**

我们不发明新模型，而是把已有经验融进 context：

1. 构图：

   * 用 (x,y) 做 kNN 或 Delaunay graph 得到邻居 (\mathcal{N}(i))([ScienceDirect][9])

2. 计算一个简洁的空间嵌入 (\mathbf{s}_i)：

   * 简单版：
     [
     \mathbf{s}*i = \text{MLP}*{xy}(x_i,y_i)
     ]
   * 稍强版：
     [
     \mathbf{s}*i = \text{GNN}\big({z^{(j)}*{\text{int}}}_{j\in\mathcal{N}(i)}\big)
     ]
     类似 scVIVA / scVIVA‑spatial 中的邻域编码([Project name not set][10])。

3. 把 (\mathbf{s}_i) 拼进 context encoder：
   [
   \text{enc}_c(\mathbf{c},\mathbf{s}) = \text{MLP}_c([\mathbf{c};\mathbf{s}])
   ]

然后 **Flow 完全不变**，只是 context 更丰富：

* 这样，CFM‑VC 在空间数据上自然变成一个可以做「虚拟组织扰动」的模型，和 Celcomen 那种图生成因果模型属于同一 family，但结构更简单（Celcomen 是 GNN +显式 do‑calculus 分解）([arXiv][11])。

> 这是非常“融洽”的：
> (\mathbf{s}) 只影响 trunk 的 context，而向量场形式完全统一。
> 不会有“一套 for scRNA、一套 for spatial”的割裂感。

---

### 1.6 训练目标（真的是 2+1 块，而不是 5 块）

我们把所有东西归结成 **两大模块 + 一个可选小正则**：

#### 模块 A：表示层损失 (\mathcal{L}_{VAE})

上面已经写了，不再重复。可以直接用 scVI 的实现/超参当起点([suyccc.github.io][3])。

#### 模块 B：Flow + 分布匹配 (\mathcal{L}_{flow})

> 推荐起步版：**只用 Flow Matching**，再视需要加轻量分布项。

基础版：

[
\mathcal{L}*{flow} = \mathcal{L}*{FM}
]

增强版（scDFM 风格）：在此基础上加一个小权重的分布匹配项：([OpenReview][6])

[
\tilde{\mathcal{L}}*{dist}
= \sum*{p\in \mathcal{P}*{train}}
\alpha_p ,\text{MMD}\big(
P*\theta(z_1\mid do(p),c),;
P_{obs}(z_{\text{int}}\mid p,c)
\big)
]

总的 flow 模块 loss：

[
\mathcal{L}^{tot}*{flow} =
\mathcal{L}*{FM} + \lambda_{dist},\tilde{\mathcal{L}}_{dist}
]

* (\lambda_{dist}) 可以从 0 开始（只 flow），确认稳定后再设 0.1、0.01 这种量级。
* MMD/ED 在 perturbation generative 模型中已经被广泛使用，属于有经验可循的组件([OpenReview][6])。

#### 模块 C：可选的轻量因果正则 (\mathcal{L}_{reg})

如果你希望进一步“拉出 effect”，可以加一个很小权重的正则项，例如：

* 只在 control 条件下惩罚 adapter 的输出大小：
  [
  \mathcal{L}*{reg} = \mathbb{E}*{p=0}|\alpha(\mathbf{p})|^2
  ]
* (\lambda_{reg}) 可以取 1e‑3 这类极小值，只是鼓励 p=0 时 effect 真正为 0。

> 这就是全部：
> **主线只需要 (\mathcal{L}*{VAE}+\mathcal{L}*{FM})**，
> 分布项 (\tilde{\mathcal{L}}*{dist}) 和 (\mathcal{L}*{reg}) 都是“小开关”，不用的时候就关闭，不会强行塞一堆 λ 进来。

---

### 1.7 推断与使用方式（虚拟细胞 / 虚拟组织）

训练好之后：

1. **分布级预测**（主要对比 CellFlow / CFM‑GP / CPA）：([bioRxiv][2])

   * 从 (z_0\sim\mathcal{N}(0,I)) 采样多条轨迹，指定 ((\mathbf{p},\mathbf{c},\mathbf{s}))，积分到 t=1 得到 (z_1)，再通过 NB 解码得到 (\hat{\mathbf{X}})，即 (P_\theta(Y(\mathbf{p})\mid \mathbf{c},\mathbf{s})) 的样本。

2. **对 control 细胞做反事实**

   * 选真实 control 细胞 (\mathbf{X}*{ctrl})，encode 得到 (z*{1,ctrl})；
   * 在生成端，用同一 (z_0) 种子，在 p=0 和 p=p' 下分别跑 Flow，得到两种终点分布，再解码对比。
   * 这不是严格 one‑to‑one mapping，但提供了 **cell‑state / cell‑type 级别的 counterfactual 分布**（和 CPA、GEARS 的评估方式类似）([bioRxiv][12])。

3. **空间反事实组织**

   * 对 ST 数据：对某个区域/细胞群，将 (\mathbf{p}) 从 0 设为某 perturbation，其他不变，通过 Flow 生成新的表达，再投回空间 map，观察局部/全局表达格局变化；
   * 可用来回答“在这个组织环境下，如果这块区域受到药物 A，会发生什么”的问题，与 Celcomen 的目标接近，但实现途径是 Flow 而不是纯 GNN([arXiv][11])。

> 这里我们有意不吹“严谨 individual‑level causality”，
> 把重点放在 **“分布 + cell‑state aware + 空间虚拟组织”**，这样更诚实也更符合现有 data 条件。

---

## 二、后续实验设计方案（怎么证明这东西值得一区）

下面这块是按一区方法论文的标准来的，尽量控制在你半年内**真正能做完**的量级。

### 2.1 数据集选择（scRNA + 空间）

#### (1) scRNA 单细胞扰动

目标：至少涵盖 3 种场景（基因 KO、药物、细胞因子），对应 CPA/GEARS/CellFlow/scDFM/CFM‑GP 常用数据([bioRxiv][2])。

* 典型选择（举例）：

  1. CRISPR KO screen（Replogle/K562、Norman 等）
  2. 药物 dose–response 数据（类似 Sci‑Plex 系列）
  3. 细胞因子刺激 PBMC 数据（IFN‑β 等）

这些在近期 perturbation review 和 CFM‑GP / scDFM / CellFlow 的 paper 里都有用到，可直接借鉴 split 设置([OpenReview][6])。

#### (2) 空间转录组 + 条件差异

严格意义上的 ST + perturbation 数据不多，但可以利用：

1. **不同组织区域 / 条件** 作为“quasi‑perturbation”（例如肿瘤 vs 邻近正常、治疗前 vs 后），和 Celcomen / STORIES / CASCAT 的设定类似([arXiv][11])。
2. 若有真正的 ST perturbation（如局部药物注射、体内模型）那是加分项，但不是必需；你也可以先在 public ST（如人类脾、肺癌）上做“空间 counterfactual of region identity” 的验证([arXiv][11])。

### 2.2 Baseline 模型（要覆盖两条线：perturbation & flow）

**经典 perturbation 模型：**

* CPA（compositional perturbation autoencoder）([GitHub][7])
* GEARS（graph‑enhanced simulator）([GitHub][13])
* 1–2 个 VAE/Transformer 类模型（如 scGen / 综述中推荐的模型）([ScienceDirect][14])

**Flow / diffusion 线：**

* scDFM（distributional flow matching）([OpenReview][6])
* CFM‑GP（conditional FM for gene perturbation across cell types）([arXiv][8])
* CellFlow（flow matching for single-cell phenotypes）([bioRxiv][2])

**空间 generative / causal baseline：**

* 至少 1 个 deep generative ST 模型（如 scVIVA / DestVI / stDiffusion）([Project name not set][10])
* Celcomen（空间因果 GNN，可对比 counterfactual 空间表达）([arXiv][11])

> 关键是：你要站在 **CPA + GEARS + scDFM + CFM‑GP + CellFlow + 1–2 个 ST 模型** 之上，才有资格说自己是“新一代 virtual cell / tissue 模型”。

### 2.3 任务与指标（和现有 benchmark 对齐）

**任务划分：**

1. **In‑distribution**：已见 perturbation & cell type 上，新细胞的预测。
2. **Unseen perturbation**：hold‑out 一部分 perturbation，在其上评估泛化（CPA / GEARS / CellFlow 都做过）([bioRxiv][12])。
3. **Unseen combination**：只用单基因/单药训练，对组合进行预测（CPA / GEARS 标配）([bioRxiv][12])。
4. **Cross‑context**：不同 cell type / donor / batch / time 的 transfer（CFM‑GP 的重点）([arXiv][8])。
5. **空间任务**：

   * 根据局部条件预测不同区域的表达分布；
   * 做“虚拟 perturbation map”：改变特定区域的 p，观察全组织响应。

**指标：**

* 点预测：MSE / MAE / R²（在 gene × cell 上）([OpenReview][6])
* 相关性：Spearman / Pearson 对 logFC 或 Δexpression
* DE overlap / pathway 富集一致性：和 CPA / GEARS / CellFlow / CFM‑GP 一致([bioRxiv][12])
* 分布级指标：

  * latent 上的 MMD / ED；
  * gene‑set level 的 Wasserstein/FID（参考 Flow / diffusion 文献）([OpenReview][6])
* 空间结构指标：

  * g.e. Moran’s I / variogram / 空间自相关变化，或 STORIES/CASCAT 使用的轨迹一致性指标([Nature][15])。

### 2.4 实验结构（从“最小可行”到“完全体”，四步走）

**Step 1：CFM‑VC‑base（最小版）**

* encoder：NB‑VAE（scVI 原样）
* vector field：conditional FM，但 **不拆 base+adapter**，即
  [
  v_\theta(z,t\mid p,c,s) = \text{MLP}(z,t,p,c,s)
  ]
  相当于“CFM‑GP with scVI latent”。
* loss：(\mathcal{L}*{VAE} + \mathcal{L}*{FM})，无额外正则。

**目的**：
确认你整个 pipeline（数据处理、VAE latent、Flow Matching）在 1–2 个数据集上能达到 **不比 CFM‑GP/scDFM 差** 的水平([OpenReview][6])。

---

**Step 2：引入 base + perturbation adapter（CFM‑VC 核心结构）**

* 把向量场替换为上面那种
  [
  v_\theta = v_{base}(h) + v_{eff}(h,\mathbf{p})
  ]
* 仍然只用 (\mathcal{L}*{FM})（(\lambda*{dist}=0)），保持训练稳定。

对比：

* CFM‑VC‑base vs CFM‑VC（base+adapter）
* 看 **unseen perturbation / combo / cross‑context** 三类任务的差别：

  * 如果 adapter 明显提升这些 OOD 任务，说明你的结构确实在用“基线+effect”分解做事，而不是摆设。

---

**Step 3：加入轻量的分布 matching（MMD/ED）**

* 打开 (\lambda_{dist})（例如 0.05–0.1），让 model 不止拟合端到端 velocity，也对生成 distribution 做校准。
* 重点看：

  * 分布级指标（ED/MMD / pathway FID）是否明显改善；
  * 有无 training unstable 迹象。
* 若不稳定，可以只在后期训练阶段加上（类似 fine‑tune），而不是从第一个 epoch 就开。

---

**Step 4：空间实验 & 微调（GNN / adversarial 作为 optional）**

* 在 ST 数据上：

  * 先用简单坐标 MLP 做 (\mathbf{s})，看虚拟 perturbation map 是否合理；
  * 再替换为 1–2 层 GNN，做 ablation：

    * CFM‑VC + coord vs CFM‑VC + GNN vs Celcomen / scVIVA / DestVI([arXiv][11])。
* 如果力有不逮，可以完全不做 adversarial，把“因果性”主要放在 Flow+结构上即可。

### 2.5 生物 case study：挑 1–2 个“你能讲出故事”的数据集

例如：

1. **免疫刺激 PBMC（IFN‑β 等）**

   * 展示 CFM‑VC 能在 latent 轨迹上区分 responder / non‑responder cell states，
   * 并且预测不同 dose / 时间点的虚拟 dose–response curve，和 CPA / CellFlow 比较。([bioRxiv][12])

2. **肿瘤 ST（如 glioblastoma / 肺癌）**

   * 参考 Celcomen 的设置：识别 intra‑/inter‑cellular 程序，做局部虚拟 perturbation（例如抑制某条 pathway），看是否削弱免疫抑制 micro‑environment([arXiv][11])。

生物学 story 只需要做对两件事：

* **你的虚拟轨迹 / 虚拟组织变化，与已知 pathway / cell–cell communication 逻辑一致**（可结合 GEARS graph / Spacia/GLACIER 等工具做解释）([Computer Science][16])；
* 提 1–2 个“模型预测的新组合 / 新空间靶点”作为 testable hypothesis，就足以打动生物审稿人。([ScienceDirect][14])

---

## 三、最后非常坦诚的 sanity check

**1. 这套设计是不是“为优化而优化”？**

* 没有：

  * VAE 部分基本就是 scVI/SpatialScope/ResolVI 这条成熟路线；([Project name not set][1])
  * Flow 部分严格跟着 Flow Matching 正统（Lipman 2023 + latent FM 的扩展），仅在结构上引入一个 adapter，这在 CellFlow/CFM‑GP 里都有类似思想([arXiv][5])；
  * 空间部分只把已有的 scVIVA / DestVI / GNN‑spatial 想法，塞进 context encoder，没有额外新 loss。([Project name not set][10])

**2. 会不会理论很美，结果很差/实现不了？**

* 实现层面：

  * 每一块都有对应的已发表模型作为“存在性证明”：scVI（latent）、CellFlow/CFM‑GP/scDFM（Flow Matching in single cell）、Celcomen/scVIVA/DestVI（空间 generative）([OpenReview][6])。
* 性能层面：

  * 最坏情况：CFM‑VC‑base ≈ CFM‑GP/scDFM 水平，你的 adapter 也许只带来温和提升；这在论文里可以如实写为“结构剖析表明某些数据集上效果有限”；
  * 最好情况：在 unseen perturbation / combo / cross‑cell‑type / 空间任务上明显优于 CPA/GEARS/scDFM/CFM‑GP，这就足够构成一区 paper 的“核心卖点”。

**3. 整体是否“融合、统一”，而不是东一块西一块？**

* 所有数据（scRNA 或 spatial）都走同一条路径：
  [
  X \xrightarrow{\text{NB-VAE}} z_{\text{int}} \xrightarrow{\text{FlowMatch}(p,c,s)} z_1 \xrightarrow{\text{decoder}} \hat{X}
  ]
* 差异只在 context encoder 里（是否带 (\mathbf{s})），
* 向量场形式始终是 “base + adapter”，
* 损失始终是 “VAE + Flow（+ 小分布项）”。

> 在我看来，这已经是一个“各部分互相咬合、没有明显拼凑感”的 CFM‑VC 2.x 设计了。



[1]: https://docs.scvi-tools.org/en/stable/tutorials/index_spatial.html?utm_source=chatgpt.com "Spatial transcriptomics — scvi-tools - Python"
[2]: https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1?utm_source=chatgpt.com "CellFlow enables generative single-cell phenotype modeling ... - bioRxiv"
[3]: https://suyccc.github.io/data/CellFlow.pdf?utm_source=chatgpt.com "CellFlow: Simulating Cellular Morphology Changes via Flow Matching"
[4]: https://cpa-tools.readthedocs.io/en/latest/index.html?utm_source=chatgpt.com "CPA - Compositional Perturbation Autoencoder — cpa-tools"
[5]: https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "[2210.02747] Flow Matching for Generative Modeling - arXiv.org"
[6]: https://openreview.net/forum?id=QSGanMEcUV&utm_source=chatgpt.com "scDFM: Distributional Flow Matching Model for Robust Single-Cell ..."
[7]: https://github.com/theislab/CPA?utm_source=chatgpt.com "CPA - Compositional Perturbation Autoencoder - GitHub"
[8]: https://arxiv.org/abs/2508.08312?utm_source=chatgpt.com "CFM-GP: Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types"
[9]: https://www.sciencedirect.com/science/article/pii/S2001037023004701?utm_source=chatgpt.com "A comprehensive overview of graph neural network-based approaches to ..."
[10]: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/scVIVA_tutorial.html?utm_source=chatgpt.com "scVIVA for representing cells and their environment in spatial ..."
[11]: https://arxiv.org/abs/2409.05804?utm_source=chatgpt.com "Celcomen: spatial causal disentanglement for single-cell and tissue ..."
[12]: https://www.biorxiv.org/content/10.1101/2021.04.14.439903v1?utm_source=chatgpt.com "Compositional perturbation autoencoder for single-cell response ..."
[13]: https://github.com/snap-stanford/GEARS?utm_source=chatgpt.com "GitHub - snap-stanford/GEARS: GEARS is a geometric deep learning model ..."
[14]: https://www.sciencedirect.com/science/article/pii/S2001037024001417?utm_source=chatgpt.com "A mini-review on perturbation modelling across single-cell omic ..."
[15]: https://www.nature.com/articles/s41592-025-02855-4.pdf?utm_source=chatgpt.com "STORIES: learning cell fate landscapes from spatial transcriptomics ..."
[16]: https://cs.stanford.edu/people/jure/pubs/gears-natbio23.pdf?utm_source=chatgpt.com "Predicting transcriptional outcomes of novel multigene perturbations ..."
