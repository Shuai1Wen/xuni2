"""
SingleCellDataset：将AnnData对象转换为PyTorch Dataset

完全按照details.md 1.2节的伪代码实现：
- 从AnnData.layers["counts"]读取原始计数
- 扰动标签转为one-hot向量
- 协变量（batch、cell_type）转为索引
- 空间坐标直接返回或None
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any


class SingleCellDataset(Dataset):
    """
    单细胞扰动数据集
    
    参数
    ----
    adata : anndata.AnnData
        输入的单细胞数据对象。必须包含以下内容：
        - layers["counts"]：原始计数矩阵 (n_obs, n_genes)
        - obs["perturbation"]：扰动标签（categorical）
        - obs["batch"]：批次标签（categorical）
        - obs["cell_type"]：细胞类型标签（categorical）
        - obsm["spatial"]（可选）：空间坐标 (n_obs, n_spatial_dims)
    
    gene_key : str，default="counts"
        AnnData.layers中计数矩阵的键名
    
    device : str，default="cpu"
        计算设备（这里仅用于文档，实际数据转移在dataloader中进行）
    """
    
    def __init__(
        self,
        adata,
        gene_key: str = "counts",
        device: str = "cpu"
    ):
        self.adata = adata
        self.device = device
        
        # 获取原始计数矩阵 (n_obs, n_genes)
        if gene_key not in adata.layers:
            raise ValueError(
                f"gene_key='{gene_key}'在adata.layers中不存在。"
                f"可用的层: {list(adata.layers.keys())}"
            )
        
        # 转为numpy数组，确保数据类型一致性
        self.X = np.asarray(adata.layers[gene_key], dtype=np.float32)
        
        # 处理扰动标签
        if "perturbation" not in adata.obs:
            raise ValueError("obs中必须包含'perturbation'列")
        
        pert_categories = adata.obs["perturbation"].astype("category")
        self.pert_codes = pert_categories.cat.codes.values  # [0..K-1]
        self.num_perts = len(pert_categories.cat.categories)
        self.pert_categories = pert_categories.cat.categories.tolist()
        
        # 处理批次标签
        if "batch" not in adata.obs:
            raise ValueError("obs中必须包含'batch'列")
        
        batch_categories = adata.obs["batch"].astype("category")
        self.batch_codes = batch_categories.cat.codes.values
        self.num_batches = len(batch_categories.cat.categories)
        self.batch_categories = batch_categories.cat.categories.tolist()
        
        # 处理细胞类型标签
        if "cell_type" not in adata.obs:
            raise ValueError("obs中必须包含'cell_type'列")
        
        ct_categories = adata.obs["cell_type"].astype("category")
        self.ct_codes = ct_categories.cat.codes.values
        self.num_cell_types = len(ct_categories.cat.categories)
        self.ct_categories = ct_categories.cat.categories.tolist()
        
        # 处理空间坐标（可选）
        if "spatial" in adata.obsm:
            self.spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
            self.spatial_dim = self.spatial.shape[1]
        else:
            self.spatial = None
            self.spatial_dim = None
        
        self.n_obs = self.X.shape[0]
        self.n_genes = self.X.shape[1]
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        关键点：返回numpy数组或标量，让DataLoader进行批处理和转移到GPU
        这样做的优点：
        - 内存高效（不在dataset中就转换为tensor）
        - 灵活性好（支持pin_memory加速）
        
        返回
        ----
        dict，包含键：
        - "x"：原始计数向量 (n_genes,) - numpy float32
        - "p"：one-hot扰动向量 (n_perturb,) - numpy float32
        - "batch"：批次索引标量
        - "cell_type"：细胞类型索引标量
        - "spatial"：空间坐标 (n_spatial,) 或 None
        """
        
        # 计数数据（numpy数组，避免频繁的torch转换）
        x_counts = self.X[idx]  # (n_genes,) float32
        
        # 扰动one-hot向量（numpy数组）
        p_idx = self.pert_codes[idx]
        p = np.zeros(self.num_perts, dtype=np.float32)
        p[p_idx] = 1.0
        
        # 批次索引
        batch = int(self.batch_codes[idx])
        
        # 细胞类型索引
        ct = int(self.ct_codes[idx])
        
        # 空间坐标
        if self.spatial is not None:
            spatial = self.spatial[idx]  # (n_spatial,) float32
        else:
            spatial = None
        
        return {
            "x": x_counts,
            "p": p,
            "batch": batch,
            "cell_type": ct,
            "spatial": spatial,
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        返回数据集元数据
        
        返回
        ----
        dict，包含：
        - n_genes：基因数
        - n_perturbs：扰动类别数
        - n_batches：批次数
        - n_cell_types：细胞类型数
        - spatial_dim：空间维度或None
        - pert_categories：扰动标签列表
        - batch_categories：批次标签列表
        - ct_categories：细胞类型标签列表
        """
        return {
            "n_genes": self.n_genes,
            "n_perturbs": self.num_perts,
            "n_batches": self.num_batches,
            "n_cell_types": self.num_cell_types,
            "spatial_dim": self.spatial_dim,
            "pert_categories": self.pert_categories,
            "batch_categories": self.batch_categories,
            "ct_categories": self.ct_categories,
        }


# ============ DataLoader的正确用法 ============

def collate_fn_cfm(batch_list):
    """
    自定义collate函数，确保数据转换和转移的正确性
    
    这个函数在DataLoader中使用，确保：
    1. numpy数组转为torch tensor
    2. 列表正确stack为batch
    3. None值正确处理
    
    使用方式：
        train_loader = DataLoader(
            dataset, 
            batch_size=64, 
            collate_fn=collate_fn_cfm
        )
    """
    batch_dict = {}
    
    # 转换x (计数数据)
    batch_dict["x"] = torch.from_numpy(
        np.stack([item["x"] for item in batch_list], axis=0)
    ).float()  # (B, n_genes)
    
    # 转换p (one-hot扰动)
    batch_dict["p"] = torch.from_numpy(
        np.stack([item["p"] for item in batch_list], axis=0)
    ).float()  # (B, n_perts)
    
    # 转换batch indices
    batch_dict["batch"] = torch.tensor(
        [item["batch"] for item in batch_list],
        dtype=torch.long
    )  # (B,)
    
    # 转换cell_type indices
    batch_dict["cell_type"] = torch.tensor(
        [item["cell_type"] for item in batch_list],
        dtype=torch.long
    )  # (B,)
    
    # 转换spatial（如果存在）
    spatial_list = [item["spatial"] for item in batch_list]
    if spatial_list[0] is not None:
        batch_dict["spatial"] = torch.from_numpy(
            np.stack(spatial_list, axis=0)
        ).float()  # (B, spatial_dim)
    else:
        batch_dict["spatial"] = None
    
    return batch_dict
