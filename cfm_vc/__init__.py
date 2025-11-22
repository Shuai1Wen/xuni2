"""
CFM-VC 2.x：用Flow Matching学习单细胞扰动响应的生成模型

本实现严格遵循claude.md、details.md、model.md的设计规范：
- 完整实现，无MVP或占位符
- 结构约束优先于Loss堆砌
- 生态复用（scVI风格NB-VAE + Flow Matching标准）
- 统一的scRNA+spatial处理管道

关键特点：
1. 数值稳定性优先：所有log/exp操作都带eps防护
2. 梯度流清晰：显式的detach和requires_grad管理
3. 内存优化：及时释放中间张量，无不必要复制
4. 因果约束硬编码：adapter无bias确保p=0→α=0
"""

__version__ = "2.0.1-optimized"
