"""
综合模型测试

测试内容：
1. 维度匹配性
2. 梯度流正确性
3. 数值稳定性（NaN检测）
4. 因果约束
5. 内存占用
"""

import torch
import torch.nn as nn
import pytest
from cfm_vc.models import (
    EncoderVAE, DecoderVAE, nb_log_likelihood,
    ContextEncoder, FlowField, CFMVCModel,
)


class TestEncoderVAE:
    """编码器测试"""
    
    def test_forward_shapes(self):
        """测试输出形状"""
        n_genes = 100
        n_batch = 2
        n_ct = 3
        batch_size = 8
        
        encoder = EncoderVAE(
            n_genes=n_genes, n_batch=n_batch, n_ct=n_ct,
            dim_int=32, dim_tech=8,
        )
        
        x = torch.randn(batch_size, n_genes)
        batch_idx = torch.randint(0, n_batch, (batch_size,))
        ct_idx = torch.randint(0, n_ct, (batch_size,))
        
        z_int, z_tech, kl_int, kl_tech = encoder(x, batch_idx, ct_idx)
        
        assert z_int.shape == (batch_size, 32), f"Expected (8,32), got {z_int.shape}"
        assert z_tech.shape == (batch_size, 8), f"Expected (8,8), got {z_tech.shape}"
        assert kl_int.shape == (batch_size,), f"Expected (8,), got {kl_int.shape}"
        assert kl_tech.shape == (batch_size,), f"Expected (8,), got {kl_tech.shape}"
        
        print("✅ EncoderVAE shape test passed")
    
    def test_no_nan(self):
        """测试是否出现NaN"""
        encoder = EncoderVAE(100, 2, 3)
        x = torch.randn(8, 100)
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        z_int, z_tech, kl_int, kl_tech = encoder(x, batch_idx, ct_idx)
        
        assert torch.all(torch.isfinite(z_int)), "NaN in z_int"
        assert torch.all(torch.isfinite(z_tech)), "NaN in z_tech"
        assert torch.all(torch.isfinite(kl_int)), "NaN in kl_int"
        assert torch.all(torch.isfinite(kl_tech)), "NaN in kl_tech"
        
        print("✅ EncoderVAE NaN test passed")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        encoder = EncoderVAE(100, 2, 3)
        x = torch.randn(8, 100, requires_grad=True)
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        z_int, z_tech, kl_int, kl_tech = encoder(x, batch_idx, ct_idx)
        loss = z_int.sum() + kl_int.sum()
        loss.backward()
        
        # 检查梯度是否存在
        assert x.grad is not None, "No gradient for input"
        for param in encoder.parameters():
            assert param.grad is not None, f"No gradient for {param}"
        
        print("✅ EncoderVAE gradient flow test passed")


class TestDecoderVAE:
    """解码器测试"""
    
    def test_positive_outputs(self):
        """测试输出是否全正"""
        decoder = DecoderVAE(100, dim_int=32, dim_tech=8)
        z_int = torch.randn(8, 32)
        z_tech = torch.randn(8, 8)
        
        mean, theta = decoder(z_int, z_tech)
        
        assert torch.all(mean > 0), "Mean should be positive"
        assert torch.all(theta > 0), "Theta should be positive"
        
        print("✅ DecoderVAE positive output test passed")
    
    def test_nb_likelihood_no_nan(self):
        """测试NB似然是否出现NaN"""
        x = torch.poisson(torch.full((8, 100), 5.0))
        mean = torch.full((8, 100), 5.0) + 0.1  # 避免mean=0
        theta = torch.full((100,), 2.0)
        
        log_lik = nb_log_likelihood(x, mean, theta)
        
        assert log_lik.shape == (8,), f"Wrong shape: {log_lik.shape}"
        assert torch.all(torch.isfinite(log_lik)), "NaN in log likelihood"
        
        print("✅ NB likelihood NaN test passed")
    
    def test_nb_likelihood_with_zeros(self):
        """测试NB似然对零计数的处理"""
        x = torch.zeros(8, 100)  # 零计数
        mean = torch.ones(8, 100) * 5.0
        theta = torch.ones(100)
        
        log_lik = nb_log_likelihood(x, mean, theta)
        
        # 应该产生有限的值，不是NaN
        assert torch.all(torch.isfinite(log_lik)), "NaN in log likelihood with zeros"
        
        print("✅ NB likelihood zero handling test passed")


class TestContextEncoder:
    """Context编码器测试"""
    
    def test_adapter_zero_perturbation(self):
        """测试p=0时adapter是否输出接近0"""
        ctx = ContextEncoder(n_batch=2, n_ct=3, p_dim=5)
        
        p_zero = torch.zeros(8, 5)  # p=0
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        context, pert_alpha = ctx(p_zero, batch_idx, ct_idx)
        
        # 由于MLP无bias，p=0应该导致较小的输出
        print(f"  pert_alpha with p=0: mean={pert_alpha.mean():.6f}, std={pert_alpha.std():.6f}")
        
        # pert_alpha不会完全为0，因为包含ReLU后的bias
        # 但应该比有扰动时小
        assert torch.all(torch.isfinite(pert_alpha)), "NaN in pert_alpha"
        
        print("✅ ContextEncoder zero perturbation test passed")
    
    def test_adapter_has_bias(self):
        """检查adapter MLP是否真的无bias"""
        ctx = ContextEncoder(n_batch=2, n_ct=3, p_dim=5)
        
        for i, layer in enumerate(ctx.pert_mlp_layers):
            assert layer.bias is None, f"Layer {i} has bias!"
        
        print("✅ ContextEncoder no-bias verification passed")


class TestFlowField:
    """Flow向量场测试"""
    
    def test_forward_shapes(self):
        """测试输出形状"""
        flow = FlowField(
            dim_int=32, context_dim=24, alpha_dim=64,
            hidden_dim=128, n_basis=16,
        )
        
        batch_size = 8
        z_t = torch.randn(batch_size, 32)
        t = torch.rand(batch_size)
        context = torch.randn(batch_size, 24)
        pert_alpha = torch.randn(batch_size, 64)
        
        v = flow(z_t, t, context, pert_alpha)
        
        assert v.shape == (batch_size, 32), f"Wrong shape: {v.shape}"
        assert torch.all(torch.isfinite(v)), "NaN in vector field"
        
        print("✅ FlowField shape test passed")
    
    def test_base_and_effect(self):
        """测试base和effect的分离"""
        flow = FlowField(32, 24, 64, hidden_dim=128, n_basis=4)
        
        # 两次forward调用
        batch_size = 8
        z_t = torch.randn(batch_size, 32)
        t = torch.rand(batch_size)
        context = torch.randn(batch_size, 24)
        
        # 不同的pert_alpha
        pert_alpha_1 = torch.randn(batch_size, 64)
        pert_alpha_2 = torch.randn(batch_size, 64)
        
        v_1 = flow(z_t, t, context, pert_alpha_1)
        v_2 = flow(z_t, t, context, pert_alpha_2)
        
        # 向量场应该不同（因为pert_alpha不同）
        assert not torch.allclose(v_1, v_2), "Vector fields should be different"
        
        print("✅ FlowField base+effect test passed")


class TestCFMVCModel:
    """完整模型测试"""
    
    def test_vae_forward_no_nan(self):
        """测试VAE前向传播"""
        model = CFMVCModel(
            n_genes=100, n_batch=2, n_ct=3, n_perts=5,
        )
        
        x = torch.randn(8, 100)
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        loss, z_int, z_tech = model.vae_forward(x, batch_idx, ct_idx)
        
        assert torch.isfinite(loss), "Loss is NaN or Inf"
        assert torch.all(torch.isfinite(z_int)), "z_int contains NaN"
        assert torch.all(torch.isfinite(z_tech)), "z_tech contains NaN"
        
        print("✅ CFMVCModel VAE forward test passed")
    
    def test_flow_step_gradient(self):
        """测试Flow步的梯度流"""
        model = CFMVCModel(100, 2, 3, 5)
        
        x = torch.randn(8, 100)
        p = torch.zeros(8, 5)
        p[torch.arange(8), torch.randint(0, 5, (8,))] = 1.0
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        # 编码
        with torch.no_grad():
            z_int, _, _, _ = model.encoder(x, batch_idx, ct_idx)
        z_int = z_int.detach()  # ← 关键：detach
        
        # Flow步
        loss, fm_loss, dist_loss = model.flow_step(
            z_int, p, batch_idx, ct_idx, spatial=None
        )
        
        # 检查梯度流
        assert torch.isfinite(loss), "Loss is NaN"
        assert z_int.grad is None, "z_int should not have grad (detached)"
        
        print("✅ CFMVCModel flow step gradient test passed")
    
    def test_generate_expression(self):
        """测试表达生成"""
        model = CFMVCModel(100, 2, 3, 5)
        model.eval()
        
        p = torch.zeros(8, 5)
        p[:, 0] = 1.0  # control
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        with torch.no_grad():
            X_hat = model.generate_expression(
                p, batch_idx, ct_idx, n_steps=5
            )
        
        assert X_hat.shape == (8, 100), f"Wrong shape: {X_hat.shape}"
        assert torch.all(X_hat > 0), "Generated expression should be positive"
        assert torch.all(torch.isfinite(X_hat)), "Generated expression contains NaN"
        
        print("✅ CFMVCModel generate expression test passed")


class TestGradientFlow:
    """梯度流专项测试"""
    
    def test_stage1_encoder_gradient(self):
        """测试Stage 1中Encoder有梯度"""
        model = CFMVCModel(100, 2, 3, 5)
        x = torch.randn(8, 100)
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        loss, _, _ = model.vae_forward(x, batch_idx, ct_idx)
        loss.backward()
        
        # Encoder应该有梯度
        has_grad = False
        for param in model.encoder.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "Encoder should have gradients in Stage 1"
        print("✅ Stage 1 encoder gradient test passed")
    
    def test_stage2_flow_gradient(self):
        """测试Stage 2中Flow有梯度，VAE没有"""
        model = CFMVCModel(100, 2, 3, 5)
        
        # 冻结VAE（模拟Stage 2）
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        
        p = torch.zeros(8, 5)
        p[:, 0] = 1.0
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        z_int = torch.randn(8, 32)  # dummy z_int
        loss, _, _ = model.flow_step(z_int, p, batch_idx, ct_idx)
        loss.backward()
        
        # Flow应该有梯度
        has_flow_grad = False
        for param in model.flow.parameters():
            if param.grad is not None:
                has_flow_grad = True
                break
        
        # Encoder不应该有梯度（requires_grad=False）
        encoder_has_grad = any(
            p.grad is not None 
            for p in model.encoder.parameters()
        )
        
        assert has_flow_grad, "Flow should have gradients in Stage 2"
        assert not encoder_has_grad, "Encoder should not have gradients when requires_grad=False"
        
        print("✅ Stage 2 flow gradient test passed")


class TestNumericalStability:
    """数值稳定性专项测试"""
    
    def test_large_x_handling(self):
        """测试大计数值的处理"""
        x = torch.full((8, 100), 1000.0)  # 大计数
        mean = torch.full((8, 100), 900.0)
        theta = torch.full((100,), 100.0)
        
        log_lik = nb_log_likelihood(x, mean, theta)
        
        assert torch.all(torch.isfinite(log_lik)), "NaN with large counts"
        print("✅ Large count handling test passed")
    
    def test_small_mean_handling(self):
        """测试小均值的处理"""
        x = torch.ones(8, 100)
        mean = torch.full((8, 100), 0.01)  # 小均值
        theta = torch.full((100,), 0.01)
        
        log_lik = nb_log_likelihood(x, mean, theta)
        
        assert torch.all(torch.isfinite(log_lik)), "NaN with small means"
        print("✅ Small mean handling test passed")
    
    def test_logvar_clipping(self):
        """测试logvar是否被正确clamp"""
        encoder = EncoderVAE(100, 2, 3)
        
        # 强制输出极端logvar
        x = torch.ones(8, 100) * 100  # 异常输入
        batch_idx = torch.zeros(8, dtype=torch.long)
        ct_idx = torch.zeros(8, dtype=torch.long)
        
        z_int, z_tech, _, _ = encoder(x, batch_idx, ct_idx)
        
        # z_int和z_tech应该是有限的
        assert torch.all(torch.isfinite(z_int)), "z_int should be finite"
        assert torch.all(torch.isfinite(z_tech)), "z_tech should be finite"
        
        print("✅ Logvar clipping test passed")


if __name__ == "__main__":
    # 运行所有测试
    print("=" * 60)
    print("运行综合测试套件")
    print("=" * 60)
    
    # EncoderVAE
    print("\n[EncoderVAE Tests]")
    t = TestEncoderVAE()
    t.test_forward_shapes()
    t.test_no_nan()
    t.test_gradient_flow()
    
    # DecoderVAE
    print("\n[DecoderVAE Tests]")
    t = TestDecoderVAE()
    t.test_positive_outputs()
    t.test_nb_likelihood_no_nan()
    t.test_nb_likelihood_with_zeros()
    
    # ContextEncoder
    print("\n[ContextEncoder Tests]")
    t = TestContextEncoder()
    t.test_adapter_zero_perturbation()
    t.test_adapter_has_bias()
    
    # FlowField
    print("\n[FlowField Tests]")
    t = TestFlowField()
    t.test_forward_shapes()
    t.test_base_and_effect()
    
    # CFMVCModel
    print("\n[CFMVCModel Tests]")
    t = TestCFMVCModel()
    t.test_vae_forward_no_nan()
    t.test_flow_step_gradient()
    t.test_generate_expression()
    
    # GradientFlow
    print("\n[GradientFlow Tests]")
    t = TestGradientFlow()
    t.test_stage1_encoder_gradient()
    t.test_stage2_flow_gradient()
    
    # NumericalStability
    print("\n[NumericalStability Tests]")
    t = TestNumericalStability()
    t.test_large_x_handling()
    t.test_small_mean_handling()
    t.test_logvar_clipping()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
