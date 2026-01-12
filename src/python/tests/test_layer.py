import torch
import pytest

torch.manual_seed(42)


def _naive_mhc_dynamic(
    x,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    b_pre,
    b_post,
    b_res,
    sinkhorn_iters,
    eps,
):
    B, n, C = x.shape
    x_flat = x.reshape(B, n * C)
    rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + eps)
    x_norm = x_flat / rms

    p_pre = x_norm @ phi_pre.t()
    p_post = x_norm @ phi_post.t()
    p_res = x_norm @ phi_res.t()

    tilde_pre = alpha_pre * p_pre + b_pre
    tilde_post = alpha_post * p_post + b_post
    tilde_res = alpha_res * p_res + b_res.reshape(1, n * n)

    H_pre = torch.sigmoid(tilde_pre)
    H_post = 2.0 * torch.sigmoid(tilde_post)

    M = torch.exp(tilde_res).view(B, n, n)
    for _ in range(sinkhorn_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)

    x_agg = torch.einsum("bi,bic->bc", H_pre, x)
    rms_agg = torch.sqrt(torch.mean(x_agg * x_agg, dim=-1, keepdim=True) + eps)
    y_norm = x_agg / rms_agg * rmsnorm_weight.view(1, C)

    y_dist = H_post.unsqueeze(-1) * y_norm.unsqueeze(1)
    x_mixed = torch.einsum("bij,bjc->bic", M, x)

    return x_mixed + y_dist


def test_layer_forward_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    x = torch.randn(B, n, C, device="cuda")
    out = layer(x)

    assert out.shape == (B, n, C)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_forward_dynamic():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    x = torch.randn(B, n, C, device="cuda")
    out = layer(x)

    assert out.shape == (B, n, C)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_backward_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


def test_layer_backward_dynamic():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    assert layer.rmsnorm_weight.grad is not None
    assert not torch.isnan(layer.rmsnorm_weight.grad).any()


def test_layer_backward_dynamic_params():
    from mhc import MHCLayer

    B, n, C = 4, 4, 64
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)
    out.sum().backward()

    for name in [
        "phi_pre",
        "phi_post",
        "phi_res",
        "b_pre",
        "b_post",
        "b_res",
        "alpha_pre",
        "alpha_post",
        "alpha_res",
    ]:
        param = getattr(layer, name)
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is zero"


def test_layer_dynamic_matches_naive_small():
    from mhc import MHCLayer

    B, n, C = 2, 4, 16
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, sinkhorn_iters=3, use_dynamic_h=True).cuda()

    x = torch.randn(B, n, C, device="cuda", requires_grad=True)
    out_fused = layer(x)
    out_fused.sum().backward()

    fused_x_grad = x.grad.detach().clone()
    fused_phi_res_grad = layer.phi_res.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    phi_pre = layer.phi_pre.detach().clone().requires_grad_(True)
    phi_post = layer.phi_post.detach().clone().requires_grad_(True)
    phi_res = layer.phi_res.detach().clone().requires_grad_(True)
    b_pre = layer.b_pre.detach().clone().requires_grad_(True)
    b_post = layer.b_post.detach().clone().requires_grad_(True)
    b_res = layer.b_res.detach().clone().requires_grad_(True)
    alpha_pre = layer.alpha_pre.detach().clone().requires_grad_(True)
    alpha_post = layer.alpha_post.detach().clone().requires_grad_(True)
    alpha_res = layer.alpha_res.detach().clone().requires_grad_(True)
    rmsnorm_weight = layer.rmsnorm_weight.detach().float().clone().requires_grad_(True)

    out_ref = _naive_mhc_dynamic(
        x_ref,
        rmsnorm_weight,
        phi_pre,
        phi_post,
        phi_res,
        alpha_pre,
        alpha_post,
        alpha_res,
        b_pre,
        b_post,
        b_res,
        layer.sinkhorn_iters,
        layer.eps,
    )
    out_ref.sum().backward()

    torch.testing.assert_close(out_fused.detach(), out_ref.detach(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(fused_x_grad, x_ref.grad, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(
        fused_phi_res_grad, phi_res.grad, rtol=1e-1, atol=1e-1
    )


def test_layer_parameters_static():
    from mhc import MHCLayer

    C, n = 128, 4
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    params = dict(layer.named_parameters())
    assert "rmsnorm_weight" in params
    assert "H_pre" in params
    assert "H_post" in params
    assert "H_res" in params

    assert params["rmsnorm_weight"].shape == (C,)
    assert params["H_pre"].shape == (n,)
    assert params["H_post"].shape == (n,)
    assert params["H_res"].shape == (n, n)


def test_layer_parameters_dynamic():
    from mhc import MHCLayer

    C, n = 128, 4
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    params = dict(layer.named_parameters())
    assert "rmsnorm_weight" in params
    assert "H_pre" not in params
    assert "H_post" not in params
    assert "H_res" not in params

    assert params["rmsnorm_weight"].shape == (C,)


def test_layer_training_step_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    x = torch.randn(B, n, C, device="cuda")
    target = torch.randn(B, n, C, device="cuda")

    for _ in range(3):
        optimizer.zero_grad()
        out = layer(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    assert not torch.isnan(loss)


def test_layer_different_batch_sizes_static():
    from mhc import MHCLayer

    n, C = 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    for B in [1, 4, 16, 32]:
        x = torch.randn(B, n, C, device="cuda")
        out = layer(x)
        assert out.shape == (B, n, C)


def test_layer_different_batch_sizes_dynamic():
    from mhc import MHCLayer

    n, C = 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    for B in [1, 4, 16, 32]:
        x = torch.randn(B, n, C, device="cuda")
        out = layer(x)
        assert out.shape == (B, n, C)


def test_layer_numerical_stability_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    x_large = torch.randn(B, n, C, device="cuda") * 100
    out = layer(x_large)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    x_small = torch.randn(B, n, C, device="cuda") * 0.001
    out = layer(x_small)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_deterministic_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    x = torch.randn(B, n, C, device="cuda")
    out1 = layer(x)
    out2 = layer(x)

    assert torch.allclose(out1, out2)


def test_layer_deterministic_dynamic():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=True).cuda()

    x = torch.randn(B, n, C, device="cuda")

    out1 = layer(x)
    out2 = layer(x)

    assert torch.allclose(out1, out2)


def test_layer_gradient_flow_static():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n, use_dynamic_h=False).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    assert x.grad is not None
    grad_norm = x.grad.norm()
    assert grad_norm > 0, "Gradient should flow through"
    assert grad_norm < 1e6, "Gradient should not explode"


def test_default_is_dynamic():
    from mhc import MHCLayer

    layer = MHCLayer(hidden_dim=128, expansion_rate=4)
    assert layer.use_dynamic_h is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
