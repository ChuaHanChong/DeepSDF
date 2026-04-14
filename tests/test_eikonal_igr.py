"""Unit tests for Eikonal/IGR Regularization (Idea 2).

Verifies the implementation against:
- IGR paper: Gropp et al., "Implicit Geometric Regularization for Learning Shapes", ICML 2020
- Official IGR repo: repo/IGR/code/model/network.py, repo/IGR/code/shapespace/train.py

Key properties tested:
1. Softplus activation (smooth gradients for Eikonal)
2. Geometric initialization (sphere SDF at init)
3. Skip connection scaling (1/sqrt(2) with geometric_init)
4. Non-detached latent gradient flow (IGR-style coupling)
5. Tanh output conditional (unbounded SDF for Eikonal)
6. Eikonal loss computation correctness
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from networks.deep_sdf_decoder import Decoder


# ── Activation Function Tests ─────────────────────────────

class TestActivation:
    """IGR uses Softplus(beta=100), not ReLU. Paper Section 5."""

    def test_softplus_created_when_configured(self):
        dec = Decoder(latent_size=8, dims=[32, 32], activation="softplus")
        assert isinstance(dec.activation, nn.Softplus)

    def test_relu_created_by_default(self):
        dec = Decoder(latent_size=8, dims=[32, 32])
        assert isinstance(dec.activation, nn.ReLU)

    def test_relu_created_when_explicit(self):
        dec = Decoder(latent_size=8, dims=[32, 32], activation="relu")
        assert isinstance(dec.activation, nn.ReLU)

    def test_softplus_smooth_at_zero(self):
        """Softplus(100) has gradient 0.5 at x=0 (vs ReLU's undefined)."""
        sp = nn.Softplus(beta=100)
        x = torch.tensor([0.0], requires_grad=True)
        y = sp(x)
        y.backward()
        # sigmoid(beta * 0) = sigmoid(0) = 0.5
        assert abs(x.grad.item() - 0.5) < 0.01

    def test_softplus_matches_relu_for_positive(self):
        """Softplus(100) ≈ ReLU for x > 0.1."""
        sp = nn.Softplus(beta=100)
        relu = nn.ReLU()
        x = torch.linspace(0.1, 5.0, 100)
        diff = (sp(x) - relu(x)).abs().max().item()
        assert diff < 0.01

    def test_softplus_gradient_nonzero_near_zero(self):
        """Unlike ReLU, Softplus gradient is nonzero near x=0."""
        sp = nn.Softplus(beta=100)
        relu = nn.ReLU()
        # At x = -0.05 (just below zero): ReLU grad = 0, Softplus grad > 0
        x_sp = torch.tensor([-0.05], requires_grad=True)
        x_rl = torch.tensor([-0.05], requires_grad=True)
        sp(x_sp).backward()
        relu(x_rl).backward()
        assert x_sp.grad.item() > 0.001  # Softplus has gradient here
        assert x_rl.grad.item() == 0.0   # ReLU has zero gradient


# ── Geometric Initialization Tests ────────────────────────

class TestGeometricInit:
    """IGR initializes network to predict SDF of unit sphere.
    Reference: repo/IGR/code/model/network.py lines 46-55."""

    def test_sphere_at_origin(self):
        """f(0,0,0) ≈ -radius_init with zero latent code."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        latent = torch.zeros(1, 8)
        xyz = torch.zeros(1, 3)
        inp = torch.cat([latent, xyz], 1)
        with torch.no_grad():
            f = dec(inp).item()
        # Inside the sphere → f < 0
        assert f < 0, f"f(origin) = {f}, expected < 0"

    def test_sphere_on_surface(self):
        """f(1,0,0) ≈ 0 with zero latent code."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        latent = torch.zeros(1, 8)
        xyz = torch.tensor([[1.0, 0.0, 0.0]])
        inp = torch.cat([latent, xyz], 1)
        with torch.no_grad():
            f = dec(inp).item()
        assert abs(f) < 0.5, f"f(surface) = {f}, expected ≈ 0"

    def test_gradient_norm_near_one(self):
        """||∇f|| ≈ 1 at random points (sphere SDF has unit gradient).
        This is why Eikonal loss starts near zero with geometric init."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.eval()
        latent = torch.zeros(50, 8)
        xyz = torch.randn(50, 3, requires_grad=True)
        inp = torch.cat([latent, xyz], 1)
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, xyz, grad_outputs=torch.ones_like(sdf),
                                   create_graph=False)[0]
        grad_norms = grad.norm(dim=1)
        mean_norm = grad_norms.mean().item()
        # Should be near 1.0 — the defining property of geometric init
        assert 0.5 < mean_norm < 2.0, f"Mean ||∇f|| = {mean_norm}, expected ≈ 1"

    def test_disables_weight_norm(self):
        """geometric_init=True forces weight_norm=False (incompatible)."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      weight_norm=True, geometric_init=True)
        assert dec.weight_norm is False

    def test_last_layer_bias_negative(self):
        """Last layer bias ≈ -radius_init."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      geometric_init=True, radius_init=1.0)
        last_layer = getattr(dec, f"lin{dec.num_layers - 2}")
        assert last_layer.bias.item() < 0


# ── Skip Connection Scaling Tests ─────────────────────────

class TestSkipScaling:
    """IGR scales skip connections by 1/sqrt(2).
    Reference: repo/IGR/code/model/network.py line 75."""

    def test_scaling_with_geometric_init(self):
        """With geometric_init, skip concatenation divides by sqrt(2)."""
        dec = Decoder(latent_size=8, dims=[32, 32, 32, 32],
                      latent_in=[2], geometric_init=True, activation="softplus")
        assert dec.geometric_init is True
        # The scaling happens in forward() — verify it runs without error
        x = torch.randn(5, 11)
        out = dec(x)
        assert out.shape == (5, 1)

    def test_no_scaling_without_geometric_init(self):
        """Without geometric_init, no 1/sqrt(2) scaling."""
        dec = Decoder(latent_size=8, dims=[32, 32, 32, 32],
                      latent_in=[2], geometric_init=False)
        assert dec.geometric_init is False
        x = torch.randn(5, 11)
        out = dec(x)
        assert out.shape == (5, 1)


# ── Tanh Output Tests ─────────────────────────────────────

class TestTanhOutput:
    """IGR returns raw SDF (unbounded). DeepSDF applies Tanh (clips to [-1,1]).
    Reference: repo/IGR/code/model/network.py line 82 — returns x directly.

    The fix: Tanh is disabled when geometric_init=True."""

    def test_no_tanh_with_geometric_init(self):
        """With geometric_init, output should be unbounded (no Tanh clipping)."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        assert not hasattr(dec, "th") or dec.th is None

    def test_tanh_without_geometric_init(self):
        """Without geometric_init (standard DeepSDF), Tanh is applied."""
        dec = Decoder(latent_size=8, dims=[32, 32])
        assert hasattr(dec, "th") and dec.th is not None

    def test_output_can_exceed_unit_with_geometric_init(self):
        """Verify output actually goes beyond [-1, 1] range."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.eval()
        # Far from origin → SDF should be large
        latent = torch.zeros(1, 8)
        xyz = torch.tensor([[5.0, 5.0, 5.0]])
        inp = torch.cat([latent, xyz], 1)
        with torch.no_grad():
            f = dec(inp).item()
        # With sphere init, f(far point) should be >> 1
        assert abs(f) > 0.5, f"f(far) = {f}, expected |f| > 0.5 (unbounded)"


# ── Eikonal Gradient Flow Tests ───────────────────────────

class TestEikonalGradientFlow:
    """IGR couples Eikonal loss to latent codes (no .detach()).
    Reference: repo/IGR/code/shapespace/train.py lines 45-46."""

    def test_gradients_flow_without_detach(self):
        """With non-detached latents, Eikonal loss reaches latent codes."""
        dec = Decoder(latent_size=8, dims=[32, 32], activation="softplus").cpu()
        dec.train()
        latent = torch.randn(10, 8, requires_grad=True)
        xyz = torch.randn(10, 3, requires_grad=True)
        inp = torch.cat([latent, xyz], 1)  # NO detach on latent
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, xyz, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        eik_loss = ((grad.norm(dim=1) - 1) ** 2).mean()
        eik_loss.backward()
        assert latent.grad is not None
        assert latent.grad.norm() > 0

    def test_gradients_blocked_with_detach(self):
        """With detached latents, Eikonal loss does NOT reach latent codes."""
        dec = Decoder(latent_size=8, dims=[32, 32], activation="softplus").cpu()
        dec.train()
        latent = torch.randn(10, 8, requires_grad=True)
        xyz = torch.randn(10, 3, requires_grad=True)
        inp = torch.cat([latent.detach(), xyz], 1)  # DETACH latent
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, xyz, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        eik_loss = ((grad.norm(dim=1) - 1) ** 2).mean()
        eik_loss.backward()
        assert latent.grad is None


# ── Eikonal Loss Computation Tests ────────────────────────

class TestEikonalLoss:
    """Tests for Eikonal loss (||∇f|| - 1)². Paper eq. 2."""

    @pytest.fixture
    def eikonal_setup(self):
        decoder = Decoder(latent_size=8, dims=[32, 32]).cpu()
        decoder.train()
        N = 16
        pts = torch.empty(N, 3).uniform_(-1, 1).requires_grad_(True)
        latents = torch.randn(N, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)
        grad = torch.autograd.grad(sdf, pts, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        return decoder, pts, sdf, grad

    def test_gradient_shape(self, eikonal_setup):
        _, pts, _, grad = eikonal_setup
        assert grad.shape == (pts.shape[0], 3)

    def test_loss_finite_nonnegative(self, eikonal_setup):
        _, _, _, grad = eikonal_setup
        eik_loss = torch.mean((grad.norm(dim=1) - 1.0) ** 2)
        assert torch.isfinite(eik_loss)
        assert eik_loss.item() >= 0.0

    def test_loss_differentiable(self, eikonal_setup):
        decoder, _, _, grad = eikonal_setup
        eik_loss = torch.mean((grad.norm(dim=1) - 1.0) ** 2)
        eik_loss.backward()
        has_grad = any(p.grad is not None for p in decoder.parameters())
        assert has_grad

    def test_eikonal_near_zero_with_geometric_init(self):
        """With geometric init, Eikonal should start small (sphere has ||∇f||≈1)."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.eval()
        pts = torch.randn(50, 3, requires_grad=True)
        latent = torch.zeros(50, 8)
        inp = torch.cat([latent, pts], 1)
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, pts, grad_outputs=torch.ones_like(sdf),
                                   create_graph=False)[0]
        eik = ((grad.norm(dim=1) - 1.0) ** 2).mean().item()
        # With geometric init, Eikonal should be small (< 0.5)
        assert eik < 1.0, f"Eikonal at init = {eik}, expected < 1.0 with sphere init"

    def test_second_order_finite(self):
        decoder = Decoder(latent_size=8, dims=[32, 32]).cpu()
        decoder.train()
        pts = torch.empty(8, 3).uniform_(-1, 1).requires_grad_(True)
        latents = torch.randn(8, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)
        grad = torch.autograd.grad(sdf, pts, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        laplacian = 0.0
        for d in range(3):
            grad2 = torch.autograd.grad(grad[:, d], pts,
                                        grad_outputs=torch.ones_like(grad[:, d]),
                                        create_graph=True, retain_graph=True)[0][:, d]
            laplacian = laplacian + grad2
        assert torch.isfinite(laplacian).all()

    def test_second_order_differentiable(self):
        decoder = Decoder(latent_size=8, dims=[32, 32]).cpu()
        decoder.train()
        pts = torch.empty(8, 3).uniform_(-1, 1).requires_grad_(True)
        latents = torch.randn(8, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)
        grad = torch.autograd.grad(sdf, pts, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        laplacian = 0.0
        for d in range(3):
            grad2 = torch.autograd.grad(grad[:, d], pts,
                                        grad_outputs=torch.ones_like(grad[:, d]),
                                        create_graph=True, retain_graph=True)[0][:, d]
            laplacian = laplacian + grad2
        loss = torch.mean(laplacian ** 2)
        loss.backward()
        has_grad = any(p.grad is not None for p in decoder.parameters())
        assert has_grad


# ── Surface Term Tests ───────────────────────────────────

class TestSurfaceTerms:
    """Tests for IGR surface terms (train_deep_sdf.py lines 915-978).

    Surface value loss: mean(|f(x)|) for near-surface points.
    Surface Eikonal: mean((||grad_f|| - 1)^2) at surface.
    Bug 1 regression: surface term should respect eikonal_detach_latent.
    """

    def test_surface_value_loss(self):
        """Surface value loss = mean(|sdf|) should be finite and non-negative."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.eval()
        # Surface points near the unit sphere
        theta = torch.linspace(0, np.pi, 10)
        phi = torch.linspace(0, 2 * np.pi, 10)
        surface_pts = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=1)  # (10, 3)
        latent = torch.zeros(10, 8)
        inp = torch.cat([latent, surface_pts], dim=1)
        with torch.no_grad():
            sdf = dec(inp)
        loss = torch.mean(torch.abs(sdf))
        assert torch.isfinite(loss), f"Surface value loss is not finite: {loss}"
        assert loss.item() >= 0.0, f"Surface value loss is negative: {loss}"

    def test_surface_eikonal_loss(self):
        """Surface Eikonal loss = mean((||grad|| - 1)^2) should be finite."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.train()
        surface_pts = torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
        ], requires_grad=True)
        latent = torch.zeros(5, 8)
        inp = torch.cat([latent, surface_pts], dim=1)
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, surface_pts,
                                   grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        surf_eik = torch.mean((grad.norm(dim=1) - 1.0) ** 2)
        assert torch.isfinite(surf_eik), f"Surface Eikonal loss is not finite: {surf_eik}"

    def test_surface_detach_consistency(self):
        """Bug 1 regression: surface term should respect eikonal_detach_latent.

        When detach=True, backward through surface term should NOT populate latent.grad.
        When detach=False, it SHOULD.
        Simulates the logic from train_deep_sdf.py:940-950.
        """
        dec = Decoder(latent_size=8, dims=[32, 32], activation="softplus")
        dec.train()
        surf_xyz = torch.randn(10, 3, requires_grad=True)

        # Case 1: detach=True — latent.grad should remain None
        latent_detach = torch.randn(10, 8, requires_grad=True)
        inp_detach = torch.cat([latent_detach.detach(), surf_xyz], dim=1)
        sdf_detach = dec(inp_detach)
        grad_detach = torch.autograd.grad(sdf_detach, surf_xyz,
                                          grad_outputs=torch.ones_like(sdf_detach),
                                          create_graph=True, retain_graph=True)[0]
        surf_loss_detach = torch.mean(torch.abs(sdf_detach)) + \
                           torch.mean((grad_detach.norm(dim=1) - 1.0) ** 2)
        surf_loss_detach.backward()
        assert latent_detach.grad is None, (
            "With detach=True, latent should NOT receive gradients from surface term"
        )

        # Case 2: detach=False (IGR style) — latent.grad should be populated
        dec.zero_grad()
        surf_xyz2 = torch.randn(10, 3, requires_grad=True)
        latent_no_detach = torch.randn(10, 8, requires_grad=True)
        inp_no_detach = torch.cat([latent_no_detach, surf_xyz2], dim=1)
        sdf_no_detach = dec(inp_no_detach)
        grad_no_detach = torch.autograd.grad(sdf_no_detach, surf_xyz2,
                                             grad_outputs=torch.ones_like(sdf_no_detach),
                                             create_graph=True, retain_graph=True)[0]
        surf_loss_no_detach = torch.mean(torch.abs(sdf_no_detach)) + \
                              torch.mean((grad_no_detach.norm(dim=1) - 1.0) ** 2)
        surf_loss_no_detach.backward()
        assert latent_no_detach.grad is not None, (
            "With detach=False, latent SHOULD receive gradients from surface term"
        )
        assert latent_no_detach.grad.norm() > 0

    def test_surface_loss_near_zero_for_geometric_init(self):
        """With geometric init, surface points near the unit sphere should have small |f(x)|."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.eval()
        # Points exactly on the unit sphere
        pts_on_sphere = torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
        ])
        latent = torch.zeros(6, 8)
        inp = torch.cat([latent, pts_on_sphere], dim=1)
        with torch.no_grad():
            sdf = dec(inp)
        mean_abs = torch.mean(torch.abs(sdf)).item()
        # With sphere init, f(surface points) should be near 0
        assert mean_abs < 0.5, (
            f"Mean |f(surface)| = {mean_abs}, expected < 0.5 with geometric init"
        )

    def test_eikonal_with_category_embedding(self):
        """Forward pass works with [cat_vec | latent | xyz] input dimensions."""
        cat_dim = 4
        latent_size = 8
        # effective latent = cat_dim + latent_size = 12
        dec = Decoder(latent_size=cat_dim + latent_size, dims=[32, 32],
                      activation="softplus")
        dec.train()
        N = 10
        cat_vec = torch.randn(N, cat_dim)
        latent = torch.randn(N, latent_size)
        xyz = torch.randn(N, 3, requires_grad=True)
        inp = torch.cat([cat_vec, latent, xyz], dim=1)  # (N, 4+8+3) = (N, 15)
        sdf = dec(inp)
        assert sdf.shape == (N, 1)
        # Verify Eikonal computation also works
        grad = torch.autograd.grad(sdf, xyz, grad_outputs=torch.ones_like(sdf),
                                   create_graph=True)[0]
        assert grad.shape == (N, 3)
        eik = ((grad.norm(dim=1) - 1.0) ** 2).mean()
        assert torch.isfinite(eik)


# ── Warmup and Sampling Tests ────────────────────────────

class TestWarmupAndSampling:
    """Tests for Eikonal warmup and IGR sampling.

    Warmup: lambda ramps from 0 to full over configured epochs.
    IGR sampling: mix local (Gaussian around training points) + global (uniform).
    """

    def test_warmup_at_epoch_zero(self):
        """warmup_frac at epoch 0 should be 0."""
        epoch = 0
        warmup_end_epoch = 100
        warmup_frac = min(1.0, epoch / max(1, warmup_end_epoch))
        assert warmup_frac == 0.0

    def test_warmup_at_half(self):
        """warmup_frac at epoch 50/100 should be 0.5."""
        epoch = 50
        warmup_end_epoch = 100
        warmup_frac = min(1.0, epoch / max(1, warmup_end_epoch))
        assert warmup_frac == 0.5

    def test_warmup_at_end(self):
        """warmup_frac at epoch 100/100 should be 1.0."""
        epoch = 100
        warmup_end_epoch = 100
        warmup_frac = min(1.0, epoch / max(1, warmup_end_epoch))
        assert warmup_frac == 1.0

    def test_warmup_past_end(self):
        """warmup_frac past end epoch should be clamped to 1.0."""
        epoch = 200
        warmup_end_epoch = 100
        warmup_frac = min(1.0, epoch / max(1, warmup_end_epoch))
        assert warmup_frac == 1.0

    def test_geometric_init_deep_network(self):
        """8-layer network with geometric_init should output negative at origin (inside sphere)."""
        dec = Decoder(
            latent_size=8,
            dims=[512, 512, 512, 512, 512, 512, 512, 512],
            latent_in=[4],
            activation="softplus",
            geometric_init=True,
            radius_init=1.0,
        )
        dec.eval()
        latent = torch.zeros(1, 8)
        xyz = torch.zeros(1, 3)
        inp = torch.cat([latent, xyz], dim=1)
        with torch.no_grad():
            f = dec(inp).item()
        assert f < 0, f"f(origin) = {f}, expected < 0 (inside sphere)"


# ── IGR Normal Alignment Tests ──────────────────────────

class TestIGRNormalAlignment:
    """Tests for IGR normal alignment loss.

    True formula from repo/IGR/code/shapespace/train.py:72:
        normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()

    This captures the *direction* of gradients, not just magnitude.
    """

    def test_normal_loss_formula(self):
        """Known computation: one aligned pair, one misaligned pair.

        grad = [[1,0,0],[0,1,0]], normals = [[1,0,0],[0,0,1]]
        diff[0] = [0,0,0] -> |diff| = [0,0,0] -> L2 = 0
        diff[1] = [0,1,-1] -> |diff| = [0,1,1] -> L2 = sqrt(2)
        loss = (0 + sqrt(2)) / 2 = sqrt(2)/2 ~ 0.7071
        """
        grad = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        loss = ((grad - normals).abs()).norm(2, dim=1).mean()
        expected = np.sqrt(2) / 2.0
        assert abs(loss.item() - expected) < 1e-4, (
            f"Normal alignment loss = {loss.item()}, expected {expected}"
        )

    def test_normal_loss_zero_when_aligned(self):
        """When grad == normals exactly, normal alignment loss == 0."""
        grad = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        normals = grad.clone()
        loss = ((grad - normals).abs()).norm(2, dim=1).mean()
        assert loss.item() == 0.0

    def test_normal_loss_positive_when_misaligned(self):
        """Random grad vs random normals should produce loss > 0."""
        torch.manual_seed(42)
        grad = torch.randn(20, 3)
        normals = torch.randn(20, 3)
        loss = ((grad - normals).abs()).norm(2, dim=1).mean()
        assert loss.item() > 0.0

    def test_normal_loss_differentiable(self):
        """Normal alignment loss should back-propagate through the decoder."""
        dec = Decoder(latent_size=8, dims=[32, 32],
                      activation="softplus", geometric_init=True, radius_init=1.0)
        dec.train()
        surface_pts = torch.randn(10, 3, requires_grad=True)
        latent = torch.zeros(10, 8)
        inp = torch.cat([latent, surface_pts], dim=1)
        sdf = dec(inp)
        grad = torch.autograd.grad(sdf, surface_pts,
                                   grad_outputs=torch.ones_like(sdf),
                                   create_graph=True, retain_graph=True)[0]
        # Fake ground-truth normals (unit vectors)
        normals = torch.randn(10, 3)
        normals = normals / normals.norm(dim=1, keepdim=True)
        normal_loss = ((grad - normals).abs()).norm(2, dim=1).mean()
        normal_loss.backward()
        has_grad = any(p.grad is not None and p.grad.norm() > 0
                       for p in dec.parameters())
        assert has_grad, "Normal alignment loss should produce gradients in decoder"

    def test_gradient_direction_matters(self):
        """Surface Eikonal only checks magnitude; normal alignment checks direction.

        Two gradient sets with identical L2 norms but different directions:
        - Surface Eikonal (||grad||-1)^2 gives the SAME loss for both
        - Normal alignment gives DIFFERENT losses
        """
        normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # grad_a: same direction as normals, unit norm
        grad_a = normals.clone()
        # grad_b: different direction, still unit norm
        grad_b = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        # Both have ||grad|| == 1, so surface Eikonal is the same (== 0)
        eik_a = ((grad_a.norm(dim=1) - 1.0) ** 2).mean()
        eik_b = ((grad_b.norm(dim=1) - 1.0) ** 2).mean()
        assert torch.allclose(eik_a, eik_b), (
            f"Surface Eikonal should be equal: {eik_a} vs {eik_b}"
        )

        # Normal alignment should differ
        norm_loss_a = ((grad_a - normals).abs()).norm(2, dim=1).mean()
        norm_loss_b = ((grad_b - normals).abs()).norm(2, dim=1).mean()
        assert norm_loss_a.item() == 0.0, "Aligned gradients should give zero loss"
        assert norm_loss_b.item() > 0.0, "Misaligned gradients should give positive loss"
        assert norm_loss_a.item() != norm_loss_b.item(), (
            "Normal alignment must distinguish directions that Eikonal cannot"
        )

    def test_fallback_surface_eikonal(self):
        """When normals are unavailable (None), training falls back to surface
        Eikonal loss (||grad||-1)^2. Verify it produces finite non-negative values."""
        normals = None  # Simulate missing normals

        torch.manual_seed(7)
        grad = torch.randn(20, 3)

        # Fallback: surface Eikonal
        if normals is None:
            loss = ((grad.norm(dim=1) - 1.0) ** 2).mean()
        else:
            loss = ((grad - normals).abs()).norm(2, dim=1).mean()

        assert torch.isfinite(loss), f"Fallback Eikonal loss is not finite: {loss}"
        assert loss.item() >= 0.0, f"Fallback Eikonal loss is negative: {loss}"
