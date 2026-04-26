"""Unit tests for Positional Encoding variants (Idea 5).

Tests all PE implementations: sinusoidal, SPE, LFF-v2, Hybrid, Adaptive,
Progressive Frequency, and Per-Axis Frequency.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from networks.deep_sdf_decoder import (
    Decoder,
    PositionalEncoding,
    LearnableFourierFeaturesV2,
    HybridPositionalEncoding,
    AdaptiveFrequencyPositionalEncoding,
    ProgressiveFrequencyPositionalEncoding,
    PerAxisFrequencyPositionalEncoding,
    RandomFourierFeatures,
    LearnableFourierFeatures,
    SplinePositionalEncoding,
)


@pytest.fixture
def sinusoidal_pe():
    return PositionalEncoding(num_freqs=8, input_dim=3)


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(100, 3)


class TestSinusoidalPE:

    def test_output_dim(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        assert pe.output_dim == 39  # 3 + 3*2*6

    def test_output_dim_8_freqs(self):
        pe = PositionalEncoding(num_freqs=8, input_dim=3)
        assert pe.output_dim == 51  # 3 + 3*2*8

    def test_freq_bands_values(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        expected = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        assert torch.allclose(pe.freq_bands, expected)

    def test_freq_bands_is_buffer(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        assert "freq_bands" in dict(pe.named_buffers())
        assert "freq_bands" not in dict(pe.named_parameters())

    def test_forward_shape(self, sinusoidal_pe, x):
        out = sinusoidal_pe(x)
        assert out.shape == (100, 51)

    def test_forward_starts_with_identity(self, sinusoidal_pe, x):
        out = sinusoidal_pe(x)
        assert torch.allclose(out[:, :3], x)

    def test_forward_sin_cos_pairs(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        x = torch.randn(5, 3)
        out = pe(x)
        assert torch.allclose(out[:, 3:6], torch.sin(1.0 * x))
        assert torch.allclose(out[:, 6:9], torch.cos(1.0 * x))

    def test_decoder_integration(self):
        dec = Decoder(latent_size=8, dims=[32, 32],
                      positional_encoding=True, positional_encoding_num_freqs=6).eval()
        x = torch.randn(4, 11)  # latent(8) + xyz(3)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_decoder_different_freqs(self):
        dec = Decoder(latent_size=8, dims=[32, 32],
                      positional_encoding=True, positional_encoding_num_freqs=4).eval()
        assert dec.pos_enc.output_dim == 27  # 3 + 3*2*4
        x = torch.randn(4, 11)
        assert dec(x).shape == (4, 1)


class TestLFFv2:

    def test_matches_sinusoidal_at_init(self, x):
        pe_sin = PositionalEncoding(num_freqs=8)
        pe_lff = LearnableFourierFeaturesV2(num_freqs=8)
        assert torch.allclose(pe_sin(x), pe_lff(x), atol=1e-6)

    def test_output_dim(self):
        pe = LearnableFourierFeaturesV2(num_freqs=8)
        assert pe.output_dim == 51

    def test_has_learnable_params(self):
        pe = LearnableFourierFeaturesV2(num_freqs=8)
        assert sum(p.numel() for p in pe.parameters()) == 8

    def test_gradients_flow(self, x):
        pe = LearnableFourierFeaturesV2(num_freqs=8)
        out = pe(x)
        out.sum().backward()
        assert pe.W_r.grad is not None
        assert pe.W_r.grad.norm() > 0


class TestHybridPE:

    def test_matches_sinusoidal_at_init(self, x):
        pe_sin = PositionalEncoding(num_freqs=8)
        pe_hyb = HybridPositionalEncoding(num_freqs=8)
        assert torch.allclose(pe_sin(x), pe_hyb(x), atol=1e-6)

    def test_output_dim(self):
        pe = HybridPositionalEncoding(num_freqs=8)
        assert pe.output_dim == 51

    def test_residual_starts_zero(self):
        pe = HybridPositionalEncoding(num_freqs=8)
        assert torch.allclose(pe.residual.weight, torch.zeros_like(pe.residual.weight))
        assert torch.allclose(pe.residual.bias, torch.zeros_like(pe.residual.bias))

    def test_gradients_flow(self, x):
        pe = HybridPositionalEncoding(num_freqs=8)
        out = pe(x)
        out.sum().backward()
        assert pe.residual.weight.grad is not None


class TestAdaptiveFreqPE:

    def test_matches_sinusoidal_at_init(self, x):
        pe_sin = PositionalEncoding(num_freqs=8)
        pe_ada = AdaptiveFrequencyPositionalEncoding(num_freqs=8)
        assert torch.allclose(pe_sin(x), pe_ada(x), atol=1e-6)

    def test_frequencies_stay_positive(self):
        pe = AdaptiveFrequencyPositionalEncoding(num_freqs=8)
        freqs = torch.exp(pe.log_freqs)
        assert (freqs > 0).all()

    def test_gradients_flow(self, x):
        pe = AdaptiveFrequencyPositionalEncoding(num_freqs=8)
        out = pe(x)
        out.sum().backward()
        assert pe.log_freqs.grad is not None


class TestProgFreqPE:

    def test_matches_sinusoidal_at_epoch_100(self, x):
        pe_sin = PositionalEncoding(num_freqs=8)
        pe_prog = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        pe_prog.set_current_epoch(100)
        assert torch.allclose(pe_sin(x), pe_prog(x), atol=1e-6)

    def test_inactive_freqs_are_zero_at_epoch_1(self, x):
        pe = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        pe.set_current_epoch(1)  # L=3 active
        out = pe(x)
        # Inactive bands start at column 3 + 3*2*3 = 21
        inactive = out[:, 21:]
        assert inactive.abs().max() == 0.0

    def test_curriculum_schedule(self):
        pe = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        schedule = [(1, 3), (16, 4), (31, 5), (46, 6), (61, 7), (76, 8)]
        for epoch, expected_active in schedule:
            pe.set_current_epoch(epoch)
            assert pe.active_freqs.item() == expected_active

    def test_state_persists_in_checkpoint(self):
        pe = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        pe.set_current_epoch(100)
        state = pe.state_dict()
        pe2 = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        pe2.load_state_dict(state)
        assert pe2.active_freqs.item() == 8

    def test_no_learnable_params(self):
        pe = ProgressiveFrequencyPositionalEncoding(num_freqs=8)
        assert sum(p.numel() for p in pe.parameters()) == 0


class TestPerAxisPE:

    def test_matches_sinusoidal_at_init(self, x):
        pe_sin = PositionalEncoding(num_freqs=8)
        pe_ax = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        assert torch.allclose(pe_sin(x), pe_ax(x), atol=1e-6)

    def test_output_dim(self):
        pe = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        assert pe.output_dim == 51

    def test_params_per_axis(self):
        pe = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        assert pe.log_freqs.shape == (3, 8)  # 3 axes × 8 freqs

    def test_axes_independent(self):
        pe = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        pe.log_freqs.data[0, :] += 1.0  # perturb x-axis only
        x = torch.randn(10, 3)
        out = pe(x)
        # Verify output changed (axes are now different)
        pe_ref = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        out_ref = pe_ref(x)
        assert not torch.allclose(out, out_ref)

    def test_gradients_flow(self, x):
        pe = PerAxisFrequencyPositionalEncoding(num_freqs=8)
        out = pe(x)
        out.sum().backward()
        assert pe.log_freqs.grad is not None


class TestRandomFourierFeatures:

    @pytest.fixture
    def rff(self):
        torch.manual_seed(42)
        return RandomFourierFeatures(num_frequencies=64, scale=10.0, input_dim=3)

    def test_output_dim(self, rff):
        assert rff.output_dim == 2 * 64  # 128

    def test_output_shape(self, rff):
        x = torch.randn(10, 3)
        out = rff(x)
        assert out.shape == (10, 128)

    def test_B_is_buffer(self, rff):
        assert "B" in dict(rff.named_buffers())
        assert "B" not in dict(rff.named_parameters())

    def test_B_shape(self, rff):
        assert rff.B.shape == (64, 3)

    def test_no_learnable_params(self, rff):
        assert sum(p.numel() for p in rff.parameters()) == 0

    def test_B_scale(self):
        torch.manual_seed(42)
        rff = RandomFourierFeatures(num_frequencies=64, scale=4.0, input_dim=3)
        expected_std = 4.0 * 2.0 * torch.pi  # ~25.13 (2π matching Tancik et al. 2020)
        actual_std = rff.B.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.20

    def test_no_identity(self, rff):
        x = torch.randn(10, 3)
        out = rff(x)
        assert not torch.allclose(out[:, :3], x)

    def test_sin_cos_split(self, rff):
        x = torch.randn(10, 3)
        out = rff(x)
        proj = x @ rff.B.T
        assert torch.allclose(out[:, :64], torch.sin(proj), atol=1e-6)
        assert torch.allclose(out[:, 64:], torch.cos(proj), atol=1e-6)

    def test_deterministic(self, rff):
        x = torch.randn(10, 3)
        out1 = rff(x)
        out2 = rff(x)
        assert torch.allclose(out1, out2)

    def test_different_instances(self):
        torch.manual_seed(0)
        rff1 = RandomFourierFeatures(num_frequencies=64, scale=10.0, input_dim=3)
        torch.manual_seed(99)
        rff2 = RandomFourierFeatures(num_frequencies=64, scale=10.0, input_dim=3)
        assert not torch.allclose(rff1.B, rff2.B)

    def test_decoder_integration(self):
        torch.manual_seed(42)
        dec = Decoder(
            latent_size=256,
            dims=[512, 512, 512, 512],
            positional_encoding=True,
            positional_encoding_type="rff",
            rff_num_frequencies=64,
            rff_scale=10.0,
        ).eval()
        x = torch.randn(4, 259)  # latent(256) + xyz(3)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_output_range(self, rff):
        x = torch.randn(100, 3)
        out = rff(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestLearnableFourierFeatures:

    @pytest.fixture
    def lff(self):
        torch.manual_seed(42)
        return LearnableFourierFeatures(
            fourier_dim=128, hidden_dim=32, output_dim=128, gamma=10.0, input_dim=3
        )

    def test_output_dim(self, lff):
        assert lff.output_dim == 128

    def test_output_shape(self, lff):
        x = torch.randn(10, 3)
        out = lff(x)
        assert out.shape == (10, 128)

    def test_W_r_is_parameter(self, lff):
        # W_r is now nn.Linear (matching reference), so params are under "W_r.weight"
        assert "W_r.weight" in dict(lff.named_parameters())

    def test_W_r_shape(self, lff):
        # Reference: W_r projects to fourier_dim//2 = 64, shape (64, 3)
        assert lff.W_r.weight.shape == (64, 3)

    def test_W_r_init_scale(self):
        torch.manual_seed(42)
        lff = LearnableFourierFeatures(
            fourier_dim=128, hidden_dim=32, output_dim=128, gamma=10.0, input_dim=3
        )
        expected_std = 10.0 ** -2  # 0.01
        actual_std = lff.W_r.weight.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.50

    def test_mlp_structure(self, lff):
        import torch.nn as nn
        assert isinstance(lff.mlp[0], nn.Linear)
        assert isinstance(lff.mlp[1], nn.GELU)
        assert isinstance(lff.mlp[2], nn.Linear)

    def test_gradients_flow(self, lff):
        x = torch.randn(10, 3)
        out = lff(x)
        out.sum().backward()
        assert lff.W_r.weight.grad is not None
        assert lff.mlp[0].weight.grad is not None

    def test_scale_factor(self, lff):
        # Reference: 1/sqrt(F_dim) where F_dim=128 (post-cat dimension)
        expected_scale = 1.0 / (128 ** 0.5)
        assert abs(lff.scale - expected_scale) < 1e-9

    def test_decoder_integration(self):
        torch.manual_seed(42)
        dec = Decoder(
            latent_size=256,
            dims=[512, 512, 512, 512],
            positional_encoding=True,
            positional_encoding_type="lff",
            lff_fourier_dim=128,
            lff_hidden_dim=32,
            lff_output_dim=128,
        ).eval()
        x = torch.randn(4, 259)  # latent(256) + xyz(3)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_fourier_feature_structure(self, lff):
        x = torch.randn(5, 3)
        out = lff(x)
        assert out.shape == (5, 128)  # output_dim=128 matching reference


class TestSplinePositionalEncoding:

    @pytest.fixture
    def spe(self):
        torch.manual_seed(42)
        return SplinePositionalEncoding(code_num=64, code_channel=64, input_dim=3)

    def test_output_dim(self, spe):
        assert spe.output_dim == 64

    def test_output_shape(self, spe):
        x = torch.randn(10, 3).clamp(-1, 1)
        out = spe(x)
        assert out.shape == (10, 64)

    def test_codes_are_parameters(self, spe):
        import torch.nn as nn
        for code in spe.codes:
            assert isinstance(code, nn.Parameter)

    def test_codes_shape(self, spe):
        for code in spe.codes:
            assert code.shape == (64, 64)

    def test_num_code_tables(self, spe):
        assert len(spe.codes) == 3

    def test_partition_of_unity(self):
        frac_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        for f in frac_values:
            w0 = 0.5 * (1.0 - f) ** 2
            w1 = 0.5 + f * (1.0 - f)
            w2 = 0.5 * f ** 2
            assert abs((w0 + w1 + w2).item() - 1.0) < 1e-6

    def test_progressive_resolution(self, spe):
        x = torch.randn(10, 3).clamp(-1, 1)
        out_full = spe(x).detach().clone()
        spe.set_effective_resolution(16)
        out_low = spe(x).detach().clone()
        assert not torch.allclose(out_full, out_low)

    def test_resolution_clamped(self, spe):
        spe.set_effective_resolution(9999)
        assert spe.effective_code_num == spe.code_num

    def test_boundary_handling(self, spe):
        x = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        out = spe(x)
        assert out.shape == (2, 64)
        assert torch.isfinite(out).all()

    def test_interpolation_continuity(self, spe):
        x = torch.tensor([[0.0, 0.0, 0.0]])
        eps = 1e-4
        x_eps = torch.tensor([[eps, eps, eps]])
        out = spe(x)
        out_eps = spe(x_eps)
        diff = (out - out_eps).abs().max().item()
        assert diff < 0.1  # outputs should be close for nearby inputs

    def test_per_dimension_codes(self, spe):
        x = torch.randn(10, 3).clamp(-1, 1)
        # Baseline output
        out_base = spe(x).detach().clone()

        # Perturb codes[0] with a unique pattern
        original_codes0 = spe.codes[0].data.clone()
        spe.codes[0].data += torch.linspace(0, 5, spe.codes[0].numel()).reshape_as(spe.codes[0].data)
        out_perturbed0 = spe(x).detach().clone()
        assert not torch.allclose(out_base, out_perturbed0)

        # Restore codes[0], perturb codes[1] with a different pattern
        spe.codes[0].data = original_codes0
        original_codes1 = spe.codes[1].data.clone()
        spe.codes[1].data += torch.linspace(5, 0, spe.codes[1].numel()).reshape_as(spe.codes[1].data)
        out_perturbed1 = spe(x).detach().clone()
        assert not torch.allclose(out_base, out_perturbed1)
        assert not torch.allclose(out_perturbed0, out_perturbed1)

        # Restore codes[1]
        spe.codes[1].data = original_codes1

    def test_gradients_flow(self, spe):
        x = torch.randn(10, 3).clamp(-1, 1)
        out = spe(x)
        out.sum().backward()
        for d in range(3):
            assert spe.codes[d].grad is not None

    def test_decoder_integration(self):
        torch.manual_seed(42)
        dec = Decoder(
            latent_size=256,
            dims=[512, 512, 512, 512],
            positional_encoding=True,
            positional_encoding_type="spline",
            spline_code_num=64,
            spline_code_channel=64,
        ).eval()
        x = torch.randn(4, 259)  # latent(256) + xyz(3)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_effective_resolution_not_in_state_dict(self, spe):
        sd = spe.state_dict()
        assert "effective_code_num" not in sd
