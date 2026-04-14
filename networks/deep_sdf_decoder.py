#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Idea 5 — Positional Encoding
#
# Maps low-dimensional xyz coordinates into a higher-dimensional space using
# sinusoidal functions at exponentially increasing frequencies. This helps the
# network learn high-frequency surface details that a plain MLP struggles with.
#
# Formula:
#   gamma(p) = [p, sin(2^0*p), cos(2^0*p), ..., sin(2^(L-1)*p), cos(2^(L-1)*p)]
#
# Tensor shapes (with num_freqs=6, input_dim=3):
#   input:  (N, 3)   — raw xyz coordinates
#   output: (N, 39)  — encoded coordinates  [3 + 3*2*6 = 39]
#
# The output is structured as blocks of 3 columns each:
#   cols  0- 2: identity (raw xyz)
#   cols  3- 5: sin(1*xyz)   i.e. sin(2^0 * xyz)
#   cols  6- 8: cos(1*xyz)
#   cols  9-11: sin(2*xyz)   i.e. sin(2^1 * xyz)
#   cols 12-14: cos(2*xyz)
#   ... and so on for frequencies 4, 8, 16, 32
#
# Demo:
#   pe = PositionalEncoding(num_freqs=6, input_dim=3)
#   x = torch.randn(100, 3)   # 100 points in 3D
#   y = pe(x)                 # (100, 39)
#   assert y[:, :3] == x      # first 3 cols are identity
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6, input_dim=3):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        # output_dim = input_dim + input_dim * 2 * num_freqs
        # e.g. 3 + 3*2*6 = 39 for the default configuration
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # freq_bands = [2^0, 2^1, ..., 2^(num_freqs-1)] = [1, 2, 4, 8, 16, 32]
        # Stored as a buffer (not a parameter) so it moves with .cuda() but is not learned
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        # x: (N, 3)  — raw xyz coordinates
        encoded = [x]  # Start with identity: (N, 3)
        for freq in self.freq_bands:
            # Each iteration adds 2 blocks of 3 columns: sin(freq*x) and cos(freq*x)
            encoded.append(torch.sin(freq * x))  # (N, 3)
            encoded.append(torch.cos(freq * x))  # (N, 3)
        # Concatenate along feature dim: (N, 3 + 3*2*num_freqs) = (N, 39)
        return torch.cat(encoded, dim=-1)


# =============================================================================
# Random Fourier Features (Tancik et al., 2020)
#
# Fixed random projection: gamma(x) = [sin(x @ B^T), cos(x @ B^T)]
# B ~ N(0, scale^2) is sampled once at init and never updated.
# =============================================================================
class RandomFourierFeatures(nn.Module):
    def __init__(self, num_frequencies=64, scale=10.0, input_dim=3):
        super().__init__()
        self.output_dim = 2 * num_frequencies
        # Paper: γ(v) = [cos(2πBv), sin(2πBv)], B ~ N(0, σ²I)
        # Absorb 2π into B so forward is just x @ B.T
        # Note: uses 2π scaling (matching Tancik et al., 2020 paper formula).
        # The SplinePosEnc reference uses π for [-1,1] inputs — see docs for discussion.
        B = torch.randn(num_frequencies, input_dim) * scale * 2.0 * np.pi
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B.T  # (N, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# =============================================================================
# Learnable Fourier Features (Li et al., 2021)
#
# Learnable projection W_r followed by MLP modulation.
# r_x = (1/sqrt(F)) * [cos(x @ W_r^T) || sin(x @ W_r^T)]
# output = MLP(r_x)
# =============================================================================
class LearnableFourierFeatures(nn.Module):
    def __init__(self, fourier_dim=128, hidden_dim=64, output_dim=64,
                 gamma=10.0, input_dim=3):
        super().__init__()
        self.output_dim = output_dim
        # Reference: 1/sqrt(F_dim) where F_dim is post-cat dim (2*fourier_dim)
        self.scale = 1.0 / ((2 * fourier_dim) ** 0.5)
        # Reference: nn.init.normal_(Wr.weight, mean=0, std=gamma^-2)
        self.W_r = nn.Parameter(torch.randn(fourier_dim, input_dim) * (gamma ** -2))
        self.mlp = nn.Sequential(
            nn.Linear(2 * fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        proj = x @ self.W_r.T  # (N, fourier_dim)
        r_x = self.scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.mlp(r_x)  # (N, output_dim)


# =============================================================================
# Spline Positional Encoding (Zheng et al., 2021)
#
# Quadratic B-spline interpolation over a grid of learnable codes.
# Per-dimension: look up 3 neighboring knot codes, weight by B-spline basis.
# Sum contributions across all input dimensions.
#
# Progressive training: uses effective_code_num that starts small and grows,
# so early training learns coarse structure, later training adds fine detail.
# Call set_effective_resolution(eff_num) from the training loop each epoch.
# =============================================================================
class SplinePositionalEncoding(nn.Module):
    def __init__(self, code_num=64, code_channel=64, input_dim=3):
        super().__init__()
        self.output_dim = code_channel
        self.code_num = code_num
        self.input_dim = input_dim
        self.effective_code_num = code_num  # starts at full, training loop can override
        # Learnable codes: one table per spatial dimension
        # Note: reference SplinePosEnc uses xavier_uniform_ init; we use small random
        # init (randn*0.01) for conservative start. Reference also uses 0.8 contraction
        # factor for [-1,1] inputs and round() for knot selection — we use floor() and
        # no contraction, relying on boundary clamping instead. See docs for discussion.
        self.codes = nn.ParameterList([
            nn.Parameter(torch.randn(code_num, code_channel) * 0.01)
            for _ in range(input_dim)
        ])

    def set_effective_resolution(self, eff_num):
        """Set effective resolution for progressive training (coarse-to-fine)."""
        self.effective_code_num = min(eff_num, self.code_num)

    def forward(self, x):
        # x: (N, 3), coordinates in [-1, 1]
        K = self.effective_code_num  # may be < code_num during progressive training
        result = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        for d in range(self.input_dim):
            coord = x[:, d]  # (N,)
            # Map [-1, 1] to [0, K-1] (effective resolution)
            t = (coord + 1.0) * ((K - 1) / 2.0)  # (N,)
            idx = t.long().clamp(0, K - 2)
            frac = t - idx.float()  # (N,) in [0, 1]
            # Quadratic B-spline: 3-point support
            idx0 = (idx - 1).clamp(0, K - 1)
            idx1 = idx.clamp(0, K - 1)
            idx2 = (idx + 1).clamp(0, K - 1)
            w0 = 0.5 * (1.0 - frac) ** 2
            w1 = 0.5 + frac * (1.0 - frac)
            w2 = 0.5 * frac ** 2
            c0 = self.codes[d][idx0]  # (N, C)
            c1 = self.codes[d][idx1]
            c2 = self.codes[d][idx2]
            result = result + w0.unsqueeze(1) * c0 + w1.unsqueeze(1) * c1 + w2.unsqueeze(1) * c2
        return result


# =============================================================================
# Experiment 1: LearnableFourierFeaturesV2 (LFF-v2)
#
# Fixes the prior LFF latent collapse by:
# 1. Initializing W_r as a sinusoidal frequency matrix (NOT random)
# 2. Removing the MLP layer that created gradient bottleneck
# 3. Output shape matches sinusoidal PE (identity + sin/cos blocks)
#
# At initialization, LFF-v2(x) produces output IDENTICAL to PositionalEncoding(x).
# The only learnable parameter is W_r (shape: num_freqs x input_dim), so the model
# starts at our proven best config and can only drift to learn per-dimension
# frequency adjustments.
#
# Tensor shapes (num_freqs=8, input_dim=3):
#   W_r:    (8, 3)    learnable, init = diag-like sinusoidal frequencies
#   x:      (N, 3)
#   proj:   (N, 8)  = x @ W_r.T
#   encoded: (N, 51) = cat([x, sin(proj_xyz_expanded), cos(proj_xyz_expanded)])
# =============================================================================
class LearnableFourierFeaturesV2(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        # Output matches sinusoidal PE: identity + num_freqs * 2 * input_dim
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # W_r is learnable but initialized as the sinusoidal frequency vector.
        # Shape: (num_freqs,) — one frequency per band, applied to all input dims.
        # This is the key insight: if W_r[k] = 2^k, then W_r[k] * x_d produces
        # identical outputs to the standard sinusoidal PE.
        init_freqs = 2.0 ** torch.arange(num_freqs).float()
        self.W_r = nn.Parameter(init_freqs.clone())

    def forward(self, x):
        # x: (N, 3)  — raw xyz coordinates
        encoded = [x]  # Identity component: (N, 3)
        for k in range(self.num_freqs):
            freq = self.W_r[k]
            # Apply the learnable frequency to all 3 input dims
            encoded.append(torch.sin(freq * x))  # (N, 3)
            encoded.append(torch.cos(freq * x))  # (N, 3)
        return torch.cat(encoded, dim=-1)  # (N, 3 + 3*2*num_freqs)


# =============================================================================
# Experiment 2: HybridPositionalEncoding
#
# Sinusoidal PE backbone + zero-initialized learnable residual.
# At initialization, the output is IDENTICAL to sinusoidal PE because the
# residual Linear layer has zero weights and zero bias.
#
# During training, the residual can learn to add small corrections to the
# encoded features if they help reconstruction. Since it starts at zero,
# this experiment is mathematically guaranteed to be non-worse than sinusoidal
# at initialization.
#
# Tensor shapes:
#   x:              (N, 3)
#   sinusoidal_out: (N, 3 + 3*2*num_freqs) = (N, 51) for num_freqs=8
#   residual:       (N, 3*2*num_freqs) = (N, 48) — residual for sin/cos bands only
#   output:         (N, 51) = sinusoidal_out with residual added to sin/cos part
# =============================================================================
class HybridPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # Fixed sinusoidal frequencies (buffer, not parameter)
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)

        # Zero-initialized learnable residual applied to sin/cos bands
        residual_dim = input_dim * 2 * num_freqs  # 48 for default
        self.residual = nn.Linear(input_dim, residual_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        # Standard sinusoidal PE
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        sinusoidal_out = torch.cat(encoded, dim=-1)  # (N, 51)

        # Learnable residual (starts at zero)
        residual_output = self.residual(x)  # (N, 48)

        # Add residual to sin/cos portion only (keep identity clean)
        # Concatenate: [identity (N, 3), sin/cos + residual (N, 48)]
        sin_cos_part = sinusoidal_out[:, self.input_dim:]  # (N, 48)
        corrected = sin_cos_part + residual_output  # (N, 48)
        return torch.cat([x, corrected], dim=-1)  # (N, 51)


# =============================================================================
# Experiment 3: AdaptiveFrequencyPositionalEncoding
#
# Makes the sinusoidal frequencies themselves learnable via log parameterization.
# log_freqs ensures frequencies stay positive and allows multiplicative updates.
# Initialized at standard dyadic frequencies [1, 2, 4, ..., 2^(num_freqs-1)].
#
# At initialization: identical to standard sinusoidal PE.
# During training: model can drift frequencies to find optimal values for the data.
# =============================================================================
class AdaptiveFrequencyPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # Learnable log-frequencies — initialized at standard dyadic frequencies
        init_freqs = 2.0 ** torch.arange(num_freqs).float()
        self.log_freqs = nn.Parameter(torch.log(init_freqs))

    def forward(self, x):
        # Compute positive frequencies from log parameterization
        freqs = torch.exp(self.log_freqs)  # (num_freqs,)

        encoded = [x]
        for k in range(self.num_freqs):
            encoded.append(torch.sin(freqs[k] * x))
            encoded.append(torch.cos(freqs[k] * x))
        return torch.cat(encoded, dim=-1)


# =============================================================================
# Experiment 4: ProgressiveFrequencyPositionalEncoding (ProgFreqPE)
#
# Curriculum learning: start with only low frequencies active, gradually unmask
# higher frequencies as training progresses. Tests the spectral bias hypothesis
# (Rahaman et al. 2019) — that networks learn low frequencies first.
#
# Active frequency schedule (for num_freqs=8):
#   Epochs 1-15:   3 freqs active ([1, 2, 4])
#   Epochs 16-30:  4 freqs active ([1, 2, 4, 8])
#   Epochs 31-45:  5 freqs ([1, 2, 4, 8, 16])
#   Epochs 46-60:  6 freqs ([1, 2, 4, 8, 16, 32])
#   Epochs 61-75:  7 freqs ([1, 2, 4, 8, 16, 32, 64])
#   Epochs 76-100: 8 freqs (full schedule)
#
# Inactive frequencies produce sin/cos outputs multiplied by 0 — the tensor
# shape stays constant at (N, output_dim) so the decoder input dim never changes
# mid-training. No new parameters are introduced at transitions.
#
# Requires training loop to call `decoder.set_current_epoch(epoch)` each epoch.
# =============================================================================
class ProgressiveFrequencyPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # Fixed sinusoidal frequencies
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)

        # Active frequency count, updated each epoch by training loop
        # Starts at 3 (low frequencies only)
        self.register_buffer("active_freqs", torch.tensor(3, dtype=torch.long))

    def set_current_epoch(self, epoch):
        """Called by training loop each epoch to update the frequency curriculum.
        Schedule (matches 100-epoch budget):
          ep 1-15:   L=3
          ep 16-30:  L=4
          ep 31-45:  L=5
          ep 46-60:  L=6
          ep 61-75:  L=7
          ep 76-100: L=8
        """
        if epoch <= 15:
            active = 3
        elif epoch <= 30:
            active = 4
        elif epoch <= 45:
            active = 5
        elif epoch <= 60:
            active = 6
        elif epoch <= 75:
            active = 7
        else:
            active = self.num_freqs  # 8
        self.active_freqs.fill_(min(active, self.num_freqs))

    def forward(self, x):
        active = int(self.active_freqs.item())
        encoded = [x]
        for k in range(self.num_freqs):
            if k < active:
                encoded.append(torch.sin(self.freq_bands[k] * x))
                encoded.append(torch.cos(self.freq_bands[k] * x))
            else:
                # Inactive frequency: output zeros (maintains tensor shape)
                encoded.append(torch.zeros_like(x))
                encoded.append(torch.zeros_like(x))
        return torch.cat(encoded, dim=-1)


# =============================================================================
# Experiment 5: PerAxisFrequencyPositionalEncoding (PerAxisPE)
#
# Independent learnable frequencies for each spatial axis (x, y, z).
# ShapeNet objects are axis-aligned (chairs upright, tables flat), so different
# axes have different geometric complexity. This lets the model allocate
# frequency budget per direction.
#
# Parameters: 3 axes * num_freqs = 24 learnable log-frequencies.
# At initialization, all three axes have identical log_freqs so the output is
# equivalent to standard sinusoidal PE.
# =============================================================================
class PerAxisFrequencyPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.output_dim = input_dim + input_dim * 2 * num_freqs

        # Learnable log-frequencies, one per axis
        # Shape: (input_dim, num_freqs) = (3, 8)
        init_log_freqs = torch.log(2.0 ** torch.arange(num_freqs).float())
        # Broadcast to shape (input_dim, num_freqs)
        init_log_freqs = init_log_freqs.unsqueeze(0).expand(input_dim, -1).contiguous()
        self.log_freqs = nn.Parameter(init_log_freqs)  # (3, 8)

    def forward(self, x):
        # x: (N, 3)
        # Positive frequencies per axis: (3, num_freqs)
        freqs = torch.exp(self.log_freqs)

        encoded = [x]
        for k in range(self.num_freqs):
            # For frequency band k, use per-axis frequencies freqs[:, k]
            axis_freqs = freqs[:, k]  # (3,)
            scaled = x * axis_freqs.unsqueeze(0)  # (N, 3) — broadcast
            encoded.append(torch.sin(scaled))
            encoded.append(torch.cos(scaled))
        return torch.cat(encoded, dim=-1)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding=False,
        positional_encoding_num_freqs=6,
        positional_encoding_type="sinusoidal",
        # Spline PE params
        spline_code_num=64,
        spline_code_channel=64,
        # LFF params
        lff_fourier_dim=128,
        lff_hidden_dim=64,
        lff_output_dim=64,
        lff_gamma=10.0,
        # RFF params
        rff_num_frequencies=64,
        rff_scale=10.0,
        # IGR params (Gropp et al., 2020)
        activation="relu",
        activation_beta=100,
        geometric_init=False,
        radius_init=1.0,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        # IGR: geometric_init is incompatible with weight_norm
        if geometric_init and weight_norm:
            weight_norm = False

        self.geometric_init = geometric_init

        # Idea 5: Determine internal xyz dimension based on positional encoding
        # Supports: sinusoidal (original), spe, rff, lff, spline
        self.positional_encoding = positional_encoding
        self.use_spe = False  # SPE: sin() activation on first layer

        if positional_encoding:
            pe_type = positional_encoding_type
            if pe_type in ("sinusoidal", "spe"):
                self.pos_enc = PositionalEncoding(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
                self.use_spe = (pe_type == "spe")
            elif pe_type == "rff":
                self.pos_enc = RandomFourierFeatures(
                    num_frequencies=rff_num_frequencies, scale=rff_scale, input_dim=3
                )
            elif pe_type == "lff":
                self.pos_enc = LearnableFourierFeatures(
                    fourier_dim=lff_fourier_dim, hidden_dim=lff_hidden_dim,
                    output_dim=lff_output_dim, gamma=lff_gamma, input_dim=3
                )
            elif pe_type == "spline":
                self.pos_enc = SplinePositionalEncoding(
                    code_num=spline_code_num, code_channel=spline_code_channel, input_dim=3
                )
            elif pe_type == "lff_v2":
                self.pos_enc = LearnableFourierFeaturesV2(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
            elif pe_type == "hybrid":
                self.pos_enc = HybridPositionalEncoding(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
            elif pe_type == "adaptive":
                self.pos_enc = AdaptiveFrequencyPositionalEncoding(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
            elif pe_type == "progfreq":
                self.pos_enc = ProgressiveFrequencyPositionalEncoding(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
            elif pe_type == "peraxis":
                self.pos_enc = PerAxisFrequencyPositionalEncoding(
                    num_freqs=positional_encoding_num_freqs, input_dim=3
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {pe_type}")
            self.xyz_dim = self.pos_enc.output_dim
        else:
            self.pos_enc = None
            self.xyz_dim = 3

        # First layer width adapts to xyz_dim:
        #   PE OFF: dims[0] = latent_size + 3   (e.g. 256+3 = 259)
        #   PE ON:  dims[0] = latent_size + 39  (e.g. 256+39 = 295)
        dims = [latent_size + self.xyz_dim] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.xyz_dim

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                lin = nn.Linear(dims[layer], out_dim)
                # IGR geometric initialization (Gropp et al., 2020)
                # Initializes network to predict SDF of a unit sphere:
                #   f(x) ≈ ||x|| - radius_init
                # This ensures ||∇f|| ≈ 1 from the start, so Eikonal loss
                # begins near zero instead of fighting reconstruction loss.
                if geometric_init:
                    if layer == self.num_layers - 2:
                        # Last layer: predict unit sphere
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[layer]),
                            std=0.00001,
                        )
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(
                            lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                        )
                setattr(self, "lin" + str(layer), lin)

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        # IGR: configurable activation — Softplus(β=100) for smooth Eikonal
        # gradients, or ReLU (default, backward compatible)
        if activation == "softplus":
            self.activation = nn.Softplus(beta=activation_beta)
        else:
            self.activation = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # IGR fix: Tanh output clipping breaks Eikonal (clips SDF to [-1,1]).
        # With geometric_init, the network should output unbounded SDF values.
        # Standard DeepSDF (no geometric_init) uses Tanh for backward compat.
        if not geometric_init:
            self.th = nn.Tanh()

    # Idea 5: Forward pass with internal positional encoding
    #
    # The EXTERNAL interface is unchanged — callers always pass:
    #   input: (N, L+3)  where last 3 cols are raw xyz
    #
    # INTERNALLY, forward() splits off xyz, optionally encodes it, then
    # rebuilds full_input with the wider representation:
    #
    # Tensor shape flow (PE ON, L=256, 8-layer, latent_in=[4]):
    #   input:       (N, 259) = [latent(256) | xyz(3)]     <- external interface
    #   xyz:         (N, 3)                                  <- split off last 3 cols
    #   xyz_encoded: (N, 39)                                 <- PositionalEncoding
    #   latent_vecs: (N, 256)                                <- everything before xyz
    #   full_input:  (N, 295) = cat([latent(256), encoded_xyz(39)])
    #
    #   Layer 0: Linear(295 -> 512) -> ReLU -> Dropout   x: (N, 512)
    #   Layer 1: Linear(512 -> 512) -> ReLU -> Dropout   x: (N, 512)
    #   Layer 2: Linear(512 -> 512) -> ReLU -> Dropout   x: (N, 512)
    #   Layer 3: Linear(512 -> 512) -> ReLU -> Dropout   x: (N, 512)
    #   Layer 4: cat([x, full_input]) = (N, 807)  <- SKIP CONNECTION
    #            Linear(807 -> 512) -> ReLU -> Dropout   x: (N, 512)
    #   Layer 5-7: Linear(512 -> 512) each               x: (N, 512)
    #   Layer 8: Linear(512 -> 1)                        x: (N, 1)
    #   Tanh:                                            x: (N, 1)
    def forward(self, input):
        # Split: always take last 3 columns as raw xyz
        xyz = input[:, -3:]  # (N, 3)

        # Idea 5: Encode xyz if positional encoding is enabled
        # (N, 3) -> (N, 39) with PE ON, or stays (N, 3) with PE OFF
        if self.pos_enc is not None:
            xyz_encoded = self.pos_enc(xyz)  # (N, 3) -> (N, 39)
        else:
            xyz_encoded = xyz  # (N, 3) unchanged

        # Everything before the last 3 cols is the "latent" portion
        # This may include [cat_emb | instance_code] if Idea 4 is active
        latent_vecs = input[:, :-3]  # (N, L) or (N, C+L) with category embedding

        if latent_vecs.shape[1] > 0 and self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)

        # Rebuild internal representation with encoded xyz
        # PE OFF: full_input = (N, L+3)   e.g. (N, 259)
        # PE ON:  full_input = (N, L+39)  e.g. (N, 295)
        full_input = torch.cat([latent_vecs, xyz_encoded], 1)
        x = full_input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                # Skip connection: re-inject full_input (with encoded xyz)
                # IGR scales by 1/sqrt(2) to prevent skip from dominating
                x = torch.cat([x, full_input], 1)
                if self.geometric_init:
                    x = x / np.sqrt(2)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz_encoded], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = torch.sin(x) if self.use_spe and layer == 0 else self.activation(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th") and self.th is not None:
            x = self.th(x)

        return x
