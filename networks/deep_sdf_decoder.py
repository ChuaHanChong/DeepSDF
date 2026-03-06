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
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        # Idea 5: Determine internal xyz dimension based on positional encoding
        # PE OFF: xyz_dim = 3 (raw coordinates)
        # PE ON:  xyz_dim = 3 + 3*2*num_freqs = 39 (with 6 frequency bands)
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_enc = PositionalEncoding(
                num_freqs=positional_encoding_num_freqs, input_dim=3
            )
            self.xyz_dim = self.pos_enc.output_dim  # e.g. 39
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
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
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
                # x: (N, 512) + full_input: (N, 295) -> (N, 807)
                x = torch.cat([x, full_input], 1)
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
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
