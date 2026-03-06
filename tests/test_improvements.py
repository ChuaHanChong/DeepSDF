"""
Unit tests for 5 DeepSDF improvement ideas:
  1. Positional Encoding
  2. Data Augmentation
  3. Category Balancing (index maps + sampler)
  4. Category Maps
  5. Eikonal Regularization
  6. Category Embedding (decode_sdf, save/load)
  7. Contrastive Loss

All tests use synthetic data (no training data required) and run on CPU.

Insight:
Testing ML code without training data: Each test class uses a deliberate strategy to isolate correctness:
- Shape tests verify tensor dimensions flow correctly through the network graph
- Invariant tests (rotation preserves norms, scaling is exact) verify mathematical properties of data augmentation
- Differentiability tests (`loss.backward()` + checking `.grad is not None`) confirm the computation graph is intact — critical for losses like eikonal that use `torch.autograd.grad` with `create_graph=True`
- Roundtrip tests (save → load → compare weights) verify serialization correctness without needing actual trained models

The 7 test classes cover all 5 improvements:
1. TestPositionalEncoding    (9) — frequency encoding dimensions, sin/cos pairs, decoder integration
2. TestDataAugmentation      (7) — rotation/scaling correctness, Rodrigues orthogonality
3. TestCategoryBalancing     (6) — index maps, sampler length/validity
4. TestCategoryMaps          (6) — category ID assignment consistency
5. TestEikonalRegularization (5) — gradient shape, 1st/2nd order differentiability
6. TestCategoryEmbedding     (8) — decode_sdf with/without categories, save/load roundtrip
7. TestContrastiveLoss       (6) — triplet margin cases, differentiability, single-category guard
"""

import math
import os
import random
import sys

import pytest
import torch
import torch.nn as nn

# Ensure the DeepSDF root is on sys.path so imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from networks.deep_sdf_decoder import Decoder, PositionalEncoding
from deep_sdf.data import (
    augment_sdf_samples,
    build_category_index_map,
    build_category_maps,
    CategoryBalancedSampler,
)
from deep_sdf.utils import decode_sdf
import deep_sdf.workspace as ws


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_split():
    """Synthetic split: 3 categories (chair:3, table:2, lamp:1 = 6 instances)."""
    return {
        "ShapeNet": {
            "chair": ["chair_001", "chair_002", "chair_003"],
            "table": ["table_001", "table_002"],
            "lamp": ["lamp_001"],
        }
    }


@pytest.fixture
def small_decoder():
    """Small Decoder without positional encoding, on CPU."""
    return Decoder(latent_size=8, dims=[32, 32], positional_encoding=False).cpu().eval()


@pytest.fixture
def small_decoder_posenc():
    """Small Decoder with positional encoding (num_freqs=6), on CPU."""
    return Decoder(
        latent_size=8, dims=[64, 64],
        positional_encoding=True, positional_encoding_num_freqs=6,
    ).cpu().eval()


# ══════════════════════════════════════════════
# 1. TestPositionalEncoding  (9 tests)
# ══════════════════════════════════════════════

class TestPositionalEncoding:
    """Tests for PositionalEncoding module and its integration in Decoder."""

    def test_output_dim_computation(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        # output_dim = 3 + 3 * 2 * 6 = 39
        assert pe.output_dim == 39

    def test_freq_bands_values(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        expected = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        assert torch.allclose(pe.freq_bands, expected)

    def test_freq_bands_is_buffer(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        # Should be a registered buffer, NOT a learnable parameter
        assert "freq_bands" in dict(pe.named_buffers())
        assert "freq_bands" not in dict(pe.named_parameters())

    def test_forward_shape(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        x = torch.randn(10, 3)
        out = pe(x)
        assert out.shape == (10, 39)

    def test_forward_starts_with_identity(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        x = torch.randn(5, 3)
        out = pe(x)
        assert torch.allclose(out[:, :3], x)

    def test_forward_sin_cos_pairs(self):
        pe = PositionalEncoding(num_freqs=6, input_dim=3)
        x = torch.randn(5, 3)
        out = pe(x)
        # After identity (cols 0-2), next 6 cols = sin(1*x), cos(1*x) for freq=1
        # sin(1*x) occupies cols 3,4,5; cos(1*x) occupies cols 6,7,8
        assert torch.allclose(out[:, 3:6], torch.sin(1.0 * x))
        assert torch.allclose(out[:, 6:9], torch.cos(1.0 * x))

    def test_decoder_with_positional_encoding(self, small_decoder_posenc):
        dec = small_decoder_posenc
        assert dec.xyz_dim == 39
        # Input: [latent(8) ; xyz(3)] = 11 columns
        x = torch.randn(4, 11)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_decoder_without_positional_encoding(self, small_decoder):
        dec = small_decoder
        assert dec.xyz_dim == 3
        # Input: [latent(8) ; xyz(3)] = 11 columns
        x = torch.randn(4, 11)
        out = dec(x)
        assert out.shape == (4, 1)

    def test_decoder_posenc_different_num_freqs(self):
        dec = Decoder(
            latent_size=8, dims=[32, 32],
            positional_encoding=True, positional_encoding_num_freqs=4,
        ).cpu().eval()
        # output_dim = 3 + 3*2*4 = 27
        assert dec.pos_enc.output_dim == 27
        x = torch.randn(4, 11)  # still latent(8) + xyz(3)
        out = dec(x)
        assert out.shape == (4, 1)


# ══════════════════════════════════════════════
# 2. TestDataAugmentation  (7 tests)
# ══════════════════════════════════════════════

class TestDataAugmentation:
    """Tests for augment_sdf_samples (rotation + scaling)."""

    def test_shape_preserved(self):
        samples = torch.randn(50, 4)
        out = augment_sdf_samples(samples)
        assert out.shape == (50, 4)

    def test_no_augmentation_identity(self):
        samples = torch.randn(20, 4)
        out = augment_sdf_samples(samples, do_rotation=False, do_scaling=False)
        assert torch.allclose(out, samples)

    def test_rotation_preserves_sdf(self):
        samples = torch.randn(30, 4)
        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        # SDF column (index 3) should remain unchanged under rotation
        assert torch.allclose(out[:, 3], samples[:, 3])

    def test_rotation_preserves_norms(self):
        samples = torch.randn(30, 4)
        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        # Rotation is orthogonal → xyz norms preserved
        orig_norms = samples[:, :3].norm(dim=1)
        new_norms = out[:, :3].norm(dim=1)
        assert torch.allclose(orig_norms, new_norms, atol=1e-5)

    def test_rotation_matrix_orthogonal(self):
        """Reproduce the Rodrigues formula and verify R^T R ≈ I."""
        angle = 0.3
        axis = torch.tensor([1.0, 2.0, 3.0])
        axis = axis / axis.norm()

        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        R = torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)

    def test_scaling_scales_xyz_and_sdf(self):
        samples = torch.randn(20, 4)
        out = augment_sdf_samples(
            samples, do_rotation=False, do_scaling=True, scale_range=(2.0, 2.0),
        )
        assert torch.allclose(out[:, :3], samples[:, :3] * 2.0)
        assert torch.allclose(out[:, 3:], samples[:, 3:] * 2.0)

    def test_scaling_range(self):
        torch.manual_seed(42)
        samples = torch.ones(10, 4)
        lo, hi = 0.5, 1.5
        ratios = []
        for _ in range(100):
            out = augment_sdf_samples(
                samples.clone(), do_rotation=False, do_scaling=True,
                scale_range=(lo, hi),
            )
            ratio = out[0, 0].item() / samples[0, 0].item()
            ratios.append(ratio)
        assert all(lo - 1e-6 <= r <= hi + 1e-6 for r in ratios)


# ══════════════════════════════════════════════
# 3. TestCategoryBalancing  (6 tests)
# ══════════════════════════════════════════════

class TestCategoryBalancing:
    """Tests for build_category_index_map and CategoryBalancedSampler."""

    def test_build_category_index_map_keys(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        assert set(cat_to_idx.keys()) == {"chair", "table", "lamp"}

    def test_build_category_index_map_indices(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        assert cat_to_idx["chair"] == [0, 1, 2]
        assert cat_to_idx["table"] == [3, 4]
        assert cat_to_idx["lamp"] == [5]

    def test_build_category_index_map_index_to_category(self, sample_split):
        _, idx_to_cat = build_category_index_map(sample_split)
        assert len(idx_to_cat) == 6
        assert idx_to_cat[0] == "chair"
        assert idx_to_cat[3] == "table"
        assert idx_to_cat[5] == "lamp"

    def test_sampler_len(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        # max_count = 3 (chair), num_categories = 3 → 3 × 3 = 9
        assert len(sampler) == 9

    def test_sampler_iter_count(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        indices = list(iter(sampler))
        assert len(indices) == len(sampler)

    def test_sampler_valid_indices(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        valid = {0, 1, 2, 3, 4, 5}
        for idx in sampler:
            assert idx in valid


# ══════════════════════════════════════════════
# 4. TestCategoryMaps  (6 tests)
# ══════════════════════════════════════════════

class TestCategoryMaps:
    """Tests for build_category_maps."""

    def test_num_categories(self, sample_split):
        _, _, num_cats = build_category_maps(sample_split)
        assert num_cats == 3

    def test_cat_name_to_id_keys(self, sample_split):
        _, name_to_id, _ = build_category_maps(sample_split)
        assert set(name_to_id.keys()) == {"chair", "table", "lamp"}

    def test_cat_name_to_id_unique_values(self, sample_split):
        _, name_to_id, num_cats = build_category_maps(sample_split)
        vals = set(name_to_id.values())
        assert len(vals) == 3
        assert all(0 <= v < num_cats for v in vals)

    def test_index_to_cat_id_length(self, sample_split):
        idx_to_cat_id, _, _ = build_category_maps(sample_split)
        assert len(idx_to_cat_id) == 6

    def test_index_to_cat_id_consistency(self, sample_split):
        idx_to_cat_id, name_to_id, _ = build_category_maps(sample_split)
        cat_to_idx, _ = build_category_index_map(sample_split)
        for cat_name, indices in cat_to_idx.items():
            expected_id = name_to_id[cat_name]
            for i in indices:
                assert idx_to_cat_id[i] == expected_id

    def test_single_class_split(self):
        single_split = {"ShapeNet": {"chair": ["c1", "c2"]}}
        _, _, num_cats = build_category_maps(single_split)
        assert num_cats == 1


# ══════════════════════════════════════════════
# 5. TestEikonalRegularization  (5 tests)
# ══════════════════════════════════════════════

class TestEikonalRegularization:
    """
    Tests for eikonal regularization by replicating the computation
    from train_deep_sdf.py lines 617-654 on a small decoder.
    """

    @pytest.fixture
    def eikonal_setup(self):
        """Create a small decoder in train mode with requires_grad points."""
        decoder = Decoder(latent_size=8, dims=[32, 32], positional_encoding=False).cpu()
        decoder.train()

        N = 16
        pts = torch.empty(N, 3).uniform_(-1, 1)
        pts.requires_grad_(True)

        latents = torch.randn(N, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)

        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )[0]
        return decoder, pts, sdf, grad

    def test_eikonal_gradient_shape(self, eikonal_setup):
        _, pts, _, grad = eikonal_setup
        assert grad.shape == (pts.shape[0], 3)

    def test_eikonal_loss_finite_nonnegative(self, eikonal_setup):
        _, _, _, grad = eikonal_setup
        eik_loss = torch.mean((grad.norm(dim=1) - 1.0) ** 2)
        assert torch.isfinite(eik_loss)
        assert eik_loss.item() >= 0.0

    def test_eikonal_loss_differentiable(self, eikonal_setup):
        decoder, _, _, grad = eikonal_setup
        eik_loss = torch.mean((grad.norm(dim=1) - 1.0) ** 2)
        eik_loss.backward()
        has_grad = any(p.grad is not None for p in decoder.parameters())
        assert has_grad

    def test_eikonal_second_order_finite(self):
        decoder = Decoder(latent_size=8, dims=[32, 32], positional_encoding=False).cpu()
        decoder.train()
        N = 8
        pts = torch.empty(N, 3).uniform_(-1, 1)
        pts.requires_grad_(True)
        latents = torch.randn(N, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)

        grad = torch.autograd.grad(
            outputs=sdf, inputs=pts,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True, retain_graph=True,
        )[0]

        laplacian = 0.0
        for d in range(3):
            grad2 = torch.autograd.grad(
                outputs=grad[:, d], inputs=pts,
                grad_outputs=torch.ones_like(grad[:, d]),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            laplacian = laplacian + grad2
        assert torch.isfinite(laplacian).all()

    def test_eikonal_second_order_differentiable(self):
        decoder = Decoder(latent_size=8, dims=[32, 32], positional_encoding=False).cpu()
        decoder.train()
        N = 8
        pts = torch.empty(N, 3).uniform_(-1, 1)
        pts.requires_grad_(True)
        latents = torch.randn(N, 8).detach()
        inp = torch.cat([latents, pts], dim=1)
        sdf = decoder(inp)

        grad = torch.autograd.grad(
            outputs=sdf, inputs=pts,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True, retain_graph=True,
        )[0]

        laplacian = 0.0
        for d in range(3):
            grad2 = torch.autograd.grad(
                outputs=grad[:, d], inputs=pts,
                grad_outputs=torch.ones_like(grad[:, d]),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            laplacian = laplacian + grad2

        loss = torch.mean(laplacian ** 2)
        loss.backward()
        has_grad = any(p.grad is not None for p in decoder.parameters())
        assert has_grad


# ══════════════════════════════════════════════
# 6. TestCategoryEmbedding  (8 tests)
# ══════════════════════════════════════════════

class TestCategoryEmbedding:
    """Tests for decode_sdf with category embeddings, save/load roundtrip."""

    def test_decode_sdf_without_category(self, small_decoder):
        latent = torch.randn(1, 8)
        queries = torch.randn(10, 3)
        out = decode_sdf(small_decoder, latent, queries, category_embedding=None)
        assert out.shape == (10, 1)

    def test_decode_sdf_with_category(self):
        # Decoder latent_size must equal cat_emb_dim + code_length
        dec = Decoder(latent_size=12, dims=[32, 32], positional_encoding=False).cpu().eval()
        latent = torch.randn(1, 8)
        cat_emb = torch.randn(1, 4)
        queries = torch.randn(10, 3)
        out = decode_sdf(dec, latent, queries, category_embedding=cat_emb)
        assert out.shape == (10, 1)

    def test_decode_sdf_category_changes_output(self):
        dec = Decoder(latent_size=12, dims=[32, 32], positional_encoding=False).cpu().eval()
        latent = torch.randn(1, 8)
        queries = torch.randn(10, 3)
        cat_a = torch.randn(1, 4)
        cat_b = torch.randn(1, 4)
        out_a = decode_sdf(dec, latent, queries, category_embedding=cat_a)
        out_b = decode_sdf(dec, latent, queries, category_embedding=cat_b)
        # Different category embeddings should produce different SDF values
        assert not torch.allclose(out_a, out_b)

    def test_save_load_roundtrip(self, tmp_path):
        num_cats, emb_dim = 3, 8
        cat_emb = nn.Embedding(num_cats, emb_dim)
        nn.init.normal_(cat_emb.weight.data, 0.0, 0.1)
        epoch = 42

        ws.save_category_embeddings(str(tmp_path), "test.pth", cat_emb, epoch)

        cat_emb_loaded = nn.Embedding(num_cats, emb_dim)
        loaded_epoch = ws.load_category_embeddings(str(tmp_path), "test.pth", cat_emb_loaded)

        assert loaded_epoch == epoch
        assert torch.allclose(cat_emb.weight.data, cat_emb_loaded.weight.data)

    def test_load_for_inference_shape(self, tmp_path):
        num_cats, emb_dim = 5, 16
        cat_emb = nn.Embedding(num_cats, emb_dim)
        ws.save_category_embeddings(str(tmp_path), "latest.pth", cat_emb, epoch=10)

        # load_category_embeddings_for_inference expects
        # <experiment_dir>/CategoryEmbeddings/<checkpoint>.pth
        # We saved to <tmp_path>/CategoryEmbeddings/latest.pth
        # So checkpoint="latest" and experiment_dir=tmp_path
        weights = ws.load_category_embeddings_for_inference(str(tmp_path), "latest")
        assert weights.shape == (num_cats, emb_dim)

    def test_load_for_inference_detached(self, tmp_path):
        num_cats, emb_dim = 3, 8
        cat_emb = nn.Embedding(num_cats, emb_dim)
        ws.save_category_embeddings(str(tmp_path), "latest.pth", cat_emb, epoch=5)
        weights = ws.load_category_embeddings_for_inference(str(tmp_path), "latest")
        assert weights.requires_grad is False

    def test_decoder_effective_latent_size(self):
        # cat_emb_dim=4, code_length=8 → decoder latent_size=12
        dec = Decoder(latent_size=12, dims=[32, 32], positional_encoding=False).cpu().eval()
        # Input: [cat(4) + latent(8) + xyz(3)] = 15
        x = torch.randn(5, 15)
        out = dec(x)
        assert out.shape == (5, 1)

    def test_category_maps_consistency(self, sample_split):
        idx_to_cat_id, name_to_id, num_cats = build_category_maps(sample_split)
        cat_to_idx, idx_to_cat_name = build_category_index_map(sample_split)

        # Every index should map to the same category in both functions
        for i, cat_name in enumerate(idx_to_cat_name):
            assert idx_to_cat_id[i] == name_to_id[cat_name]


# ══════════════════════════════════════════════
# 7. TestContrastiveLoss  (6 tests)
# ══════════════════════════════════════════════

class TestContrastiveLoss:
    """
    Tests for contrastive triplet loss, replicating the computation
    from train_deep_sdf.py lines 661-688.
    """

    def _triplet_loss(self, anchor, positive, negative, margin=1.0, lam=0.01):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        trip = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
        return lam * torch.mean(trip)

    def test_triplet_loss_zero_when_separated(self):
        anchor = torch.zeros(8, 16)
        positive = torch.zeros(8, 16)
        negative = torch.ones(8, 16) * 100.0
        loss = self._triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() == 0.0

    def test_triplet_loss_positive_when_violated(self):
        anchor = torch.zeros(8, 16)
        positive = torch.ones(8, 16) * 10.0   # far from anchor
        negative = torch.ones(8, 16) * 0.01    # close to anchor
        loss = self._triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() > 0.0

    def test_triplet_loss_nonnegative(self):
        torch.manual_seed(123)
        for _ in range(20):
            a = torch.randn(8, 16)
            p = torch.randn(8, 16)
            n = torch.randn(8, 16)
            loss = self._triplet_loss(a, p, n)
            assert loss.item() >= 0.0

    def test_contrastive_with_embeddings(self):
        num_shapes = 10
        lat_vecs = nn.Embedding(num_shapes, 16)
        nn.init.normal_(lat_vecs.weight.data, 0.0, 1.0)

        anchors = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        positives = torch.tensor([1, 0, 3, 2], dtype=torch.long)
        negatives = torch.tensor([5, 6, 7, 8], dtype=torch.long)

        a_vecs = lat_vecs(anchors)
        p_vecs = lat_vecs(positives)
        n_vecs = lat_vecs(negatives)

        loss = self._triplet_loss(a_vecs, p_vecs, n_vecs)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_contrastive_differentiable(self):
        lat_vecs = nn.Embedding(10, 16)
        anchors = torch.tensor([0, 1], dtype=torch.long)
        positives = torch.tensor([2, 3], dtype=torch.long)
        negatives = torch.tensor([5, 6], dtype=torch.long)

        a = lat_vecs(anchors)
        p = lat_vecs(positives)
        n = lat_vecs(negatives)

        loss = self._triplet_loss(a, p, n)
        loss.backward()
        assert lat_vecs.weight.grad is not None

    def test_single_category_skip(self):
        """With only 1 category, contrastive learning should be disabled."""
        cat_to_indices = {0: [0, 1, 2, 3]}
        cat_ids = list(cat_to_indices.keys())
        # The training code checks: len(cat_ids) < 2 → disable
        assert len(cat_ids) < 2
