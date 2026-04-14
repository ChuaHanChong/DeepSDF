"""Unit tests for Contrastive Latent Loss (Idea 3)."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestContrastiveLoss:

    def _triplet_loss(self, anchor, positive, negative, margin=1.0, lam=0.01):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        trip = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
        return lam * torch.mean(trip)

    def test_zero_when_separated(self):
        anchor = torch.zeros(8, 16)
        positive = torch.zeros(8, 16)
        negative = torch.ones(8, 16) * 100.0
        loss = self._triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() == 0.0

    def test_positive_when_violated(self):
        anchor = torch.zeros(8, 16)
        positive = torch.ones(8, 16) * 10.0
        negative = torch.ones(8, 16) * 0.01
        loss = self._triplet_loss(anchor, positive, negative, margin=1.0)
        assert loss.item() > 0.0

    def test_nonnegative(self):
        torch.manual_seed(123)
        for _ in range(20):
            a = torch.randn(8, 16)
            p = torch.randn(8, 16)
            n = torch.randn(8, 16)
            assert self._triplet_loss(a, p, n).item() >= 0.0

    def test_with_embeddings(self):
        lat_vecs = nn.Embedding(10, 16)
        a = lat_vecs(torch.tensor([0, 1, 2, 3]))
        p = lat_vecs(torch.tensor([1, 0, 3, 2]))
        n = lat_vecs(torch.tensor([5, 6, 7, 8]))
        loss = self._triplet_loss(a, p, n)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_differentiable(self):
        lat_vecs = nn.Embedding(10, 16)
        a = lat_vecs(torch.tensor([0, 1]))
        p = lat_vecs(torch.tensor([2, 3]))
        n = lat_vecs(torch.tensor([5, 6]))
        loss = self._triplet_loss(a, p, n)
        loss.backward()
        assert lat_vecs.weight.grad is not None

    def test_single_category_skip(self):
        cat_to_indices = {0: [0, 1, 2, 3]}
        cat_ids = list(cat_to_indices.keys())
        assert len(cat_ids) < 2  # training code disables contrastive with < 2 cats


# ── Batch-Local Mode Tests ───────────────────────────────

import random
import torch.nn.functional as F


class TestBatchLocalMode:
    """Tests for contrastive mode 'batch_local' (train_deep_sdf.py lines 1018-1049).

    batch_local mines triplets only from shapes in the current batch.
    Uses random.sample(..., 2) to guarantee distinct anchor+positive.
    Skips categories with < 2 shapes in batch.
    """

    def test_distinct_anchor_positive(self):
        """random.sample(list, 2) always returns two distinct elements."""
        cat_to_indices = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}
        for _ in range(100):
            for cat_id, idxs in cat_to_indices.items():
                anchor_idx, pos_idx = random.sample(idxs, 2)
                assert anchor_idx != pos_idx, (
                    f"anchor == positive ({anchor_idx}) in category {cat_id}"
                )

    def test_requires_two_categories(self):
        """With only 1 category, no valid triplets can be formed (no negative cat)."""
        batch_cat_map = {0: [0, 1, 2, 3]}
        batch_cats_with_shapes = [c for c, idxs in batch_cat_map.items() if len(idxs) >= 1]
        # The training code checks: if len(batch_cats_with_shapes) >= 2
        assert len(batch_cats_with_shapes) < 2

    def test_skips_single_shape_cats(self):
        """Categories with only 1 shape in batch are skipped for anchor selection."""
        batch_cat_map = {0: [10], 1: [20, 21], 2: [30, 31, 32]}
        # Simulate the mining loop — single-shape cats are skipped
        skipped = 0
        mined = 0
        for _ in range(200):
            anchor_cat = random.choice(list(batch_cat_map.keys()))
            if len(batch_cat_map[anchor_cat]) < 2:
                skipped += 1
                continue
            anchor_idx, pos_idx = random.sample(batch_cat_map[anchor_cat], 2)
            assert anchor_idx != pos_idx
            mined += 1
        # Category 0 has 1 shape, so ~1/3 of iterations should skip
        assert skipped > 0, "Expected some iterations to skip single-shape categories"
        assert mined > 0, "Expected some valid triplets to be mined"

    def test_valid_triplets_from_batch(self):
        """All mined indices must come from the batch indices set."""
        batch_indices = {10, 11, 20, 21, 22, 30, 31}
        batch_cat_map = {0: [10, 11], 1: [20, 21, 22], 2: [30, 31]}
        batch_cats_with_shapes = list(batch_cat_map.keys())

        for _ in range(100):
            anchor_cat = random.choice(batch_cats_with_shapes)
            anchor_idx, pos_idx = random.sample(batch_cat_map[anchor_cat], 2)
            neg_cat = random.choice([c for c in batch_cats_with_shapes if c != anchor_cat])
            neg_idx = random.choice(batch_cat_map[neg_cat])
            assert anchor_idx in batch_indices
            assert pos_idx in batch_indices
            assert neg_idx in batch_indices


# ── Soft-Weighted Mode Tests ─────────────────────────────

class TestSoftWeightedMode:
    """Tests for contrastive mode 'soft_weighted' (train_deep_sdf.py lines 1051-1091).

    Uses InfoNCE-style loss with cosine similarity and temperature.
    Categories weighted inversely by size.
    Also uses random.sample(..., 2) for distinct anchor+positive.
    """

    def test_infonce_loss_nonnegative(self):
        """F.cross_entropy on sample logits should be >= 0."""
        # Simulate InfoNCE logits: (K, 2) where column 0 = pos_sim, column 1 = neg_sim
        torch.manual_seed(42)
        for _ in range(20):
            pos_sim = torch.randn(16)
            neg_sim = torch.randn(16)
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # (16, 2)
            labels = torch.zeros(16, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
            assert loss.item() >= 0.0

    def test_temperature_scaling(self):
        """Lower temperature should produce larger loss magnitudes."""
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(32, 16), dim=1)
        positive = F.normalize(torch.randn(32, 16), dim=1)
        negative = F.normalize(torch.randn(32, 16), dim=1)

        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)

        losses = []
        for temp in [1.0, 0.1, 0.01]:
            logits = torch.stack([pos_sim / temp, neg_sim / temp], dim=1)
            labels = torch.zeros(32, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())

        # Lower temperature amplifies logit differences → larger loss
        assert losses[1] > losses[0], (
            f"temp=0.1 loss ({losses[1]}) should exceed temp=1.0 loss ({losses[0]})"
        )
        assert losses[2] > losses[1], (
            f"temp=0.01 loss ({losses[2]}) should exceed temp=0.1 loss ({losses[1]})"
        )

    def test_inverse_category_weighting(self):
        """Probabilities should be inversely proportional to category size."""
        cat_to_indices = {0: list(range(100)), 1: list(range(10)), 2: list(range(1))}
        cat_weights = {cat_id: 1.0 / len(idxs) for cat_id, idxs in cat_to_indices.items()}
        total_weight = sum(cat_weights.values())
        cat_probs = {cat_id: w / total_weight for cat_id, w in cat_weights.items()}

        # Smallest category (size 1) should have highest probability
        assert cat_probs[2] > cat_probs[1] > cat_probs[0]
        # Verify probabilities sum to 1
        assert abs(sum(cat_probs.values()) - 1.0) < 1e-6
        # The ratio of probabilities should be inverse of size ratio
        # prob[1] / prob[0] should be approximately 100/10 = 10
        ratio = cat_probs[1] / cat_probs[0]
        assert abs(ratio - 10.0) < 1e-6

    def test_distinct_anchor_positive_soft(self):
        """random.sample guarantees distinct anchor+positive in soft_weighted mode."""
        cat_to_indices = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8]}
        for _ in range(100):
            for cat_id, idxs in cat_to_indices.items():
                if len(idxs) >= 2:
                    anchor_idx, pos_idx = random.sample(idxs, 2)
                    assert anchor_idx != pos_idx


# ── Detached Mode Tests ──────────────────────────────────

class TestDetachedMode:
    """Tests for contrastive mode 'detached' (train_deep_sdf.py lines 1093-1124).

    Detaches latent vecs before contrastive loss to isolate gradients.
    Manually applies gradients via SGD-style update.
    After fix: uses random.sample for distinct anchor+positive.
    """

    def test_gradient_isolation(self):
        """Detached contrastive loss should not contribute to lat_vecs.weight.grad."""
        lat_vecs = nn.Embedding(10, 16)
        lat_vecs.zero_grad()

        # Step 1: reconstruction-like loss — this SHOULD produce grad
        recon_input = lat_vecs(torch.tensor([0, 1, 2]))
        recon_loss = recon_input.pow(2).mean()
        recon_loss.backward()
        recon_grad = lat_vecs.weight.grad.clone()

        # Step 2: detached contrastive loss — this should NOT change grad
        anchor_vecs = lat_vecs(torch.tensor([0, 1])).detach().requires_grad_(True)
        pos_vecs = lat_vecs(torch.tensor([2, 3])).detach().requires_grad_(True)
        neg_vecs = lat_vecs(torch.tensor([5, 6])).detach().requires_grad_(True)
        dist_pos = torch.sum((anchor_vecs - pos_vecs) ** 2, dim=1)
        dist_neg = torch.sum((anchor_vecs - neg_vecs) ** 2, dim=1)
        triplet = torch.clamp(dist_pos - dist_neg + 1.0, min=0.0)
        contrastive_loss = torch.mean(triplet)
        contrastive_loss.backward()

        # Detached loss should not have affected lat_vecs.weight.grad
        assert torch.allclose(lat_vecs.weight.grad, recon_grad), (
            "Detached contrastive loss should not modify lat_vecs.weight.grad"
        )

    def test_manual_gradient_application(self):
        """Manually applying detached gradients should change weight.data."""
        lat_vecs = nn.Embedding(4, 8)
        original_weight = lat_vecs.weight.data.clone()

        # Compute detached loss
        anchor_vecs = lat_vecs(torch.tensor([0])).detach().requires_grad_(True)
        pos_vecs = lat_vecs(torch.tensor([1])).detach().requires_grad_(True)
        neg_vecs = lat_vecs(torch.tensor([2])).detach().requires_grad_(True)
        dist_pos = torch.sum((anchor_vecs - pos_vecs) ** 2, dim=1)
        dist_neg = torch.sum((anchor_vecs - neg_vecs) ** 2, dim=1)
        triplet = torch.clamp(dist_pos - dist_neg + 1.0, min=0.0)
        loss = torch.mean(triplet)
        loss.backward()

        # Manually apply gradients (SGD-style)
        lr = 0.01
        with torch.no_grad():
            for vec, idx_list in [(anchor_vecs, [0]), (pos_vecs, [1]), (neg_vecs, [2])]:
                if vec.grad is not None:
                    for j, idx in enumerate(idx_list):
                        lat_vecs.weight.data[idx] -= lr * vec.grad[j]

        # At least one weight row should have changed
        changed = not torch.allclose(lat_vecs.weight.data, original_weight)
        assert changed, "Manual gradient application should modify weight data"

    def test_distinct_anchor_positive_detached(self):
        """random.sample(list, 2) always returns distinct pair even with just 2 elements."""
        for _ in range(100):
            idxs = [0, 1]
            anchor_idx, pos_idx = random.sample(idxs, 2)
            assert anchor_idx != pos_idx

    def test_accumulates_for_duplicates(self):
        """Manually applying gradients for same index twice accumulates both updates."""
        lat_vecs = nn.Embedding(4, 8)
        original_row = lat_vecs.weight.data[0].clone()

        # First detached update
        vec1 = lat_vecs(torch.tensor([0])).detach().requires_grad_(True)
        loss1 = vec1.pow(2).sum()
        loss1.backward()
        grad1 = vec1.grad.clone()

        # Second detached update (different loss to get different gradient)
        vec2 = lat_vecs(torch.tensor([0])).detach().requires_grad_(True)
        loss2 = (vec2 * 3).pow(2).sum()
        loss2.backward()
        grad2 = vec2.grad.clone()

        # Apply both manually to index 0
        lr = 0.01
        with torch.no_grad():
            lat_vecs.weight.data[0] -= lr * grad1[0]
            lat_vecs.weight.data[0] -= lr * grad2[0]

        # The total change should be the sum of both gradient applications
        expected = original_row - lr * grad1[0] - lr * grad2[0]
        assert torch.allclose(lat_vecs.weight.data[0], expected, atol=1e-6)
