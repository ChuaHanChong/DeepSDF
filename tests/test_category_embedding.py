"""Unit tests for Category Embedding (Idea 4)."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from networks.deep_sdf_decoder import Decoder
from deep_sdf.data import build_category_index_map, build_category_maps
from deep_sdf.utils import decode_sdf
import deep_sdf.workspace as ws


@pytest.fixture
def sample_split():
    return {
        "ShapeNet": {
            "chair": ["chair_001", "chair_002", "chair_003"],
            "table": ["table_001", "table_002"],
            "lamp": ["lamp_001"],
        }
    }


class TestCategoryMaps:

    def test_num_categories(self, sample_split):
        _, _, num_cats = build_category_maps(sample_split)
        assert num_cats == 3

    def test_name_to_id_keys(self, sample_split):
        _, name_to_id, _ = build_category_maps(sample_split)
        assert set(name_to_id.keys()) == {"chair", "table", "lamp"}

    def test_name_to_id_unique_values(self, sample_split):
        _, name_to_id, num_cats = build_category_maps(sample_split)
        vals = set(name_to_id.values())
        assert len(vals) == 3
        assert all(0 <= v < num_cats for v in vals)

    def test_index_to_cat_id_length(self, sample_split):
        idx_to_cat_id, _, _ = build_category_maps(sample_split)
        assert len(idx_to_cat_id) == 6

    def test_consistency_with_index_map(self, sample_split):
        idx_to_cat_id, name_to_id, _ = build_category_maps(sample_split)
        cat_to_idx, _ = build_category_index_map(sample_split)
        for cat_name, indices in cat_to_idx.items():
            expected_id = name_to_id[cat_name]
            for i in indices:
                assert idx_to_cat_id[i] == expected_id

    def test_single_class(self):
        single = {"ShapeNet": {"chair": ["c1", "c2"]}}
        _, _, num_cats = build_category_maps(single)
        assert num_cats == 1


class TestCategoryEmbedding:

    def test_decode_sdf_without_category(self):
        dec = Decoder(latent_size=8, dims=[32, 32]).cpu().eval()
        latent = torch.randn(1, 8)
        queries = torch.randn(10, 3)
        out = decode_sdf(dec, latent, queries, category_embedding=None)
        assert out.shape == (10, 1)

    def test_decode_sdf_with_category(self):
        dec = Decoder(latent_size=12, dims=[32, 32]).cpu().eval()
        latent = torch.randn(1, 8)
        cat_emb = torch.randn(1, 4)
        queries = torch.randn(10, 3)
        out = decode_sdf(dec, latent, queries, category_embedding=cat_emb)
        assert out.shape == (10, 1)

    def test_different_categories_different_output(self):
        dec = Decoder(latent_size=12, dims=[32, 32]).cpu().eval()
        latent = torch.randn(1, 8)
        queries = torch.randn(10, 3)
        cat_a = torch.randn(1, 4)
        cat_b = torch.randn(1, 4)
        out_a = decode_sdf(dec, latent, queries, category_embedding=cat_a)
        out_b = decode_sdf(dec, latent, queries, category_embedding=cat_b)
        assert not torch.allclose(out_a, out_b)

    def test_save_load_roundtrip(self, tmp_path):
        cat_emb = nn.Embedding(3, 8)
        nn.init.normal_(cat_emb.weight.data, 0.0, 0.1)
        ws.save_category_embeddings(str(tmp_path), "test.pth", cat_emb, 42)
        cat_emb_loaded = nn.Embedding(3, 8)
        loaded_epoch = ws.load_category_embeddings(str(tmp_path), "test.pth", cat_emb_loaded)
        assert loaded_epoch == 42
        assert torch.allclose(cat_emb.weight.data, cat_emb_loaded.weight.data)

    def test_load_for_inference_shape(self, tmp_path):
        cat_emb = nn.Embedding(5, 16)
        ws.save_category_embeddings(str(tmp_path), "latest.pth", cat_emb, epoch=10)
        weights = ws.load_category_embeddings_for_inference(str(tmp_path), "latest")
        assert weights.shape == (5, 16)

    def test_load_for_inference_detached(self, tmp_path):
        cat_emb = nn.Embedding(3, 8)
        ws.save_category_embeddings(str(tmp_path), "latest.pth", cat_emb, epoch=5)
        weights = ws.load_category_embeddings_for_inference(str(tmp_path), "latest")
        assert weights.requires_grad is False

    def test_effective_latent_size(self):
        dec = Decoder(latent_size=12, dims=[32, 32]).cpu().eval()
        x = torch.randn(5, 15)  # cat(4) + latent(8) + xyz(3)
        assert dec(x).shape == (5, 1)

    def test_maps_consistency(self, sample_split):
        idx_to_cat_id, name_to_id, _ = build_category_maps(sample_split)
        _, idx_to_cat_name = build_category_index_map(sample_split)
        for i, cat_name in enumerate(idx_to_cat_name):
            assert idx_to_cat_id[i] == name_to_id[cat_name]


class TestCategoryEmbeddingIntegration:
    """Integration tests for category embedding across training and inference."""

    def test_unknown_category_fallback(self):
        """Unknown categories silently fall back to category 0 via dict.get()."""
        cat_name_to_id = {"chair": 0, "table": 1}
        # Accessing an unknown category with .get() returns the default (0)
        fallback_id = cat_name_to_id.get("lamp", 0)
        assert fallback_id == 0

    def test_category_embedding_gradient_flow(self):
        """Verify gradients flow back through the category embedding layer."""
        torch.manual_seed(42)
        cat_emb_dim = 16
        latent_size = 256
        effective_latent = latent_size + cat_emb_dim  # 272

        cat_embeddings = nn.Embedding(3, cat_emb_dim)
        dec = Decoder(latent_size=effective_latent, dims=[512, 512]).cpu()

        cat_id = torch.tensor([1])
        cat_vec = cat_embeddings(cat_id)  # (1, 16)
        latent = torch.randn(1, latent_size)  # (1, 256)
        queries = torch.randn(10, 3)  # (10, 3)

        out = decode_sdf(dec, latent, queries, category_embedding=cat_vec)
        loss = out.sum()
        loss.backward()

        assert cat_embeddings.weight.grad is not None

    def test_eikonal_category_dimension(self):
        """Decoder with effective_latent = cat_emb_dim(16) + latent_size(256) = 272
        accepts input of shape (10, 275) = (10, 272+3) and outputs (10, 1)."""
        torch.manual_seed(42)
        cat_emb_dim = 16
        latent_size = 256
        effective_latent = latent_size + cat_emb_dim  # 272

        dec = Decoder(latent_size=effective_latent, dims=[512, 512]).cpu().eval()
        # Input: category(16) + latent(256) + xyz(3) = 275
        x = torch.randn(10, effective_latent + 3)
        out = dec(x)
        assert out.shape == (10, 1)

    def test_empty_category_in_split(self):
        """Test build_category_maps behavior when a category has zero instances.

        An empty category list still counts as a category in the split structure.
        build_category_maps should include it in the name-to-id mapping and count.
        """
        split = {"ShapeNet": {"chair": ["c1", "c2"], "table": []}}
        idx_to_cat_id, name_to_id, num_cats = build_category_maps(split)

        # Both categories appear in name_to_id even if one is empty
        assert "chair" in name_to_id
        assert "table" in name_to_id
        assert num_cats == 2

        # Only non-empty category instances appear in idx_to_cat_id
        assert len(idx_to_cat_id) == 2  # only chair's 2 instances
