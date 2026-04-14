"""Unit tests for Data Augmentation + Category Balancing (Idea 1)."""

import math
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deep_sdf.data import (
    augment_sdf_samples,
    build_category_index_map,
    build_category_maps,
    CategoryBalancedSampler,
    load_normals,
)


@pytest.fixture
def sample_split():
    return {
        "ShapeNet": {
            "chair": ["chair_001", "chair_002", "chair_003"],
            "table": ["table_001", "table_002"],
            "lamp": ["lamp_001"],
        }
    }


class TestDataAugmentation:

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
        assert torch.allclose(out[:, 3], samples[:, 3])

    def test_rotation_preserves_norms(self):
        samples = torch.randn(30, 4)
        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        orig_norms = samples[:, :3].norm(dim=1)
        new_norms = out[:, :3].norm(dim=1)
        assert torch.allclose(orig_norms, new_norms, atol=1e-5)

    def test_rotation_matrix_orthogonal(self):
        angle = 0.3
        axis = torch.tensor([1.0, 2.0, 3.0])
        axis = axis / axis.norm()
        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]; K[0, 2] = axis[1]
        K[1, 0] = axis[2]; K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]; K[2, 1] = axis[0]
        R = torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)

    def test_scaling_scales_xyz_and_sdf(self):
        samples = torch.randn(20, 4)
        out = augment_sdf_samples(samples, do_rotation=False, do_scaling=True,
                                  scale_range=(2.0, 2.0))
        assert torch.allclose(out[:, :3], samples[:, :3] * 2.0)
        assert torch.allclose(out[:, 3:], samples[:, 3:] * 2.0)

    def test_scaling_range(self):
        torch.manual_seed(42)
        samples = torch.ones(10, 4)
        ratios = []
        for _ in range(100):
            out = augment_sdf_samples(samples.clone(), do_rotation=False,
                                      do_scaling=True, scale_range=(0.5, 1.5))
            ratios.append(out[0, 0].item())
        assert all(0.5 - 1e-6 <= r <= 1.5 + 1e-6 for r in ratios)


class TestCategoryBalancing:

    def test_index_map_keys(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        assert set(cat_to_idx.keys()) == {"chair", "table", "lamp"}

    def test_index_map_indices(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        assert cat_to_idx["chair"] == [0, 1, 2]
        assert cat_to_idx["table"] == [3, 4]
        assert cat_to_idx["lamp"] == [5]

    def test_index_to_category(self, sample_split):
        _, idx_to_cat = build_category_index_map(sample_split)
        assert len(idx_to_cat) == 6
        assert idx_to_cat[0] == "chair"
        assert idx_to_cat[5] == "lamp"

    def test_sampler_len(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        assert len(sampler) == 9  # max(3) * 3 categories

    def test_sampler_iter_count(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        assert len(list(iter(sampler))) == len(sampler)

    def test_sampler_valid_indices(self, sample_split):
        cat_to_idx, _ = build_category_index_map(sample_split)
        sampler = CategoryBalancedSampler(cat_to_idx)
        valid = {0, 1, 2, 3, 4, 5}
        for idx in sampler:
            assert idx in valid


class TestAugmentationEdgeCases:
    """Tests edge cases in the augment_sdf_samples function (deep_sdf/data.py:17-89).

    Rodrigues rotation: R = I + sin(theta)*K + (1-cos(theta))*K^2
    SDF values are invariant under rotation, scaled linearly under uniform scaling.
    """

    def test_rotation_determinant_is_one(self):
        """Apply augment_sdf_samples with rotation only, extract R from input/output,
        and verify det(R) == 1 (proper rotation, not reflection)."""
        torch.manual_seed(99)
        # Use 3 non-collinear points so we can recover R uniquely
        xyz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        sdf = torch.zeros(3, 1)
        samples = torch.cat([xyz, sdf], dim=1)  # (3, 4)

        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        out_xyz = out[:, :3]  # (3, 3)

        # R = out_xyz @ pinv(xyz);  since xyz is identity, R = out_xyz
        R = out_xyz @ torch.linalg.pinv(xyz)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_rotation_zero_angle_is_identity(self):
        """With rotation_range=0.0 the angle is always 0, so output == input."""
        torch.manual_seed(42)
        samples = torch.randn(20, 4)
        out = augment_sdf_samples(samples, rotation_range=0.0, do_rotation=True,
                                  do_scaling=False)
        assert torch.allclose(out, samples)

    def test_rotation_pi_preserves_norms(self):
        """Rotation by up to 180 degrees still preserves all point norms."""
        torch.manual_seed(7)
        samples = torch.randn(50, 4)
        out = augment_sdf_samples(samples, rotation_range=180.0, do_rotation=True,
                                  do_scaling=False)
        orig_norms = samples[:, :3].norm(dim=1)
        new_norms = out[:, :3].norm(dim=1)
        assert torch.allclose(orig_norms, new_norms, atol=1e-5)

    def test_scaling_identity_range(self):
        """With scale_range=(1.0, 1.0), scaling is a no-op so output == input."""
        torch.manual_seed(42)
        samples = torch.randn(20, 4)
        out = augment_sdf_samples(samples, do_rotation=False, do_scaling=True,
                                  scale_range=(1.0, 1.0))
        assert torch.allclose(out, samples)

    def test_sdf_sign_preservation(self):
        """Rotation does not change SDF values, so signs must be preserved."""
        torch.manual_seed(123)
        samples = torch.randn(100, 4)
        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        # SDF is column 3; rotation leaves it unchanged entirely
        assert torch.equal(torch.sign(out[:, 3]), torch.sign(samples[:, 3]))

    def test_batch_consistency(self):
        """All points in a batch share the same rotation matrix.

        Recover R from the first 3 non-collinear points (standard basis), then
        verify every other point is rotated by the same R.
        """
        # Build 100 points; first 3 are the standard basis for easy R recovery
        torch.manual_seed(77)
        xyz_basis = torch.eye(3)
        xyz_rest = torch.randn(97, 3)
        xyz = torch.cat([xyz_basis, xyz_rest], dim=0)  # (100, 3)
        sdf = torch.zeros(100, 1)
        samples = torch.cat([xyz, sdf], dim=1)  # (100, 4)

        out = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
        out_xyz = out[:, :3]

        # The function computes (R @ xyz.T).T = xyz @ R.T
        # For the first 3 rows where xyz is I: out[:3,:] = I @ R.T = R.T
        R_T = out_xyz[:3, :]  # (3, 3) — this is R transposed

        # Verify all other points follow the same rotation: xyz[i] @ R.T
        expected = xyz[3:] @ R_T  # (97, 3)
        assert torch.allclose(out_xyz[3:], expected, atol=1e-5)


class TestLoadNormals:
    """Tests for the load_normals function (deep_sdf/data.py).

    load_normals loads surface points and normals from NormalSamples/
    and subsamples to num_samples. Returns None on missing file.
    """

    def test_load_normals_returns_correct_shape(self, tmp_path):
        """Create a temp NormalSamples dir, save an npz, load with subsampling."""
        import numpy as np

        normal_dir = tmp_path / "NormalSamples"
        normal_dir.mkdir()
        points = np.random.randn(100, 3).astype(np.float32)
        normals = np.random.randn(100, 3).astype(np.float32)
        np.savez(str(normal_dir / "test.npz"), points=points, normals=normals)

        result = load_normals(str(tmp_path), "test.npz", num_samples=50)
        assert result is not None, "load_normals should return data, not None"
        pts, nrm = result
        assert pts.shape == (50, 3), f"Expected points shape (50,3), got {pts.shape}"
        assert nrm.shape == (50, 3), f"Expected normals shape (50,3), got {nrm.shape}"

    def test_load_normals_missing_file_returns_none(self, tmp_path):
        """Calling load_normals on a nonexistent path should return None."""
        result = load_normals(str(tmp_path), "nonexistent.npz", num_samples=10)
        assert result is None, "load_normals should return None for missing file"

    def test_normals_are_unit_length(self, tmp_path):
        """Load known unit normals and verify ||n|| ~ 1.0 for all returned normals."""
        import numpy as np

        normal_dir = tmp_path / "NormalSamples"
        normal_dir.mkdir()
        # Create random unit normals
        raw = np.random.randn(80, 3).astype(np.float32)
        unit_normals = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        points = np.random.randn(80, 3).astype(np.float32)
        np.savez(str(normal_dir / "test.npz"), points=points, normals=unit_normals)

        result = load_normals(str(tmp_path), "test.npz", num_samples=40)
        assert result is not None
        _, nrm = result
        norms = nrm.norm(dim=1)
        assert torch.allclose(norms, torch.ones(40), atol=1e-5), (
            f"Expected all normals to have unit length, got norms: "
            f"min={norms.min().item():.6f}, max={norms.max().item():.6f}"
        )
