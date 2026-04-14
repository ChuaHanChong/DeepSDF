#!/usr/bin/env python3
"""Extract surface normals from raw ShapeNet meshes for IGR training.

Creates companion .npz files with surface points and normals that can be loaded
alongside the existing SDF data during training. Matches the data format used by
IGR (Gropp et al., ICML 2020): surface points paired with face normals.

Usage:
    python scripts/extract_normals.py \
        --data_dir /data/hanchong/AI6131/data \
        --mesh_dir /data/hanchong/AI6131/ShapeNetV2_raw \
        --split examples/splits/sv2_chairs_train.json \
        --num_samples 100000

Output:
    data/NormalSamples/ShapeNetV2/{category}/{shape}.npz
    Keys: 'points' (N, 3), 'normals' (N, 3)
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deep_sdf.workspace as ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_normals_for_shape(mesh_path, norm_params_path, num_samples):
    """Extract surface points and normals from a mesh.

    Args:
        mesh_path: Path to model_normalized.obj
        norm_params_path: Path to normalization parameters .npz
        num_samples: Number of surface points to sample

    Returns:
        (points, normals) each (N, 3) numpy arrays, or None on failure
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        logging.warning("Failed to load mesh %s: %s", mesh_path, e)
        return None

    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        logging.warning("Mesh has no faces: %s", mesh_path)
        return None

    # Sample surface points with face indices
    points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_idx]

    # Apply DeepSDF normalization: (point - offset) / scale
    # Normals are direction vectors — uniform scaling doesn't change direction
    try:
        norm_params = np.load(norm_params_path)
        offset = norm_params["offset"]  # (3,)
        scale = norm_params["scale"]    # scalar
        points = (points - offset) / scale
    except Exception as e:
        logging.warning("No normalization params for %s: %s", mesh_path, e)
        return None

    # Ensure normals are unit-length (trimesh face_normals should already be, but verify)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normals = normals / norms

    return points.astype(np.float32), normals.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract surface normals from ShapeNet meshes")
    parser.add_argument("--data_dir", required=True, help="DeepSDF data directory (contains SdfSamples/, NormalizationParameters/)")
    parser.add_argument("--mesh_dir", required=True, help="Raw ShapeNet mesh directory (e.g., ShapeNetV2_raw/)")
    parser.add_argument("--split", required=True, help="Training split JSON file")
    parser.add_argument("--num_samples", type=int, default=100000, help="Surface points per shape")
    args = parser.parse_args()

    with open(args.split) as f:
        split = json.load(f)

    total = 0
    success = 0
    skipped = 0

    for dataset in split:
        for class_name in split[dataset]:
            shapes = split[dataset][class_name]
            logging.info("Processing %s/%s: %d shapes", dataset, class_name, len(shapes))

            out_dir = os.path.join(args.data_dir, ws.normal_samples_subdir, dataset, class_name)
            os.makedirs(out_dir, exist_ok=True)

            for shape_id in shapes:
                total += 1
                out_path = os.path.join(out_dir, shape_id + ".npz")

                if os.path.exists(out_path):
                    skipped += 1
                    continue

                # Find mesh file
                # ShapeNetV2_raw/{category}/{shape}/models/model_normalized.obj
                # dataset is like "ShapeNetV2/03001627", class_name is already the category
                # But in our split, dataset="ShapeNetV2", class_name="03001627"
                mesh_path = os.path.join(args.mesh_dir, class_name, shape_id, "models", "model_normalized.obj")

                if not os.path.exists(mesh_path):
                    logging.warning("Mesh not found: %s", mesh_path)
                    continue

                # Normalization parameters
                norm_path = ws.get_normalization_params_filename(
                    args.data_dir, dataset, class_name, shape_id
                )

                result = extract_normals_for_shape(mesh_path, norm_path, args.num_samples)
                if result is None:
                    continue

                points, normals = result
                np.savez(out_path, points=points, normals=normals)
                success += 1

                if success % 100 == 0:
                    logging.info("  Processed %d/%d shapes", success, total)

    logging.info("Done: %d success, %d skipped (existing), %d total", success, skipped, total)


if __name__ == "__main__":
    main()
