#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import logging

from deep_sdf.efficient_sampling import sample_mesh_ea


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000,
                            sampling_method='iid'):
    """
    Compute symmetric Chamfer distance between ground truth points and a
    reconstructed mesh.

    gt_points: trimesh.points.PointCloud — ground truth surface samples
    gen_mesh: trimesh.base.Trimesh — reconstructed mesh
    offset, scale: normalization parameters
    num_mesh_samples: number of points to sample from reconstructed mesh
    sampling_method: 'iid' (original, area-weighted random) or
                     'ea' (Efficient Approximation FPS for uniform coverage)

    The EA sampling method implements the Efficient Approximation from the
    Differential Geometry lecture notes:
      1. Oversample the mesh surface (3x target, IID)
      2. Build k-NN graph (k=8)
      3. Iteratively select farthest points via Dijkstra
    This produces more uniformly distributed samples, yielding more accurate
    and robust Chamfer distance estimates.
    """

    if sampling_method == 'ea':
        gen_points_sampled = sample_mesh_ea(
            gen_mesh, num_mesh_samples,
            oversample_ratio=3, k=8, method='euclidean'
        )
        logging.debug("Using EA-FPS sampling for Chamfer evaluation")
    else:
        gen_points_sampled = trimesh.sample.sample_surface(
            gen_mesh, num_mesh_samples
        )[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    # one direction: gt -> gen
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction: gen -> gt
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer
