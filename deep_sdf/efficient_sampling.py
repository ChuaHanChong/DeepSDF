#!/usr/bin/env python3
"""
Efficient Approximation of Farthest Point Sampling (EA-FPS)

Implements the sampling method from the Differential Geometry Applications
lecture notes (Sampling-Methods.pdf, pages 5-7):

    Step 1: Oversample the shape using any fast method
    Step 2: Construct a k-Nearest Neighbors (k-NN) graph
    Step 3: Iterative sampling via Dijkstra's algorithm

This approximates geodesic FPS using graph distances on a k-NN graph,
producing uniformly distributed points on the surface without expensive
geodesic computations.

Two implementations are provided:
    1. efficient_approx_fps_graph() — Full graph-based EA from the PDF
       Uses k-NN graph + Dijkstra for approximate geodesic distances
    2. efficient_approx_fps_euclidean() — Euclidean FPS (faster fallback)
       Uses direct Euclidean distances with numpy vectorization

Both produce more uniform point distributions than IID sampling, which
is critical for accurate Chamfer distance evaluation and for providing
the latent code optimizer with spatially uniform SDF information during
reconstruction.
"""

import numpy as np
import heapq
from scipy.spatial import cKDTree


def efficient_approx_fps_graph(points, n_samples, k=8):
    """
    Efficient Approximation of FPS using k-NN graph + Dijkstra.

    Follows the algorithm from Sampling-Methods.pdf:
    1. Build k-NN graph on the oversampled points
    2. Initialize empty set S, add an arbitrary vertex
    3. For each iteration:
       a. Compute distances from S to all vertices via Dijkstra
       b. Select the vertex furthest from S and add it to S

    Args:
        points: (M, D) numpy array of point positions
        n_samples: target number of points to select (N in the PDF)
        k: neighborhood size for k-NN graph (default 8, as in the PDF)

    Returns:
        selected_indices: (n_samples,) numpy array of selected point indices

    Complexity:
        - k-NN construction: O(M log M) via cKDTree
        - Each FPS iteration: O(M k log M) via bounded Dijkstra
        - Total: O(N * M * k * log M)
    """
    M = len(points)
    if n_samples <= 0:
        return np.array([], dtype=np.int64)
    if n_samples >= M:
        return np.arange(M)

    # Step 2: Build k-NN graph
    tree = cKDTree(points)
    knn_dists, knn_indices = tree.query(points, k=k + 1)
    # knn_indices[:, 0] is self (distance 0), [:, 1:] are k nearest neighbors
    neighbors = knn_indices[:, 1:]   # (M, k) neighbor indices
    edge_weights = knn_dists[:, 1:]  # (M, k) edge distances

    # Build undirected adjacency list
    # adj[i] = [(neighbor, weight), ...] with both forward and reverse edges
    adj = [[] for _ in range(M)]
    for i in range(M):
        for j in range(k):
            nb = int(neighbors[i, j])
            w = float(edge_weights[i, j])
            adj[i].append((nb, w))
            adj[nb].append((i, w))

    # Step 3: Iterative FPS with Dijkstra
    selected = np.zeros(n_samples, dtype=np.int64)
    min_dist_to_S = np.full(M, np.inf, dtype=np.float64)

    # Initialize with vertex 0
    selected[0] = 0
    min_dist_to_S[0] = 0.0

    # Run bounded Dijkstra from vertex 0
    _bounded_dijkstra(adj, M, 0, min_dist_to_S)

    for i in range(1, n_samples):
        # Select vertex furthest from S
        next_pt = int(np.argmax(min_dist_to_S))
        selected[i] = next_pt

        # Run bounded Dijkstra from the new point
        new_dists = np.full(M, np.inf, dtype=np.float64)
        new_dists[next_pt] = 0.0

        # Bound: we only need to explore vertices closer than current max
        bound = float(np.max(min_dist_to_S))
        _bounded_dijkstra(adj, M, next_pt, new_dists, bound=bound)

        # Update minimum distances: min_dist_to_S = min(min_dist_to_S, new_dists)
        np.minimum(min_dist_to_S, new_dists, out=min_dist_to_S)

    return selected


def _bounded_dijkstra(adj, M, source, dist_array, bound=float('inf')):
    """
    Bounded Dijkstra on adjacency list graph.

    Computes shortest-path distances from source to all reachable vertices,
    stopping exploration when distance exceeds bound (early termination).

    Args:
        adj: list of lists, adj[u] = [(v, weight), ...] (undirected edges)
        M: number of vertices
        source: source vertex index
        dist_array: (M,) array to fill with distances (modified in-place)
        bound: stop exploring vertices beyond this distance
    """
    dist_array[source] = 0.0
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue
        if d > bound:
            break  # Early termination: all remaining vertices are too far

        visited.add(u)
        dist_array[u] = d

        for v, w in adj[u]:
            if v not in visited:
                new_d = d + w
                if new_d < dist_array[v]:
                    dist_array[v] = new_d
                    heapq.heappush(pq, (new_d, v))


def efficient_approx_fps_euclidean(points, n_samples):
    """
    Euclidean Farthest Point Sampling using numpy vectorization.

    Faster alternative to graph-based EA when points are in Euclidean space
    (no need for geodesic approximation). Uses vectorized distance computation.

    Args:
        points: (M, D) numpy array of point positions
        n_samples: target number of points to select

    Returns:
        selected_indices: (n_samples,) numpy array of selected point indices

    Complexity: O(N * M) with vectorized numpy operations
    """
    M = len(points)
    if n_samples <= 0:
        return np.array([], dtype=np.int64)
    if n_samples >= M:
        return np.arange(M)

    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = 0  # Start with first point
    min_dists = np.full(M, np.inf, dtype=np.float64)

    for i in range(1, n_samples):
        # Squared Euclidean distance from last selected point to all points
        dists = np.sum((points - points[selected[i - 1]]) ** 2, axis=1)
        np.minimum(min_dists, dists, out=min_dists)
        selected[i] = np.argmax(min_dists)

    return selected


def efficient_approx_fps_gpu(points_tensor, n_samples):
    """
    GPU-accelerated Euclidean FPS using PyTorch.

    Fastest implementation for use during reconstruction when GPU is available.
    Each iteration is a single CUDA kernel for distance computation.

    Args:
        points_tensor: (M, D) torch tensor on GPU
        n_samples: target number of points to select

    Returns:
        selected_indices: (n_samples,) numpy array of selected point indices
    """
    import torch

    M = points_tensor.shape[0]
    if n_samples <= 0:
        return np.array([], dtype=np.int64)
    if n_samples >= M:
        return np.arange(M)

    device = points_tensor.device
    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    min_dists = torch.full((M,), float('inf'), device=device, dtype=points_tensor.dtype)

    selected[0] = 0
    for i in range(1, n_samples):
        # Vectorized distance computation on GPU
        last_point = points_tensor[selected[i - 1]].unsqueeze(0)  # (1, D)
        dists = torch.sum((points_tensor - last_point) ** 2, dim=1)  # (M,)
        min_dists = torch.minimum(min_dists, dists)
        selected[i] = torch.argmax(min_dists)

    return selected.cpu().numpy()


def sample_mesh_ea(mesh, n_samples, oversample_ratio=3, k=8, method='graph'):
    """
    Sample points from a mesh surface using Efficient Approximation FPS.

    This is the main entry point for Chamfer distance evaluation.
    Replaces trimesh.sample.sample_surface (IID) with EA-FPS (uniform).

    Follows the PDF algorithm:
    1. Oversample using IID (fast, area-proportional)
    2. Downsample using EA-FPS (approximate geodesic FPS)

    Args:
        mesh: trimesh.Trimesh object
        n_samples: target number of surface points
        oversample_ratio: how many more points to initially sample (default 3x)
        k: k-NN parameter for graph construction (default 8)
        method: 'graph' for full EA, 'euclidean' for Euclidean FPS

    Returns:
        points: (n_samples, 3) numpy array of sampled surface points
    """
    import trimesh

    # Step 1: Oversample the mesh surface using IID (fast)
    oversample_count = n_samples * oversample_ratio
    oversampled_points = trimesh.sample.sample_surface(mesh, oversample_count)[0]

    # Step 2+3: Downsample using EA-FPS
    if method == 'graph':
        selected_idx = efficient_approx_fps_graph(oversampled_points, n_samples, k=k)
    else:
        selected_idx = efficient_approx_fps_euclidean(oversampled_points, n_samples)

    return oversampled_points[selected_idx]


def sample_sdf_ea(sdf_data, n_samples, method='euclidean'):
    """
    Select uniformly distributed SDF samples using EA-FPS.

    Used during reconstruction (test-time latent code optimization) to provide
    more spatially uniform SDF information to the optimizer.

    Args:
        sdf_data: list of [pos_tensor, neg_tensor], each (N, 4) = [x, y, z, sdf]
        n_samples: total number of samples to return
        method: 'euclidean' (fast) or 'graph' (geodesic approximation)

    Returns:
        samples: (n_samples, 4) torch tensor of selected SDF samples
    """
    import torch

    half = n_samples // 2
    pos_tensor = sdf_data[0]  # (N_pos, 4)
    neg_tensor = sdf_data[1]  # (N_neg, 4)

    # Select uniform positive samples
    if pos_tensor.shape[0] <= half:
        selected_pos = pos_tensor
    else:
        pos_xyz = pos_tensor[:, :3].numpy()  # Use xyz for spatial uniformity
        if method == 'euclidean':
            pos_idx = efficient_approx_fps_euclidean(pos_xyz, half)
        else:
            pos_idx = efficient_approx_fps_graph(pos_xyz, half, k=8)
        selected_pos = pos_tensor[pos_idx]

    # Select uniform negative samples
    if neg_tensor.shape[0] <= half:
        selected_neg = neg_tensor
    else:
        neg_xyz = neg_tensor[:, :3].numpy()
        if method == 'euclidean':
            neg_idx = efficient_approx_fps_euclidean(neg_xyz, half)
        else:
            neg_idx = efficient_approx_fps_graph(neg_xyz, half, k=8)
        selected_neg = neg_tensor[neg_idx]

    return torch.cat([selected_pos, selected_neg], dim=0)
