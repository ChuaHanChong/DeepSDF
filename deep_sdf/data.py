#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import math
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws


# =============================================================================
# Idea 1 — Data Augmentation
#
# Applies random rotation and/or scaling to SDF samples (N, 4) = [x, y, z, sdf].
# The shape is always (N, 4) in, (N, 4) out — no dimension change.
#
# Tensor shape flow:
#   samples: (N, 4) = [x, y, z, sdf]
#        | split
#   xyz: (N, 3)       sdf: (N, 1)
#        | rotation (Rodrigues formula)
#   axis:  (3,)   — random unit vector
#   angle: scalar — sampled from Uniform(-range, +range) in radians
#   K:     (3, 3) — skew-symmetric matrix from axis
#   R:     (3, 3) — rotation matrix: I + sin(theta)*K + (1-cos(theta))*K^2
#   xyz_rot = (R @ xyz.T).T       (N, 3)  — norms preserved (R is orthogonal)
#        | scaling
#   scale: scalar — sampled from Uniform(lo, hi)
#   xyz_scaled = xyz_rot * scale  (N, 3)
#   sdf_scaled = sdf * scale      (N, 1)  — SDF scales linearly with spatial scaling
#        | reassemble
#   output: cat([xyz_scaled, sdf_scaled])  (N, 4)
#
# Why scale SDF too? SDF measures distance to the surface. If you scale the
# shape by factor s, all distances also scale by s.
#
# Demo:
#   samples = torch.randn(1000, 4)
#   rotated = augment_sdf_samples(samples, do_rotation=True, do_scaling=False)
#   assert rotated.shape == (1000, 4)
#   assert torch.allclose(samples[:,:3].norm(dim=1), rotated[:,:3].norm(dim=1), atol=1e-5)
#   assert torch.allclose(samples[:,3], rotated[:,3])  # SDF unchanged by rotation
# =============================================================================
def augment_sdf_samples(samples, rotation_range=15.0, scale_range=(0.9, 1.1),
                        do_rotation=True, do_scaling=True):
    xyz = samples[:, :3]  # (N, 3)
    sdf = samples[:, 3:]  # (N, 1)

    if do_rotation and rotation_range > 0:
        # Sample random angle in [-rotation_range, +rotation_range] degrees
        angle = (torch.rand(1).item() * 2 - 1) * rotation_range * math.pi / 180.0
        # Sample random rotation axis and normalize to unit vector
        axis = torch.randn(3)   # (3,)
        axis = axis / axis.norm()

        # Build skew-symmetric matrix K from axis vector:
        # K = [  0  -az  ay ]
        #     [  az  0  -ax ]    shape: (3, 3)
        #     [ -ay  ax  0  ]
        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
        # R is orthogonal (R^T R = I) and det(R) = 1, so it's a proper rotation
        R = (torch.eye(3)                       # (3, 3)
             + math.sin(angle) * K              # (3, 3)
             + (1 - math.cos(angle)) * (K @ K)) # (3, 3)

        # Apply rotation: xyz.T is (3, N), R @ xyz.T is (3, N), .T gives (N, 3)
        xyz = (R @ xyz.T).T  # (N, 3) — norms preserved, SDF unchanged

    if do_scaling:
        # Sample uniform scale factor in [lo, hi]
        scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
        xyz = xyz * scale  # (N, 3) — scale coordinates
        sdf = sdf * scale  # (N, 1) — SDF scales linearly with spatial scaling

    return torch.cat([xyz, sdf], dim=1)  # (N, 4) — reassemble


# =============================================================================
# Idea 1 — Category Balancing (helper functions)
#
# These functions map the split JSON structure into flat index arrays used
# by the DataLoader and training loop.
#
# Example split:
#   {'ShapeNetV2': {'chair': ['a','b','c','d','e'], 'lamp': ['f'], 'table': ['g','h']}}
#
# build_category_index_map returns:
#   category_to_indices = {'chair': [0,1,2,3,4], 'lamp': [5], 'table': [6,7]}
#   index_to_category   = ['chair','chair','chair','chair','chair','lamp','table','table']
#
# build_category_maps returns (integer IDs for tensor ops):
#   index_to_cat_id = [0, 0, 0, 0, 0, 1, 2, 2]     — used as: cat_id_tensor[batch_indices]
#   cat_name_to_id  = {'chair': 0, 'lamp': 1, 'table': 2}
#   num_categories  = 3
#
# CategoryBalancedSampler:
#   Oversamples minority categories so each gets equal representation.
#   num_samples = max_count(5) * num_categories(3) = 15
#   Each category gets ~5 samples per epoch (lamp is repeated 5x).
# =============================================================================
def build_category_index_map(split):
    category_to_indices = {}
    index_to_category = []
    idx = 0
    for dataset in split:
        for class_name in split[dataset]:
            if class_name not in category_to_indices:
                category_to_indices[class_name] = []
            for instance_name in split[dataset][class_name]:
                category_to_indices[class_name].append(idx)
                index_to_category.append(class_name)
                idx += 1
    return category_to_indices, index_to_category


class CategoryBalancedSampler(torch.utils.data.Sampler):
    """Oversamples minority categories so each gets equal representation per epoch.

    Given category_to_indices = {'chair': [0,1,2,3,4], 'lamp': [5], 'table': [6,7]}:
      max_count = 5 (chair has the most)
      num_samples = 5 * 3 = 15 (each category gets 5 samples)
      lamp indices [5] are repeated 5x, table [6,7] ~3x, chair used as-is
    """
    def __init__(self, category_to_indices):
        self.category_to_indices = category_to_indices
        self.categories = list(category_to_indices.keys())
        max_count = max(len(v) for v in category_to_indices.values())
        self.num_samples = max_count * len(self.categories)

    def __iter__(self):
        indices = []
        for cat in self.categories:
            cat_indices = self.category_to_indices[cat]
            if len(cat_indices) == 0:
                continue
            # Repeat minority category indices to match target count per category
            repeats = self.num_samples // (len(self.categories) * len(cat_indices)) + 1
            expanded = (cat_indices * repeats)[: self.num_samples // len(self.categories)]
            indices.extend(expanded)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_category_maps(split):
    """Build integer ID mappings used by training loop for category embedding lookups.

    Returns:
        index_to_cat_id: list[int] — category ID for each flat dataset index
            e.g. [0, 0, 0, 0, 0, 1, 2, 2] for 5 chairs, 1 lamp, 2 tables
            Used as: batch_cat_ids = index_to_cat_id_tensor[batch_indices]  -> (N,) long
        cat_name_to_id: dict — category name to integer ID
        num_categories: int — total number of categories
    """
    index_to_cat_id = []
    cat_name_to_id = {}
    cat_counter = 0
    for dataset in split:
        for class_name in split[dataset]:
            if class_name not in cat_name_to_id:
                cat_name_to_id[class_name] = cat_counter
                cat_counter += 1
            cat_id = cat_name_to_id[class_name]
            for instance_name in split[dataset][class_name]:
                index_to_cat_id.append(cat_id)
    num_categories = cat_counter
    return index_to_cat_id, cat_name_to_id, num_categories


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        augmentation=None,
    ):
        self.subsample = subsample
        self.augmentation = augmentation

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            samples = unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample)
        else:
            samples = unpack_sdf_samples(filename, self.subsample)

        # Idea 1: Apply data augmentation on-the-fly after unpacking
        # samples: (S, 4) -> augment_sdf_samples -> (S, 4)  shape unchanged
        # where S = self.subsample (e.g. 16384)
        if self.augmentation is not None and self.augmentation.get("Enabled", False):
            samples = augment_sdf_samples(
                samples,
                rotation_range=self.augmentation.get("RotationRange", 15.0),
                scale_range=self.augmentation.get("ScaleRange", [0.9, 1.1]),
                do_rotation=self.augmentation.get("RandomRotation", True),
                do_scaling=self.augmentation.get("RandomScaling", True),
            )

        return samples, idx
