#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import random
import time

import deep_sdf
import deep_sdf.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
        if do_cat_embedding:
            ws.save_category_embeddings(experiment_directory, "latest.pth", cat_embeddings, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)
        if do_cat_embedding:
            ws.save_category_embeddings(experiment_directory, str(epoch) + ".pth", cat_embeddings, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            # Category embedding param group (index 2) shares LR with latent codes (index 1)
            schedule_idx = min(i, len(lr_schedules) - 1)
            param_group["lr"] = lr_schedules[schedule_idx].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    # =========================================================================
    # Shared category infrastructure (used by Ideas 3 & 4)
    #
    # build_category_maps produces:
    #   index_to_cat_id: [0,0,...,1,1,...,2,...] — category int ID per shape
    #   cat_name_to_id:  {'chair':0, 'lamp':1, ...}
    #   num_categories:  int
    #
    # index_to_cat_id_tensor allows fast batch lookup:
    #   batch_cat_ids = index_to_cat_id_tensor[batch_indices]  -> (N,) long
    # =========================================================================
    cat_emb_specs = get_spec_with_default(specs, "CategoryEmbedding", {})
    do_cat_embedding = cat_emb_specs.get("Enabled", False)
    contrastive_specs = get_spec_with_default(specs, "ContrastiveLearning", {})
    do_contrastive = contrastive_specs.get("Enabled", False)

    if do_cat_embedding or do_contrastive:
        index_to_cat_id, cat_name_to_id, num_categories = deep_sdf.data.build_category_maps(train_split)
        index_to_cat_id_tensor = torch.tensor(index_to_cat_id, dtype=torch.long)
        logging.info("Category infrastructure: {} categories".format(num_categories))
    else:
        index_to_cat_id_tensor = None
        num_categories = 0

    # =========================================================================
    # Idea 4 — Category Code + Instance Residual
    #
    # Instead of one big latent per shape, decompose into:
    #   [category_embedding(C) | instance_code(L)]
    #
    # The decoder sees effective_latent_size = L + C (e.g. 256 + 64 = 320).
    # It does NOT know about the decomposition — it just gets a bigger "latent".
    #
    # cat_embeddings: nn.Embedding(num_categories, C)  weight: (num_cats, C)
    # All shapes in the same category share the same category embedding.
    # =========================================================================
    cat_emb_dim = 0
    cat_embeddings = None
    cat_emb_reg_lambda = 0.0
    if do_cat_embedding:
        cat_emb_dim = cat_emb_specs.get("EmbeddingDim", 64)
        cat_emb_reg_lambda = cat_emb_specs.get("RegLambda", 1e-4)
        effective_latent_size = latent_size + cat_emb_dim  # e.g. 256 + 64 = 320
        logging.info("Category embedding: dim={}, effective latent={}".format(
            cat_emb_dim, effective_latent_size))
    else:
        effective_latent_size = latent_size  # e.g. 256

    # Decoder receives effective_latent_size (which includes cat_emb_dim if enabled)
    decoder = arch.Decoder(effective_latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    # =========================================================================
    # Idea 1 — Data Augmentation & Category Balancing (DataLoader setup)
    #
    # Augmentation is applied per-sample in SDFSamples.__getitem__:
    #   raw samples: (S, 4) -> augment_sdf_samples -> (S, 4)
    #   where S = SamplesPerScene (e.g. 16384)
    #
    # Category balancing replaces shuffle=True with a custom sampler
    # that oversamples minority categories so each gets equal representation.
    # =========================================================================
    augmentation_specs = get_spec_with_default(specs, "DataAugmentation", None)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False,
        augmentation=augmentation_specs,  # Idea 1: passed to __getitem__
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    # Idea 1: Use CategoryBalancedSampler to oversample minority categories
    do_category_balancing = get_spec_with_default(specs, "CategoryBalancing", False)
    if do_category_balancing:
        cat_to_indices, _ = deep_sdf.data.build_category_index_map(train_split)
        balanced_sampler = deep_sdf.data.CategoryBalancedSampler(cat_to_indices)
        sdf_loader = data_utils.DataLoader(
            sdf_dataset,
            batch_size=scene_per_batch,
            sampler=balanced_sampler,  # replaces shuffle=True
            num_workers=num_data_loader_threads,
            drop_last=True,
        )
    else:
        sdf_loader = data_utils.DataLoader(
            sdf_dataset,
            batch_size=scene_per_batch,
            shuffle=True,
            num_workers=num_data_loader_threads,
            drop_last=True,
        )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    # Idea 4: Category embedding table — one embedding per category
    # cat_embeddings.weight: (num_categories, cat_emb_dim) e.g. (3, 64)
    # Lookup: cat_embeddings(batch_cat_ids) -> (N, 64)
    if do_cat_embedding:
        cat_embeddings = torch.nn.Embedding(num_categories, cat_emb_dim).cuda()
        torch.nn.init.normal_(cat_embeddings.weight.data, 0.0, 0.1)
        logging.info("Initialized {} category embeddings of dim {}".format(
            num_categories, cat_emb_dim))

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    # =========================================================================
    # Idea 2 — Eikonal Regularization config
    #
    # Enforces ||grad_f(x)|| = 1 at M random unsupervised points in [-1,1]^3.
    # This makes the learned function a valid Signed Distance Function.
    # Optional 2nd-order term penalizes the Laplacian (sum of 2nd derivatives).
    # =========================================================================
    eikonal_specs = get_spec_with_default(specs, "EikonalRegularization", {})
    do_eikonal = eikonal_specs.get("Enabled", False)
    if do_eikonal:
        eikonal_lambda = eikonal_specs.get("Lambda", 0.1)
        eikonal_num_samples = eikonal_specs.get("NumUnsupervisedSamples", 4096)
        eikonal_second_order = eikonal_specs.get("SecondOrder", False)
        eikonal_second_order_lambda = eikonal_specs.get("SecondOrderLambda", 0.01)
        logging.info("Eikonal regularization enabled: lambda={}, samples={}".format(
            eikonal_lambda, eikonal_num_samples))

    # =========================================================================
    # Idea 3 — Contrastive Latent Loss config
    #
    # Mines K triplets per batch from the global latent embedding table:
    #   anchor:   random shape from random category
    #   positive: random shape from SAME category
    #   negative: random shape from DIFFERENT category
    #
    # Triplet loss pushes same-category latent codes closer, different-category apart.
    # Requires >= 2 categories; gracefully disabled for single-category experiments.
    # =========================================================================
    if do_contrastive:
        contrastive_lambda = contrastive_specs.get("Lambda", 0.01)
        contrastive_margin = contrastive_specs.get("Margin", 1.0)
        contrastive_triplets = contrastive_specs.get("TripletsPerBatch", 32)
        # Build per-category index lists for triplet mining
        # contrastive_cat_to_indices: {cat_id: [shape_idx, ...]}
        contrastive_cat_to_indices = {}
        for idx, cat_id in enumerate(index_to_cat_id):
            if cat_id not in contrastive_cat_to_indices:
                contrastive_cat_to_indices[cat_id] = []
            contrastive_cat_to_indices[cat_id].append(idx)
        contrastive_cat_ids = list(contrastive_cat_to_indices.keys())
        if len(contrastive_cat_ids) < 2:
            logging.warning("Contrastive learning requires >= 2 categories, disabling")
            do_contrastive = False
        else:
            logging.info("Contrastive learning enabled: lambda={}, margin={}, triplets={}".format(
                contrastive_lambda, contrastive_margin, contrastive_triplets))

    optimizer_param_groups = [
        {
            "params": decoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": lat_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        },
    ]
    # Idea 4: Add category embeddings as 3rd optimizer param group
    # It shares the learning rate schedule with latent codes (index 1)
    # via adjust_learning_rate's min(i, len(schedules)-1) logic
    if do_cat_embedding:
        optimizer_param_groups.append({
            "params": cat_embeddings.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        })

    optimizer_all = torch.optim.Adam(optimizer_param_groups)

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        if do_cat_embedding:
            ws.load_category_embeddings(
                experiment_directory, continue_from + ".pth", cat_embeddings
            )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for sdf_data, indices in sdf_loader:

            # =================================================================
            # Data preparation — tensor shape flow
            #
            # From DataLoader (Idea 1: possibly augmented):
            #   sdf_data: (B, S, 4)  B=ScenesPerBatch(64), S=SamplesPerScene(16384)
            #   indices:  (B,)       shape indices
            #
            # After reshape:
            #   sdf_data: (B*S, 4) = (1048576, 4)
            #   xyz:      (B*S, 3)
            #   sdf_gt:   (B*S, 1)
            #   indices:  (B*S,)   — each sample tagged with its shape index
            #
            # After chunk(batch_split):
            #   xyz[i]:     (N, 3)   where N = B*S / batch_split
            #   sdf_gt[i]:  (N, 1)
            #   indices[i]: (N,)
            # =================================================================
            sdf_data = sdf_data.reshape(-1, 4)  # (B, S, 4) -> (B*S, 4)

            num_sdf_samples = sdf_data.shape[0]  # B*S total samples

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]            # (B*S, 3) — coordinates
            sdf_gt = sdf_data[:, 3].unsqueeze(1)  # (B*S, 1) — ground truth SDF

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            # Split into sub-batches for memory efficiency
            xyz = torch.chunk(xyz, batch_split)
            # Expand indices: (B,) -> (B*S,) by repeating each index S times
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                # =============================================================
                # Idea 4 — Input construction with category embedding
                #
                # Tensor shapes per sub-batch (N = B*S / batch_split):
                #
                # WITH category embedding (C=64, L=256):
                #   batch_vecs:     (N, 256) = lat_vecs(indices)
                #   batch_cat_ids:  (N,)     = index_to_cat_id_tensor[indices] (long)
                #   batch_cat_vecs: (N, 64)  = cat_embeddings(batch_cat_ids)
                #   input:          (N, 323) = cat([cat_vecs(64), lat_vecs(256), xyz(3)])
                #
                # WITHOUT category embedding (L=256):
                #   batch_vecs:     (N, 256)
                #   input:          (N, 259) = cat([lat_vecs(256), xyz(3)])
                #
                # Inside decoder.forward() (Idea 5), xyz(3) is split off and
                # optionally expanded to (N, 39) via PositionalEncoding.
                # =============================================================
                batch_vecs = lat_vecs(indices[i])  # (N, L) = (N, 256)

                if do_cat_embedding:
                    batch_cat_ids = index_to_cat_id_tensor[indices[i]].cuda()  # (N,) long
                    batch_cat_vecs = cat_embeddings(batch_cat_ids)  # (N, C) = (N, 64)
                    input = torch.cat([batch_cat_vecs, batch_vecs, xyz[i]], dim=1)  # (N, C+L+3)
                else:
                    input = torch.cat([batch_vecs, xyz[i]], dim=1)  # (N, L+3)

                # Forward pass: input (N, 323 or 259) -> pred_sdf (N, 1)
                pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                # SDF reconstruction loss: L1(pred, gt) / total_samples
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                # Code regularization: lambda * min(1, epoch/100) * sum(||z_i||) / N
                # Ramps up linearly over first 100 epochs
                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                # Idea 4: Category embedding regularization
                # lambda_c * sum(||c_y||) / N — prevents category embeddings from growing too large
                if do_cat_embedding:
                    cat_reg_loss = cat_emb_reg_lambda * torch.sum(
                        torch.norm(batch_cat_vecs, dim=1)
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + cat_reg_loss.cuda()

                # =============================================================
                # Idea 2 — Eikonal Regularization
                #
                # Sample M random points in [-1,1]^3 (no ground truth needed).
                # Compute the spatial gradient of the SDF w.r.t. these points.
                # Penalize deviation of gradient norm from 1.0.
                #
                # Tensor shapes (M = eikonal_num_samples, e.g. 4096):
                #   eik_pts:     (M, 3)     random points, requires_grad=True
                #   eik_latents: (M, L)     random latent codes from batch, DETACHED
                #   eik_input:   (M, L+3)   or (M, C+L+3) with category embedding
                #   eik_sdf:     (M, 1)     decoder output
                #   eik_grad:    (M, 3)     = [df/dx, df/dy, df/dz] via autograd
                #   grad_norms:  (M,)       = ||eik_grad||_2 per point
                #   eik_loss:    scalar     = lambda * mean((grad_norms - 1)^2)
                #
                # Optional 2nd-order (Laplacian):
                #   For each spatial dim d in {x, y, z}:
                #     autograd.grad(eik_grad[:,d], eik_pts) -> (M, 3)
                #     take column d -> d^2f/dx_d^2          -> (M,)
                #   laplacian = sum over d                   -> (M,)
                #   2nd_loss = lambda_2 * mean(laplacian^2)  -> scalar
                # =============================================================
                if do_eikonal:
                    # Sample M random unsupervised points
                    eik_pts = torch.empty(eikonal_num_samples, 3).uniform_(-1, 1).cuda()  # (M, 3)
                    eik_pts.requires_grad_(True)  # needed for autograd.grad

                    # Pick random latents from current batch (detached so eikonal
                    # loss only updates decoder weights, not latent codes)
                    random_idx = torch.randint(0, batch_vecs.shape[0], (eikonal_num_samples,))
                    eik_latents = batch_vecs[random_idx].detach()  # (M, L)

                    if do_cat_embedding:
                        eik_cat_vecs = batch_cat_vecs[random_idx].detach()  # (M, C)
                        eik_input = torch.cat([eik_cat_vecs, eik_latents, eik_pts], dim=1)  # (M, C+L+3)
                    else:
                        eik_input = torch.cat([eik_latents, eik_pts], dim=1)  # (M, L+3)

                    eik_sdf = decoder(eik_input)  # (M, 1)

                    # Compute spatial gradient: df/d(xyz) via autograd
                    # create_graph=True keeps the graph so we can backprop through this loss
                    eik_grad = torch.autograd.grad(
                        outputs=eik_sdf,       # (M, 1)
                        inputs=eik_pts,        # (M, 3)
                        grad_outputs=torch.ones_like(eik_sdf),
                        create_graph=True,
                        retain_graph=True,
                    )[0]  # (M, 3) = [df/dx, df/dy, df/dz]

                    # Eikonal loss: penalize ||grad|| != 1
                    # grad_norms: (M,), eik_loss: scalar
                    eik_loss = eikonal_lambda * torch.mean((eik_grad.norm(dim=1) - 1.0) ** 2)

                    # Optional 2nd-order: penalize non-zero Laplacian
                    if eikonal_second_order:
                        laplacian = 0.0
                        for d in range(3):
                            # d^2f/dx_d^2: differentiate eik_grad[:,d] w.r.t. eik_pts, take col d
                            grad2 = torch.autograd.grad(
                                outputs=eik_grad[:, d],       # (M,)
                                inputs=eik_pts,               # (M, 3)
                                grad_outputs=torch.ones_like(eik_grad[:, d]),
                                create_graph=True,
                                retain_graph=True,
                            )[0][:, d]  # (M,) — d^2f/dx_d^2
                            laplacian = laplacian + grad2  # accumulate: (M,)
                        eik_loss = eik_loss + eikonal_second_order_lambda * torch.mean(laplacian ** 2)

                    chunk_loss = chunk_loss + eik_loss

                # All losses for this sub-batch accumulated; backprop
                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            # =================================================================
            # Idea 3 — Contrastive Triplet Loss
            #
            # Computed AFTER the batch_split loop, on the GLOBAL latent
            # embedding table (not just the current batch).
            #
            # Tensor shapes (K = contrastive_triplets, e.g. 32; L = 256):
            #   Step 1 — Mine K triplets:
            #     anchors:   K indices (random shape from random category)
            #     positives: K indices (random shape from SAME category)
            #     negatives: K indices (random shape from DIFFERENT category)
            #
            #   Step 2 — Look up latent vectors:
            #     anchor_vecs: (K, L)  = lat_vecs(anchors)    e.g. (32, 256)
            #     pos_vecs:    (K, L)  = lat_vecs(positives)  e.g. (32, 256)
            #     neg_vecs:    (K, L)  = lat_vecs(negatives)  e.g. (32, 256)
            #
            #   Step 3 — Squared L2 distances:
            #     dist_pos: (K,) = sum((anchor - positive)^2, dim=1)
            #     dist_neg: (K,) = sum((anchor - negative)^2, dim=1)
            #
            #   Step 4 — Triplet loss with margin:
            #     triplet:  (K,) = clamp(dist_pos - dist_neg + margin, min=0)
            #     loss: scalar   = lambda * mean(triplet)
            #
            # The loss is 0 when dist_pos < dist_neg - margin (constraint satisfied).
            # Otherwise it's positive, pushing same-category codes closer and
            # different-category codes further apart.
            # =================================================================
            if do_contrastive:
                # Step 1: Mine K triplets
                anchors = []
                positives = []
                negatives = []
                for _ in range(contrastive_triplets):
                    anchor_cat = random.choice(contrastive_cat_ids)
                    anchor_idx = random.choice(contrastive_cat_to_indices[anchor_cat])
                    pos_idx = random.choice(contrastive_cat_to_indices[anchor_cat])
                    neg_cat = random.choice([c for c in contrastive_cat_ids if c != anchor_cat])
                    neg_idx = random.choice(contrastive_cat_to_indices[neg_cat])
                    anchors.append(anchor_idx)
                    positives.append(pos_idx)
                    negatives.append(neg_idx)

                # Step 2: Look up latent vectors from global embedding table
                anchor_vecs = lat_vecs(torch.tensor(anchors, dtype=torch.long).cuda())  # (K, L)
                pos_vecs = lat_vecs(torch.tensor(positives, dtype=torch.long).cuda())   # (K, L)
                neg_vecs = lat_vecs(torch.tensor(negatives, dtype=torch.long).cuda())   # (K, L)

                # Step 3: Squared L2 distances
                dist_pos = torch.sum((anchor_vecs - pos_vecs) ** 2, dim=1)  # (K,)
                dist_neg = torch.sum((anchor_vecs - neg_vecs) ** 2, dim=1)  # (K,)

                # Step 4: Triplet loss = clamp(dist_pos - dist_neg + margin, min=0)
                triplet_loss = torch.clamp(dist_pos - dist_neg + contrastive_margin, min=0.0)  # (K,)
                contrastive_loss = contrastive_lambda * torch.mean(triplet_loss)  # scalar
                contrastive_loss.backward()

                batch_loss += contrastive_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
