#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


# =============================================================================
# Idea 4 — Category Code + Instance Residual (inference helper)
#
# Constructs decoder input by expanding and concatenating components.
# The decoder's external interface always expects [... | xyz(3)] with the
# last 3 columns being raw xyz coordinates.
#
# Tensor shape flow (with category embedding, L=256, C=64, N=100 query pts):
#   category_embedding: (1, 64)    — frozen, looked up from trained table
#   latent_vector:      (1, 256)   — optimized during reconstruction
#   queries:            (100, 3)   — 3D query points
#
#   cat_repeat = category_embedding.expand(100, -1)  -> (100, 64)
#   lat_repeat = latent_vector.expand(100, -1)       -> (100, 256)
#   inputs = cat([cat_repeat, lat_repeat, queries])   -> (100, 323)
#   sdf = decoder(inputs)                             -> (100, 1)
#
# Without category embedding (baseline, L=256):
#   lat_repeat = latent_vector.expand(100, -1)       -> (100, 256)
#   inputs = cat([lat_repeat, queries])               -> (100, 259)
#   sdf = decoder(inputs)                             -> (100, 1)
#
# Demo:
#   dec = Decoder(320, [512,512])  # 320 = 256 + 64
#   sdf = decode_sdf(dec, torch.randn(1,256), torch.randn(100,3),
#                    category_embedding=torch.randn(1,64))
#   assert sdf.shape == (100, 1)
# =============================================================================
def decode_sdf(decoder, latent_vector, queries, category_embedding=None):
    num_samples = queries.shape[0]  # N query points

    parts = []
    if category_embedding is not None:
        # (1, C) -> (N, C) by broadcasting along batch dim
        cat_repeat = category_embedding.expand(num_samples, -1)
        parts.append(cat_repeat)
    if latent_vector is not None:
        # (1, L) -> (N, L) by broadcasting along batch dim
        latent_repeat = latent_vector.expand(num_samples, -1)
        parts.append(latent_repeat)
    parts.append(queries)  # (N, 3)

    # Concatenate: (N, C+L+3) or (N, L+3) without category
    inputs = torch.cat(parts, 1)

    sdf = decoder(inputs)  # (N, 1)

    return sdf
