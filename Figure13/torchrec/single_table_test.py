#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from typing import Iterator, List

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper

from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
)
from tqdm import tqdm
import time

SINGLE_CAT_NAMES = ["t_cat_0"]

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/modules:dlrm_train
    from modules.dlrm_train import DLRMTrain, Single_Table_Train
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import (  # noqa F811
        get_dataloader,
        STAGES,
    )
    from .modules.dlrm_train import DLRMTrain  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 1  # Number of stages in TrainPipelineSparseDist.


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="batch size to use for training"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=1100,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=100,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=100,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_kaggle",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=40_000_000, # 40_000_000
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        # default=None,
        default="40000000", # 40000000
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
   
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default="/home/yuke_wang/zheng/torchrec/processed",
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=bool,
        default=False,
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--single_table",
        type=bool,
        default=True,
    )
    parser.set_defaults(pin_memory=None)
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        args.num_embeddings = None

    train_dataloader = get_dataloader(args, backend, "train")

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            if args.num_embeddings is None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(SINGLE_CAT_NAMES)
    ]

    # =============================== Auto Generate ==============================
    train_model = Single_Table_Train(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        feature_dim=args.embedding_dim,
        dense_device=device,
    )

    if dist.get_world_size() == 2:
        plan = {
            'model.sparse_arch.embedding_bag_collection': 
            {'t_t_cat_0': 
                ParameterSharding(sharding_type='column_wise', compute_kernel='fused', ranks=[1, 0, 1, 0], sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(shard_offsets=[0, 0], shard_sizes=[40000000, 32], placement="rank:1/cuda:1"), 
                        ShardMetadata(shard_offsets=[0, 32], shard_sizes=[40000000, 32], placement="rank:0/cuda:0"), 
                        ShardMetadata(shard_offsets=[0, 64], shard_sizes=[40000000, 32], placement="rank:1/cuda:1"), 
                        ShardMetadata(shard_offsets=[0, 96], shard_sizes=[40000000, 32], placement="rank:0/cuda:0")
                    ]))}
            }
    elif dist.get_world_size() == 4:
        plan = {
            'model.sparse_arch.embedding_bag_collection': 
            {'t_t_cat_0': 
                ParameterSharding(sharding_type='column_wise', compute_kernel='fused', ranks=[3, 2, 1, 0], sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(shard_offsets=[0, 0], shard_sizes=[40000000, 32], placement="rank:3/cuda:3"), 
                        ShardMetadata(shard_offsets=[0, 32], shard_sizes=[40000000, 32], placement="rank:2/cuda:2"), 
                        ShardMetadata(shard_offsets=[0, 64], shard_sizes=[40000000, 32], placement="rank:1/cuda:1"), 
                        ShardMetadata(shard_offsets=[0, 96], shard_sizes=[40000000, 32], placement="rank:0/cuda:0")]))}
            }

    # model = DistributedModelParallel(
    #     module=train_model,
    #     device=device,
    # )

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=ShardingPlan(plan)
        # plan=plan
    )

    # print(model.plan)

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=args.learning_rate),
    )

    batch = next(iter(train_dataloader))

    batch = batch.to(rank)

    print("rank:",rank, "warm up")
    if rank == 0:
        for i in range(500):
            loss, (d_loss, logits, labels) = model.forward(batch)
            torch.sum(loss, dim=0).backward()
    else:
        for i in range(500):
            loss, (d_loss, logits, labels) = model.forward(batch)
            torch.sum(loss, dim=0).backward()

    iters = 1000
    if rank == 0:
        print("start train:")
        start_time = time.time()
        for i in tqdm(range(iters)):
            loss, (d_loss, logits, labels) = model.forward(batch)
            torch.sum(loss, dim=0).backward()
        end_time = time.time()

        total = end_time - start_time
        throughput = dist.get_world_size() * iters/total

        print("time:{:.3f}, throughput:{:.3f}".format(total, throughput))
        print("Result saved to out.log")
        with open('out.log', 'a') as f:
            f.write("TorchRec_single_large_table, {}GPU, throughput: {:.3f}\n".format(dist.get_world_size(), throughput))
    else:
        for i in range(iters):
            loss, (d_loss, logits, labels) = model.forward(batch)
            torch.sum(loss, dim=0).backward()

if __name__ == "__main__":
    main(sys.argv[1:])
