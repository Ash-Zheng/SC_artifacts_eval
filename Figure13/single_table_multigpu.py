import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable

import sys
sys.path.append('/workspace/SC_artifacts_eval/models')
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from Efficient_TT.efficient_tt import Eff_TTEmbedding
from ELRec_data_parallel_emt import data_parallel_EMT
from random_dataloader import in_memory_dataloader
from unique_generator import unique_generator

import numpy as np
import random
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch_scatter import scatter, segment_coo

import time
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nDev', type=int, default=1)

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def init_process(rank, size, backend='nccl'):
    torch.manual_seed(123)
    np.random.seed(1234)
    random.seed(1234)


    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("start init:{}".format(rank))

    dist.barrier()
    print("finish init:{}".format(rank))

    dataset = "terabyte_reordered"
    table_length = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]
    t_id = 19 # 19
    n = 40000000 # 40000000
    feature_size = 128 # 128
    q_size = [4, 4, 8] # [8, 4, 4]
    tt_rank = [64, 64] # [64, 64]
    device = 'cuda:0'

    nDev = size
    batch_size = 4096
    total_batch_size = batch_size
    train_iter = in_memory_dataloader(total_batch_size,rank,dataset=dataset)
    unique_gen = unique_generator(batch_size=batch_size, nDev=1, table_num=26, rank=rank, dataloader=train_iter, generate=False, dataset=dataset)
    offset = torch.tensor(range(batch_size)).to(rank)  # int64

    eff_emb = Eff_TTEmbedding(
        num_embeddings=n,
        embedding_dim=feature_size,
        tt_p_shapes=None,
        tt_q_shapes=q_size,
        tt_ranks=tt_rank,
        weight_dist="uniform",
        device=rank
    ).to(rank)

    loss_fn = torch.nn.BCELoss(reduction="mean")
    fc_layer = torch.nn.Linear(feature_size, 1).to(rank)

    iters = 5000
    if rank == 0:
        start = time_wrap()
        for i in tqdm(range(iters)):
            label, sparse, dense = train_iter.next()
            unique, inverse = unique_gen.next()
            emb_1 = eff_emb(sparse[t_id], offset, unique[t_id], inverse[:,t_id])
            output1 = fc_layer(emb_1)
            Loss1 = loss_fn(label, output1)
            Loss1.backward()
            dist.all_reduce(eff_emb.tt_cores[0].data, op=dist.ReduceOp.SUM,async_op=True)
            dist.all_reduce(eff_emb.tt_cores[1].data, op=dist.ReduceOp.SUM,async_op=True)
            dist.all_reduce(eff_emb.tt_cores[2].data, op=dist.ReduceOp.SUM,async_op=True)
        end = time_wrap()

        total_time = end-start
        throughput = nDev*iters/(end-start)

        print("time:{:.3f}, throughput:{:.3f}".format(total_time, throughput))
        print("Result saved to out.log")
        with open('out.log', 'a') as f:
            f.write("ELRec_single_large_table, {}GPU, throughput: {:.3f}\n".format(nDev, throughput))

    else:
        for i in range(iters):
            label, sparse, dense = train_iter.next()
            unique, inverse = unique_gen.next()
            emb_1 = eff_emb(sparse[t_id], offset, unique[t_id], inverse[:,t_id])
            output1 = fc_layer(emb_1)
            Loss1 = loss_fn(label, output1)
            Loss1.backward()
            dist.all_reduce(eff_emb.tt_cores[0].data, op=dist.ReduceOp.SUM,async_op=True)
            dist.all_reduce(eff_emb.tt_cores[1].data, op=dist.ReduceOp.SUM,async_op=True)
            dist.all_reduce(eff_emb.tt_cores[2].data, op=dist.ReduceOp.SUM,async_op=True)


if __name__ == "__main__":
    args = parser.parse_args()
    size = args.nDev
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()