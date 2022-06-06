import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable

import sys
sys.path.append('/workspace/SC_artifacts_eval/models')
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from ELRec_data_parallel_emt import data_parallel_EMT
from ELRec_data_parallel_mlp import dlrm_hybrid

from random_dataloader import in_memory_dataloader
from unique_generator import unique_generator

import numpy as np
import random
from tqdm import tqdm
import argparse
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch_scatter import scatter, segment_coo

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 
parser.add_argument('--nDev', type=int, default=1)

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad != None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        # else:
        #     dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        #     param.data /= size

def average_emt_gradients(emt, nTable):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for i in range(nTable):
        if emt.emt_tag[i]>0:
            dist.all_reduce_coalesced([emt.emt[i].tt_cores[0].data,emt.emt[i].tt_cores[1].data,emt.emt[i].tt_cores[2].data], op=dist.ReduceOp.AVG)

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def init_process(rank, size, dataset_name, backend='nccl'):
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

    # kaggle avazu terabyte kaggle_reordered avazu_reordered terabyte_reordered
    dataset = dataset_name + "_reordered"

    if dataset == "avazu" or dataset == "avazu_reordered":
        table_num = 20
        feature_size = 16
        dense_num = 2
        top_num = 226
        table_length = [8, 4680, 7567, 27, 8380, 550, 36, 2512738, 6385037, 8165, 6, 5, 2621, 9, 10, 434, 5, 69, 173, 61]
    elif dataset == "kaggle" or dataset == "kaggle_reordered":
        table_num = 26
        feature_size = 16
        dense_num = 13
        top_num = 367
        table_length = [1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633, 7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16, 272266, 105, 138045]
    elif dataset == "terabyte" or dataset == "terabyte_reordered":
        table_num = 26
        feature_size = 64
        dense_num = 13
        top_num = 415
        table_length = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]

    # replicate_length = [0 for _ in range(26)]

    nDev = size
    batch_size = 4096
    total_batch_size = batch_size * nDev
    train_iter = in_memory_dataloader(total_batch_size,rank,dataset=dataset)
    unique_gen = unique_generator(batch_size=batch_size, nDev=nDev, table_num=table_num, rank=rank, dataloader=train_iter, generate=False, dataset=dataset)
    
    emt = data_parallel_EMT(replicate_length=table_length, feature_dim=feature_size, device=rank, batch_size=batch_size)
    dlrm = dlrm_hybrid(
        list_bot=[dense_num, 512, 256, 64, feature_size],
        list_top=[top_num, 512, 256, 1],
        device=rank,
    )

    distributed_dlrm = DDP(
        module=dlrm,
        device_ids=[rank],
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )

    optimizer = torch.optim.SGD(distributed_dlrm.parameters(), lr=0.1)
    optimizer_rep_emb = torch.optim.SGD(emt.emt.parameters(), lr=0.1)

    loss_fn = torch.nn.BCELoss(reduction="mean")
    max_length = unique_gen.max_length()

    index_list = []
    receive_emb_list = []
    for i in range(table_num):
        if table_length[i] <= 1000000:
            receive_emb_list.append(torch.zeros((max_length[i],feature_size),device=rank))
            index_list.append(i)

    print("start train:{}".format(rank))

    dist.barrier()

    # print(emt.emt[2].tt_cores)
    if rank == 0:
        start = time_wrap()
        for epoch in tqdm(range(1000)):
            run(train_iter, unique_gen, emt, batch_size, feature_size, nDev, table_num, rank, optimizer, optimizer_rep_emb, distributed_dlrm, loss_fn, epoch, receive_emb_list, index_list)
        end = time_wrap()

        total_time = end-start
        throughput = nDev*1000/(end-start)

        print("time:{:.3f}, throughput:{:.3f}".format(total_time, throughput))
        print("Result saved to out.log")
        with open('out.log', 'a') as f:
            f.write("ELRec, {}, {}GPU, throughput: {:.3f}\n".format(dataset_name, nDev, throughput))
    else:
        for epoch in range(1000):
            run(train_iter, unique_gen, emt, batch_size, feature_size, nDev, table_num, rank, optimizer, optimizer_rep_emb, distributed_dlrm, loss_fn, epoch, receive_emb_list, index_list)
    # dist.barrier()
    # print("finish init:{}".format(rank))
    # # =========================== init finished ============================

    
def run(train_iter, unique_gen, emt, batch_size, feature_dim, nDev, nTable, rank, optimizer, optimizer_rep_emb, dlrm, loss_fn, epoch, receive_emb_list, index_list):
    label, sparse, dense = train_iter.next()
    unique, inverse = unique_gen.next()

    sparse_feature = emt.lookup(sparse[:,rank*batch_size:(rank+1)*batch_size])

    optimizer.zero_grad()
    output = dlrm(dense[rank*batch_size:(rank+1)*batch_size], sparse_feature)  
    loss = loss_fn(output, label[rank*batch_size:(rank+1)*batch_size])  
    loss.backward()      
    optimizer.step() # update data_parallel mlp

    # local reduce and all reduce uncompressed emt gradient
    cnt = 0
    for i in index_list:
        scatter(sparse_feature[i].grad, inverse[:,i][rank*batch_size:(rank+1)*batch_size], dim=0, out=receive_emb_list[cnt],reduce="sum")
        cnt += 1
    dist.all_reduce_coalesced(receive_emb_list)

    # update uncompressed emt gradient
    cnt = 0
    for i in index_list:
        segment_coo(receive_emb_list[cnt][0:len(unique[i])] * 0.1, unique[i], out=emt.emt[i].weight.data, reduce="sum")
        receive_emb_list[cnt].zero_()
        cnt += 1

    # all_reduce compressed emt parameter
    if nDev > 1:
        average_emt_gradients(emt, nTable)

    # dist.barrier()


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    size = args.nDev
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, dataset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
