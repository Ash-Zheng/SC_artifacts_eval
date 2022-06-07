import os
import glob
import torch

from cpu_gpu_model import EMB_Tables, MLP_Layers, TT_Tables
import time

import torch.multiprocessing as mp

from torch.utils.cpp_extension import load
cache_sync_cuda = load(name="cache_sync_cuda", sources=[
    "/workspace/SC_artifacts_eval/Figure16/cache_sync/cache_sync_wrap.cpp", 
    "/workspace/SC_artifacts_eval/Figure16/cache_sync/cache_sync_kernel.cu", 
    ], verbose=False)

import argparse
from tqdm import tqdm
import sys
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from random_dataloader import in_memory_dataloader
from unique_generator import unique_generator_cpu, unique_generator

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 


def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def generate_unique(table_num, sparse):
    unique_list = []
    inverse_list = []
    for i in range(table_num):
        unique, inverse = sparse[i].unique(sorted=True, return_inverse=True)
        inverse_list.append(inverse)
        unique_list.append(unique)
    return unique_list, inverse_list

def mlp_process(num_iters, warm_up_iters, emb_q, grad_q, MLP_layers, loss_fn, optimizer, train_iter):
    for i in range(0, warm_up_iters):
        emb = emb_q.get()
        label, sparse, dense = train_iter.next()
        emb.requires_grad = True

        if i > 0:
            for idx in range(len(emb)):
                cache_sync_cuda.cache_sync(sparse[idx], last_sparse[idx], emb[idx], last_emb[idx])

        z = MLP_layers(dense, emb)
        E = loss_fn(z, label)

        optimizer.zero_grad()
        E.backward()
        optimizer.step()

        last_emb = emb
        last_sparse = sparse
        grad_q.put(emb.grad)

    for i in tqdm(range(0, num_iters)):
        emb = emb_q.get()
        label, sparse, dense = train_iter.next()
        emb.requires_grad = True

        if i > 0:
            for idx in range(len(emb)):
                cache_sync_cuda.cache_sync(sparse[idx], last_sparse[idx], emb[idx], last_emb[idx])
        
        z = MLP_layers(dense, emb)
        E = loss_fn(z, label)

        optimizer.zero_grad()
        E.backward()
        optimizer.step()

        last_emb = emb
        last_sparse = sparse
        grad_q.put(emb.grad)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset + "_reordered"

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

    batch_size = 4096
    train_iter = in_memory_dataloader(batch_size, 0, dataset=dataset)
    train_iter_local = in_memory_dataloader(batch_size, 0, dataset=dataset)
    train_iter_update = in_memory_dataloader(batch_size, 0, dataset=dataset)
    unique_gen_gpu = unique_generator(batch_size=batch_size, nDev=1, table_num=table_num, rank=0, dataloader=train_iter, generate=False, dataset=dataset)

    max_length = unique_gen_gpu.max_length()
    receive_emb_list = [torch.zeros((max_length[i],feature_size),device=0) for i in range(table_num)]
    
    device = 'cuda:0'

    EMB_tables = EMB_Tables(
        feature_size,
        table_length,
        device,
    )

    MLP_layers = MLP_Layers(
        [dense_num, 512, 256, 64, feature_size],
        [top_num, 512, 256, 1],
        'dot',
        device
    )

    learning_rate = 0.1 # default 0.1
    parameters = MLP_layers.parameters()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    torch.multiprocessing.set_start_method('spawn')
    emb_q = mp.Queue()
    grad_q = mp.Queue()

    num_iters = 1000
    warm_up_iters = 200
    p1 = mp.Process(target=mlp_process, args=(num_iters, warm_up_iters, emb_q, grad_q, MLP_layers, loss_fn, optimizer, train_iter))

    print("start train")
    p1.start()
    

    unique_list = [[],[]]
    inverse_list = [[],[]]

    start = 0
    send_cnt = 0
    recv_cnt = 0
    while recv_cnt < num_iters + warm_up_iters:
        if recv_cnt == warm_up_iters:
            start = time_wrap()

        if send_cnt <= recv_cnt + 1:
            _, sparse, _ = train_iter_local.next()
            emb = EMB_tables.unique_get(sparse)
            stacke_emb = torch.stack(emb,dim=0)

            unique, inverse = generate_unique(table_num, sparse)
            save_idx = send_cnt % 2
            unique_list[save_idx] = unique
            inverse_list[save_idx] = inverse
            
            emb_q.put(stacke_emb)
            send_cnt += 1
        else:
            grad = grad_q.get()
            _, sparse, _ = train_iter_update.next()

            get_id = recv_cnt % 2
            unique = unique_list[get_id]
            inverse = inverse_list[get_id]
            EMB_tables.scatter_update_list(inverse, unique, grad, receive_emb_list, learning_rate)

            recv_cnt += 1
            del grad

    p1.join()

    end = time_wrap()
    print("time:",end-start)
    print("Result saved to out.log")
    with open('out.log', 'a') as f:
        f.write("EL-Rec (pipeline), {}, training_time:{:.3f}, throughput:{:.3f}\n".format(args.dataset, end-start, num_iters/(end-start)))

   