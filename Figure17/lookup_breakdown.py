import os
from time import time
import glob

# tools for data preproc/loading
import torch
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

import multiprocessing as mp
import re
import numpy as np
import time
from tqdm import tqdm

import argparse
import sys
sys.path.append('/workspace/SC_artifacts_eval/models')
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from Efficient_TT.efficient_tt import Eff_TTEmbedding
from TTRec_kernel.tt_embedding import TTEmbeddingBag

from random_dataloader import in_memory_dataloader

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 
parser.add_argument('--setting', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4096)

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def get_col_kaggle(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:26]
    x_int = torch.cat(list(x_both.values())[26:39],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

def get_col_avazu(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:23],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

def get_pick(y):
    y_list = y.detach().cpu().numpy()

    neg_num = y_list.sum()
    pos_num = int(neg_num * 4) # sub sample 20% 

    pick_list = []
    pos_cnt = 0
    for idx in range(len(y_list)):
        if pos_cnt < pos_num and y_list[idx] == 0:
            pick_list.append(idx)
            pos_cnt += 1
        elif y_list[idx] == 1:
            pick_list.append(idx)
                
    pick = torch.tensor(pick_list, dtype=torch.long).squeeze().to(device)

    return pick

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    setting = args.setting
    batch_size = args.batch_size

    if setting == 0:
        exp = "TT-Rec"
        reordered = 0
    elif setting == 1:
        exp = "+Intermediate-Result-Reuse"
        reordered = 0
    elif setting == 2:
        exp = "+Index-Reordering"
        reordered = 1
    
    if reordered == 1:
        dataset += "_reordered"

    if dataset == "avazu" or dataset == "avazu_reordered":
        table_num = 20
        feature_size = 16
        table_length = [8, 4680, 7567, 27, 8380, 550, 36, 2512738, 6385037, 8165, 6, 5, 2621, 9, 10, 434, 5, 69, 173, 61]
        q_size = [2, 4, 2]
        tt_rank = [128, 128]
        threshold = 500000
    elif dataset == "kaggle" or dataset == "kaggle_reordered":
        table_num = 26
        feature_size = 16
        table_length = [1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633, 7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16, 272266, 105, 138045]
        q_size = [2, 4, 2]
        tt_rank = [128, 128]
        threshold = 500000
    elif dataset == "terabyte" or dataset == "terabyte_reordered":
        table_num = 26
        feature_size = 64
        table_length = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]
        q_size = [4, 4, 4]
        tt_rank = [128, 128]
        threshold = 200000

    device = 'cuda:0'
    train_iter = in_memory_dataloader(batch_size,0,dataset=dataset)
    offset = torch.tensor(range(batch_size)).to(device)  # int64

    emb_list = []
    tt_index = []
    idx = 0
    for n in table_length:
        if n > threshold:
            if exp == "TT-Rec":
                eff_emb = TTEmbeddingBag(
                    num_embeddings=n,
                    embedding_dim=feature_size,
                    tt_p_shapes=None,
                    tt_q_shapes=q_size,
                    tt_ranks=tt_rank,
                    sparse=True,
                    use_cache=False,
                    weight_dist="uniform"
                ).to(device)
            else:
                eff_emb = Eff_TTEmbedding(
                    num_embeddings=n,
                    embedding_dim=feature_size,
                    tt_p_shapes=None,
                    tt_q_shapes=q_size,
                    tt_ranks=tt_rank,
                    weight_dist="uniform"
                ).to(device)

            emb_list.append(eff_emb)
            tt_index.append(idx)
        idx += 1


    if exp == "TT-Rec":
        offset = torch.tensor(range(batch_size+1)).to(device)  # int64
    else:
        offset = None
    # warm up
    for i in range(100):
        label, sparse, dense = train_iter.next()
        k = 0
        for idx in tt_index:
            index = sparse[idx]
            eff_emb = emb_list[k]
            output = eff_emb(index, offset)
            k += 1

    iters = int(4096 / batch_size * 4096)
    t1 = time_wrap()
    for i in tqdm(range(iters)):
        label, sparse, dense = train_iter.next()
        k = 0
        for idx in tt_index:
            index = sparse[idx]
            eff_emb = emb_list[k]
            output = eff_emb(index, offset)
            k += 1
    t2 = time_wrap()

    total = t2 -t1
    throughput = iters/total

    print("time:{:.3f}, throughput:{:.3f}".format(total, throughput))
    print("Result saved to out.log")
    with open('out.log', 'a') as f:
        f.write("setting:{}, dataset:{}, batch_size:{}, throughput: {:.3f}\n".format(exp, dataset, batch_size, throughput))
    
