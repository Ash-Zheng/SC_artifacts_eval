import os
from time import time
import glob

# tools for data preproc/loading
import torch

import multiprocessing as mp
import re
import numpy as np
import time
from tqdm import tqdm

import argparse
import sys
sys.path.append('/workspace/SC_artifacts_eval/models')
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from TT_dlrm import TT_DLRM_Net
from random_dataloader import in_memory_dataloader

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset

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
    num_iters = 1000

    device = 'cuda:0'

    dlrm = TT_DLRM_Net(
        feature_size, # sparse feature size
        table_length,
        [dense_num, 512, 256, 64, feature_size],
        [top_num, 512, 256, 1],
        device,
        128
    )

    learning_rate = 0.1 # default 0.1

    dlrm = dlrm.to(device)

    parameters = dlrm.parameters()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    start = time_wrap()
    for i in tqdm(range(num_iters)):
        label, sparse, dense = train_iter.next()

        z = dlrm(dense, sparse)
        E = loss_fn(z, label)

        optimizer.zero_grad()
        E.backward()
        optimizer.step()
    end = time_wrap()
    print("time:",end-start)
    
    print("Result saved to out.log")
    with open('out.log', 'a') as f:
        f.write("TTRec, {}, time: {:.3f}\n".format(dataset, end-start))

