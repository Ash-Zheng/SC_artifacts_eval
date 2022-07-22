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

from dlrm_cpu import DLRM_Net, DLRM_Net_terabyte
from random_dataloader import in_memory_dataloader_cpu

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="terabyte") 

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

    if os.path.exists('loss_record/DLRM.txt'):
        print("DLRM.txt exists, skip training")
        exit()

    if dataset == "avazu" or dataset == "avazu_reordered":
        table_num = 20
        feature_size = 16
        dense_num = 2
        top_num = 226
        table_length = [8, 4680, 7567, 27, 8380, 550, 36, 2512738, 6385037, 8165, 6, 5, 2621, 9, 10, 434, 5, 69, 173, 61]
        num_epoch = 5
    elif dataset == "kaggle" or dataset == "kaggle_reordered":
        table_num = 26
        feature_size = 16
        dense_num = 13
        top_num = 367
        table_length = [1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633, 7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16, 272266, 105, 138045]
        num_epoch = 5
    elif dataset == "terabyte" or dataset == "terabyte_reordered":
        table_num = 26
        feature_size = 64
        dense_num = 13
        top_num = 415
        table_length = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]
        num_epoch = 1
    # ============================================================== Data Profile ==================================================================

    LABEL_COLUMNS = ["label"]
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096)) # 2048  400000
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    input_path = ""
    if dataset == "kaggle":
        input_path = os.path.join(BASE_DIR, "workspace/kaggle_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
    elif dataset == "avazu":
        input_path = os.path.join(BASE_DIR, "workspace/avazu_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 3)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 21)]
    elif dataset == "terabyte":
        input_path = os.path.join(BASE_DIR, "workspace/terabyte_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]

    train_paths = glob.glob(os.path.join(input_path, "train", "*.parquet"))
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)
    valid_paths = glob.glob(os.path.join(input_path, "valid", "*.parquet"))
    valid_data = nvt.Dataset(valid_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)
    
    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )

    valid_data_itrs = TorchAsyncItr(
        valid_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )


    if dataset == "kaggle":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
        valid_dataloader = DLDataLoader(
            valid_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
    elif dataset == "avazu":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_avazu, batch_size=None, pin_memory=False, num_workers=0
        )
        valid_dataloader = DLDataLoader(
            valid_data_itrs, collate_fn=get_col_avazu, batch_size=None, pin_memory=False, num_workers=0
        )
    elif dataset == "terabyte":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
        valid_dataloader = DLDataLoader(
            valid_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )

    batch_size = 4096
    device = 'cuda:0'

    if dataset == "terabyte":
        dlrm = DLRM_Net_terabyte(
            feature_size, # sparse feature size
            table_length,
            [dense_num, 512, 256, 64, feature_size],
            [top_num, 512, 256, 1],
            'dot',
            device
        )
    else:
        dlrm = DLRM_Net(
            feature_size, # sparse feature size
            table_length,
            [dense_num, 512, 256, 64, feature_size],
            [top_num, 512, 256, 1],
            'dot',
            device
        )

    # dlrm = dlrm.to(device)
    
    learning_rate = 0.1 # default 0.1
    parameters = dlrm.parameters()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    train_iter = iter(train_dataloader)
    batch_num = 10000

    if not os.path.exists('loss_record/DLRM.txt'):
        with open('loss_record/DLRM.txt', 'w') as f:
            for j in range(10000):
                label, sparse, dense = train_iter.next()
                sparse = sparse.to('cpu')

                if dataset == "terabyte":
                    pick = get_pick(label).to(device)
                    label = label[pick]
                    z = dlrm(dense, sparse, pick)
                else:
                    z = dlrm(dense, sparse)

                E = loss_fn(z, label)
                L = E.detach().cpu().item()
                if j % 200 == 0:
                    print("batch:", j, "loss:", L)
                f.write("{:.5f}\n".format(L))

                optimizer.zero_grad()
                E.backward()
                optimizer.step()