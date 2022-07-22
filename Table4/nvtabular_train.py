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

from ELRec_dlrm import ELRec_DLRM_Net
from random_dataloader import in_memory_dataloader_cpu

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 

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
    dlrm = ELRec_DLRM_Net(
        feature_size, # sparse feature size
        table_length,
        [dense_num, 512, 256, 64, feature_size],
        [top_num, 512, 256, 1],
        device,
        128
    )

    dlrm = dlrm.to(device)
    
    if not os.path.exists("trained_models/ELRec_{}.pkl".format(dataset)):
        learning_rate = 0.1 # default 0.1
        parameters = dlrm.parameters()
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        loss_fn = torch.nn.BCELoss(reduction="mean")
        best_acc = 0
        num_epoch = 5
        for i in range(num_epoch):
            train_iter = iter(train_dataloader)
            batch_num = len(train_dataloader)
            print("training epoch:", i, "number of batchs:", batch_num)
            for j in range(batch_num-1):
                label, sparse, dense = train_iter.next()
                z = dlrm(dense, sparse)
                E = loss_fn(z, label)

                if j % 1000 == 0:
                    L = E.detach().cpu().item()
                    print("batch:",i,"loss:",L)

                optimizer.zero_grad()
                E.backward()
                optimizer.step()

            test_iter = iter(valid_dataloader)
            test_batch_num = len(valid_dataloader)

            test_accu = 0
            test_samp = 0
            dlrm.eval()
            for i in range(test_batch_num-1):
                label, sparse, dense = test_iter.next()

                z = dlrm(dense, sparse)
                torch.cuda.synchronize()

                S_test = z.detach().cpu().numpy()  # numpy array
                T_test = label.detach().cpu().numpy()  # numpy array

                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                test_accu += A_test
                test_samp += mbs_test

            acc = test_accu/test_samp
            if acc > best_acc:
                best_acc = acc
                torch.save(dlrm, "trained_models/ELRec_{}.pkl".format(dataset))
            print("acc:",acc,"best acc:",best_acc)
    
    else:
        dlrm = torch.load("trained_models/ELRec_{}.pkl".format(dataset))   
        dlrm = dlrm.to(device)

        test_iter = iter(valid_dataloader)
        test_batch_num = len(valid_dataloader)

        test_accu = 0
        test_samp = 0
        dlrm.eval()
        for i in range(test_batch_num-1):
            label, sparse, dense = test_iter.next()

            z = dlrm(dense, sparse)
            torch.cuda.synchronize()

            S_test = z.detach().cpu().numpy()  # numpy array
            T_test = label.detach().cpu().numpy()  # numpy array

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test

        acc = test_accu/test_samp
        
        print("acc:", acc)

        print("Result saved to out.log")
        with open('out.log', 'a') as f:
            f.write("ELRec, {}, acc: {:.5f}\n".format(dataset, acc))