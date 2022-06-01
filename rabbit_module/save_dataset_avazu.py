import os
import glob
import torch
import numpy as np
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

# from pure_GPU_model import Eff_TT_DLRM_Net

import time
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--reorder', type=int, default=0)

global access_list

def get_col(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:22],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    for j in range(20):
        x_cat[j] = access_list[j][x_cat[j]]
    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

if __name__ == "__main__":
    CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 3)]
    CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 21)]
    LABEL_COLUMNS = ["label"]

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16384)) # 2048  400000
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    args = parser.parse_args()
    if_reorder = args.reorder
    input_path = "/workspace/SC_artifacts_eval/processed_data/workspace/avazu_workspace/output"

    train_paths = glob.glob(os.path.join(input_path, "train", "*.parquet"))
    valid_paths = glob.glob(os.path.join(input_path, "valid", "*.parquet"))
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)
    valid_data = nvt.Dataset(valid_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)

    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )

    train_dataloader = DLDataLoader(
        train_data_itrs, collate_fn=get_col, batch_size=None, pin_memory=False, num_workers=0
    )

    
    workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))
    embeddings = list(get_embedding_sizes(workflow).values())
    feature_size = 16
    embeddings = [[emb[0], min(feature_size, emb[1])] for emb in embeddings] # embedding table size

    embedding_table_size = []  # embedding table size
    for emb in embeddings:
        embedding_table_size.append(emb[0])

    print(embedding_table_size)

    device = 'cuda:0'
    length = len(embedding_table_size)
    access_list = []
    reordering_list = [7,8]
    for i in range(length):
        if i in reordering_list:
            if if_reorder == 1:
                x = torch.load("/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_"+ str(i) +"_new.pt").to(device)
            else:
                x = torch.load("/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_"+ str(i) +".pt").to(device)
        else:
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_" + str(i) + ".pt").to(device)
        access_list.append(x)

    bottom_mlp_list = [2, 512, 256, 64, feature_size]  # bottom mlp size
    num_fea = len(embeddings) + 1
    num_int = (num_fea * (num_fea - 1)) // 2 + feature_size
    upper_mlp_list = [num_int, 512, 256, 1]  # upper mlp size

    train_iter = iter(train_dataloader)
    label, sparse, dense = train_iter.next()  

    for i in tqdm(range(500-1)):
        tmp_label, tmp_sparse, tmp_dense = train_iter.next()  
        sparse = torch.cat((sparse, tmp_sparse), 1)
        label = torch.cat((label, tmp_label), 0)
        dense = torch.cat((dense, tmp_dense), 0)

    # ====================== save ========================
    if if_reorder == 1:
        torch.save(sparse, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/reordered_sparse.pt')
        torch.save(dense, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/reordered_dense.pt')
        torch.save(label, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/reordered_label.pt')
    else:
        torch.save(sparse, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/sparse.pt')
        torch.save(dense, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/dense.pt')
        torch.save(label, '/workspace/SC_artifacts_eval/Access_Index/avazu/training_data/label.pt')





            



    


