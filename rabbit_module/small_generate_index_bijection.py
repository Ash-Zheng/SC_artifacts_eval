import os
from time import time
import glob
import argparse

# tools for data preproc/loading
import torch
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

import time

import numpy as np
import rabbit

global access_list

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 
parser.add_argument('--table_idx', type=int, default=10)
parser.add_argument('--batch_num', type=int, default=65536)

def time_wrap():
    torch.cuda.synchronize()
    return time.time()


def get_col_kaggle(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:26]
    x_int = torch.cat(list(x_both.values())[26:39],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    for j in range(26):
        x_cat[j] = access_list[j][x_cat[j]]
    
    return y, x_cat, x_int

def get_col_avazu(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:23],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    for j in range(20):
        x_cat[j] = access_list[j][x_cat[j]]
    
    return y, x_cat, x_int


if __name__ == "__main__":
    args = parser.parse_args()

    LABEL_COLUMNS = ["label"]
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512)) # 2048  400000
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    dataset = args.dataset

    input_path = ""
    if dataset == "kaggle":
        input_path = os.path.join(BASE_DIR, "workspace/kaggle_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
    elif dataset == "avazu":
        input_path = os.path.join(BASE_DIR, "workspace/avazu_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 4)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 21)]
    elif dataset == "terabyte":
        input_path = os.path.join(BASE_DIR, "workspace/terabyte_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
    
    train_paths = glob.glob(os.path.join(input_path, "train", "*.parquet"))
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)

    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )

    if dataset == "kaggle": #[2, 3, 11, 15, 20]
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
    elif dataset == "terabyte": #[0, 9, 10, 19, 20, 21]
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
    elif dataset == "avazu": #[7, 8]
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_avazu, batch_size=None, pin_memory=False, num_workers=0
        )

    
    workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))
    embeddings = list(get_embedding_sizes(workflow).values())
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings] # embedding table size

    embedding_table_size = []  # embedding table size
    for emb in embeddings:
        embedding_table_size.append(emb[0])

    device = torch.device("cuda", 0)

    table_idx = args.table_idx
    print(embedding_table_size)  # total length of the table
    print(embedding_table_size[table_idx])  # total length of the table

    if dataset == "kaggle": #[2, 3, 11, 15, 20]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/kaggle/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 26
    elif dataset == "terabyte": #[0, 9, 10, 19, 20, 21]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/terabyte/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 26
    elif dataset == "avazu": #[7, 8]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 20

    train_iter = iter(train_dataloader)
    total_batch_num = len(train_dataloader)
    # emb_index = torch.load(input_file).to(device)
    emb_index = torch.load(input_file)

    length = embedding_table_size[table_idx]
    hot_idx = int(length*0.05)

    start = time_wrap()
    edge_list = []

    batch_num = args.batch_num # 70656

    idx = 0
    for i, inputBatch in enumerate(train_data_itrs):
        # if i % step == 0:
        x_both, y = inputBatch
        x_cat = list(x_both.values())[0:cat_num]
        
        x_cat_cpu = x_cat[table_idx].cpu()

        x_cat[table_idx] = emb_index[x_cat_cpu]
        x_cat[table_idx] = torch.clamp(x_cat[table_idx],min=hot_idx)-hot_idx

        batch_index = (x_cat[table_idx].view(-1)).to(torch.int).unique()

        edge_pairs = torch.combinations(batch_index[1:])
        edge_list.append(edge_pairs)

        # edge_pairs = torch.combinations(batch_index[1:]).transpose(0, 1)
        # edge_list = torch.cat((edge_list,edge_pairs),1)

        if i % 1024 == 0:
            tmp = time_wrap()
            print("batch:",i,"finish ratio:",round(i/batch_num,4), "total ratio:",round(i/total_batch_num,4), "time:",tmp-start)
        if i == batch_num:
            break
                
    end = time_wrap()
    print("total time for generate edge_list:", end - start)

    torch_edge = torch.cat(edge_list,0).transpose(0, 1).detach().contiguous()

    # output_file = "terabyte_small/terabyte_edge" + str(table_idx) + "_new.pt"
    # torch.save(torch_edge, output_file)

    del(edge_list)
    # print(torch_edge.shape)

    print("start reordering")
    start = time_wrap()
    new_edge_index = rabbit.generate_new_index(torch_edge, hot_idx)
    end = time_wrap()
    print("reordering time:", end - start)

    final_index = []
    for i in range(length):
        k = emb_index[i].item()
        if k > hot_idx and k-hot_idx < new_edge_index.shape[0]:
            k = new_edge_index[k-hot_idx]
        final_index.append(k)
    
    tensor_index = torch.tensor(final_index) 

    if dataset == "kaggle": #[2, 3, 11, 15, 20]
        output_file = "/workspace/SC_artifacts_eval/Access_Index/kaggle/access_index/access_index_" + str(table_idx) + "_new.pt"

    elif dataset == "avazu": #[7, 8]
        output_file = "/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_" + str(table_idx) + "_new.pt"

    elif dataset == "terabyte": #[0, 9, 10, 19, 20, 21]
        output_file = "/workspace/SC_artifacts_eval/Access_Index/terabyte/access_index/access_index_" + str(table_idx) + "_new.pt"
    
    torch.save(tensor_index, output_file)
    print("saved index bijection")
