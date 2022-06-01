# record embedding table access pattern, and store the order list

import os
from time import time
import glob

# tools for data preproc/loading
import torch
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

import time

import argparse
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

    # for j in range(26):
    #     x_cat[j] = access_list[j][x_cat[j]]
    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

def get_col_avazu(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:23],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    # for j in range(26):
    #     x_cat[j] = access_list[j][x_cat[j]]
    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

if __name__ == "__main__":

    LABEL_COLUMNS = ["label"]
    # BASE_DIR = "/workspace/PipeDLRM/Nvidia-Merlin/"
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096)) # 2048  400000
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    # /home/ubuntu/zheng/PipeDLRM/Nvidia-Merlin/workspace/terabyte_new_full_workspace/output/train
    args = parser.parse_args()
    dataset = args.dataset
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
    
    train_data_itrs = TorchAsyncItr(
        train_data,
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
    elif dataset == "avazu":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_avazu, batch_size=None, pin_memory=False, num_workers=0
        )
    elif dataset == "terabyte":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=0
        )
    
    workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))
    embeddings = list(get_embedding_sizes(workflow).values())
    # We limit the output dimension to 16
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings] # embedding table size

    embedding_table_size = []  # embedding table size
    for emb in embeddings:
        embedding_table_size.append(emb[0])

    print(embedding_table_size)

    length = len(embedding_table_size)
    access_list = []
    for size in embedding_table_size:
        access_list.append(torch.zeros(size,dtype=torch.long)) 
    

    train_iter = iter(train_dataloader)
    batch_num = len(train_dataloader)

    start = time_wrap()
    src = torch.ones(4096,dtype=int) # batch size
    for i in range(0,batch_num):
        y, x_cat, x_int = train_iter.next()

        for j in range(length):
            access_list[j].scatter_add_(0,x_cat[j].cpu().squeeze(),src)
        
        if i % 1024 == 0:
            end = time_wrap()
            print("batch:",i,"ratio:",round(i/batch_num,4)," time:",end-start)

    end = time_wrap()
    print("time:",end-start)
    
    # if dataset == "kaggle_ad":
    #     for i in range(length):
    #         torch.save(access_list[i], '/workspace/PipeDLRM/Access_Index/kaggle/kaggle_access_record/kaggle_access_'+str(i)+'.pt')
    # elif dataset == "avazu":
    #     for i in range(length):
    #         torch.save(access_list[i], '/workspace/PipeDLRM/Access_Index/avazu/access_record/avazu_access_'+str(i)+'.pt')
    # elif dataset == "terabyte":
    #     for i in range(length):
    #         torch.save(access_list[i], '/workspace/PipeDLRM/Access_Index/terabyte/access_record/terabyte_access_'+str(i)+'.pt')


    if dataset == "kaggle":
        for i in range(length):
            torch.save(access_list[i], '/workspace/SC_artifacts_eval/Access_Index/kaggle/access_record/access_'+str(i)+'.pt')
    elif dataset == "avazu":
        for i in range(length):
            torch.save(access_list[i], '/workspace/SC_artifacts_eval/Access_Index/avazu/access_record/access_'+str(i)+'.pt')
    elif dataset == "terabyte":
        for i in range(length):
            torch.save(access_list[i], '/workspace/SC_artifacts_eval/Access_Index/terabyte/access_record/access_'+str(i)+'.pt')