import os
from os import path
import glob

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from nvtabular.utils import device_mem_size, get_rmm_size


if __name__ == "__main__":
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data/avazu"
    INPUT_PATH  = BASE_DIR
    OUTPUT_PATH  = os.path.join(BASE_DIR, "processed")
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    frac_size = 0.10

    # Optional
    cluster = None  # Connect to existing cluster if desired
    if cluster is None:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
            rmm_pool_size=get_rmm_size(0.8 * device_mem_size()),
            local_directory=os.path.join(OUTPUT_PATH, "dask-space"),
        )
    client = Client(cluster)

    # Specify column names
    cont_names = ["I" + str(x) for x in range(1, 4)]
    cat_names = ["C" + str(x) for x in range(1, 21)]
    hex_names = ["C" + str(x) for x in range(2, 11)]
    train_cols = ["label"] + cont_names + cat_names

    # label  I1 I2  I3  C1  C2       C3       C4       C5       C6      C7        C8       C9       C10    C11  C12  C13  C14 C15 C16 C17 C18 C19 C20
    #  0     21 00 1005 0 1fbe01fe f3845767 28905ebd ecad2386 7801e8d9 07d7df22 a99f214a ddd2926e 44956a24  1    2  15706 320 50 1722  0  35  -1  79

    dtypes = {}
    dtypes["label"] = np.int32
    for x in cont_names:
        dtypes[x] = np.int32
    for x in cat_names:
        if x in hex_names:
            dtypes[x] = "hex"
        else:
            dtypes[x] = np.int32

    # print(dtypes)

    print("train data")
    file_list = glob.glob(os.path.join(INPUT_PATH, "train_subset.txt"))
    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        names=train_cols,
        part_mem_fraction=frac_size,
        sep="\t",
        dtypes=dtypes,
        client=client,
    )

    dataset.to_parquet(
        os.path.join(OUTPUT_PATH, "avazu_train"),
        preserve_files=True,
    )


    print("val data")
    file_list = glob.glob(os.path.join(INPUT_PATH, "val_subset.txt"))
    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        names=train_cols,
        part_mem_fraction=frac_size,
        sep="\t",
        dtypes=dtypes,
        client=client,
    )

    dataset.to_parquet(
        os.path.join(OUTPUT_PATH, "avazu_val"),
        preserve_files=True,
    )
    
