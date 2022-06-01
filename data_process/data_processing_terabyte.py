import os
from os import path
import glob

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from nvtabular.utils import device_mem_size, get_rmm_size


if __name__ == "__main__":
    # BASE_DIR = "/home/ubuntu/zheng/PipeDLRM/Nvidia-Merlin"
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data/terabyte"
    INPUT_PATH  = "/workspace/SC_artifacts_eval/dlrm_dataset/terabyte"
    OUTPUT_PATH  = os.path.join(BASE_DIR, "processed")
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
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
    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]
    train_cols = ["label"] + cont_names + cat_names

    dtypes = {}
    dtypes["label"] = np.int32
    for x in cont_names:
        dtypes[x] = np.int32
    for x in cat_names:
        dtypes[x] = "hex"

    # file_list = glob.glob(os.path.join(INPUT_PATH, "day_*"))
    file_list = ['/workspace/SC_artifacts_eval/dlrm_dataset/terabyte/day_0', '/workspace/SC_artifacts_eval/dlrm_dataset/terabyte/day_1', '/workspace/SC_artifacts_eval/dlrm_dataset/terabyte/day_2']

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
        os.path.join(OUTPUT_PATH, "terabyte"),
        preserve_files=True,
    )   