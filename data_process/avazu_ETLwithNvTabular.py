import os
import re
import shutil
import warnings

# External Dependencies
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm

# NVTabular
import nvtabular as nvt
from nvtabular.ops import (
    Categorify,
    Clip,
    FillMissing,
    Normalize,
)
from nvtabular.utils import _pynvml_mem_size, device_mem_size

if __name__ == "__main__":
    dataset = "avazu" # terabyte, avazu, kaggle_ad

    if dataset == "avazu":
        Workspace = "/workspace/SC_artifacts_eval/processed_data/workspace/avazu_workspace/"

        # BASE_DIR = "/home/ubuntu/zheng/Nvidia-Merlin/"
        INPUT_DATA_DIR  = "/workspace/SC_artifacts_eval/processed_data/avazu/processed"
        dask_workdir = os.path.join(Workspace + "workdir")
        OUTPUT_DATA_DIR = os.path.join(Workspace + "output") 
        stats_path = os.path.join(Workspace + "stats")

        # Make sure we have a clean worker space for Dask
        if os.path.isdir(dask_workdir):
            shutil.rmtree(dask_workdir)
        os.makedirs(dask_workdir)

        # Make sure we have a clean stats space for Dask
        if os.path.isdir(stats_path):
            shutil.rmtree(stats_path)
        os.mkdir(stats_path)

        # Make sure we have a clean output path
        if os.path.isdir(OUTPUT_DATA_DIR):
            shutil.rmtree(OUTPUT_DATA_DIR)
        os.mkdir(OUTPUT_DATA_DIR)

        train_paths = os.path.join(INPUT_DATA_DIR, "avazu_train/train_subset.txt.parquet")
        valid_paths = os.path.join(INPUT_DATA_DIR, "avazu_val/val_subset.txt.parquet")

        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 4)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 21)]
        LABEL_COLUMNS = ["label"]
        COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMNS

      
    # Dask dashboard
    dashboard_port = "8787"

    # Deploy a Single-Machine Multi-GPU Cluster
    protocol = "tcp"  # "tcp" or "ucx"
    NUM_GPUS = [0, 1, 2, 3]
    visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Delect devices to place workers
    device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.  default:0.7 terabyte:0.5
    device_pool_frac = 0.8  # default:0.8  terabyte:0.9
    part_mem_frac = 0.15  # default:0.15   terabyte:0.05

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)

    # Check if any device memory is already occupied
    for dev in visible_devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    cluster = None  # (Optional) Specify existing scheduler port
    if cluster is None:
        cluster = LocalCUDACluster(
            protocol=protocol,
            n_workers=len(visible_devices.split(",")),
            CUDA_VISIBLE_DEVICES=visible_devices,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )

    # Create the distributed client
    client = Client(cluster)
    print(client)
    # client

    # Initialize RMM pool on ALL workers
    def _rmm_pool():
        rmm.reinitialize(
            # RMM may require the pool size to be a multiple of 256.
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,  # Use default size
        )


    client.run(_rmm_pool)
    # define our dataset schema
    

    num_buckets = 70000000
    categorify_op = Categorify(out_path=stats_path, max_size=num_buckets)
    # categorify_op = Categorify(out_path=stats_path)
    cat_features = CATEGORICAL_COLUMNS >> categorify_op
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + LABEL_COLUMNS

    workflow = nvt.Workflow(features, client=client)

    dict_dtypes = {}
    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in CONTINUOUS_COLUMNS:
        dict_dtypes[col] = np.float32

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    train_dataset = nvt.Dataset(train_paths, engine="parquet", part_size=part_size)
    valid_dataset = nvt.Dataset(valid_paths, engine="parquet", part_size=part_size)

    output_train_dir = os.path.join(OUTPUT_DATA_DIR, "train/")
    output_valid_dir = os.path.join(OUTPUT_DATA_DIR, "valid/")

    workflow.fit(train_dataset)

    workflow.transform(train_dataset).to_parquet(
        output_path=output_train_dir,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )

    workflow.transform(valid_dataset).to_parquet(
        output_path=output_valid_dir,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )
    
    print("finish")

    workflow.save(os.path.join(OUTPUT_DATA_DIR, "workflow"))


    