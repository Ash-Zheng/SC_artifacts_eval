### Docker Image 
docker pull happy233/zheng_dlrm

docker run --gpus=all -it --cap-add SYS_NICE -v <your folder>:/workspace/PipeDLRM -w /workspace/PipeDLRM happy233/zheng_dlrm:latest

Example:
```
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval:/workspace/SC_artifacts_eval -w /workspace/SC_artifacts_eval happy233/zheng_dlrm:latest
```

### Data Process
First, make directory:
```
mkdir processed_data
cd processed_data
mkdir avazu
mkdir avazu/processed
mkdir kaggle
mkdir kaggle/processed
mkdir terabyte
mkdir terabyte/processed
mkdir workspace
cd ..
```

Run script to process data:
```
cd data_process
./run.sh
```

### Generate Index Bijection
First install rabbit_module:
```
cd /workspace/SC_artifacts_eval/rabbit_module/src
apt-get install libboost-all-dev
apt-get install libgoogle-perftools-dev
python setup.py install
```

Then run script to record access pattern:
```
cd /workspace/SC_artifacts_eval/rabbit_module/
./1_record_dataset.sh
```

Last step, generate index bijection and save the generate dataset for trianing:
```
cd /workspace/SC_artifacts_eval/rabbit_module/
./2_index_bijection_generate.sh
./3_save_training_data.sh
```

### Result Reproduce



### Some data:
terabyte:
[]



# PipeDLRM

First, please follow the `README.md` in fold `Nvdia-Merlin` and preprocess the dataset

Remember to export CUDA_HOME
```
# adjust to your cuda version
export CUDA_HOME=/usr/local/cuda-11.6
export CUDA_HOME=/usr/local/cuda-11.5
```

Train_DLRM:
```
python facebook_dlrm/train_dlrm.py
```

Test_DLRM:
```
python facebook_dlrm/test_dlrm.py
```

aws s3 cp terabyte_edge21_new.pt s3://dlrm-bucket/terabyte_edges/

### Pip install
pip install ninja
pip install sympy
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install nvtabular

### prepare dataset:
cd Nvidia-Merlin
python data_split
python data_process
python ETL

### prepare embedding:
cd CPU2GPU
python embedding_access_record.py
python generate_access_based_ordering.py





### EL-Rec

#### Environment 

+ Operating systems and versions: Ubuntu 18.04.6+.
+ Compilers and versions: NVCC-11.0, GCC-7.5.0
+ Libraries and versions: CUDA-11.0, Pytorch-1.7,1
+ Input datasets and versions: Criteo Kaggle dataset

One can create a conda environment with the `environment.yml`

#### Data Process

1. Download the Criteo Kaggle dataset from https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310

2. Data prepocessing:
   Create `dataset/kaggle_ad` folder in `Nvidia-Merlin`
   Run the following command:

```
cd Nvdia-Merlin
python data_split_kaggle_ad.py
python data_processing_kaggle_ad.py
python ETLwithNvTabular.py
```

If you face the error:
```
AttributeError: module 'torch' has no attribute 'tensor_split'
```

please edit ` File "/home/ubuntu/anaconda3/envs/Merlin_fastai/lib/python3.8/site-packages/nvtabular/loader/backend.py", line 550` as:
```
 # tensors = self._tensor_split(tensor, len(names), axis=1)
 tensors = self._split_fn(tensor, 1, axis=1)
```


3. Generate Index Bijection:
   Run the following command:

```
cd rabbit_module
cd src

# follow "https://github.com/YukeWang96/OSDI21_AE"
sudo apt-get install libboost-all-dev
sudo apt-get install libgoogle-perftools-dev
sudo apt-get update && sudo apt-get -y install cmake protobuf-compiler
make

python setup.py install
python embedding_access_record.py
python generate_access_based_ordering.py
./graph_generate.sh
```

4. Train Facebook DLRM baseline:
   Run the following command:

```
cd facebook_dlrm_baseline
python train_dlrm.py
```

5. Train TT-Rec baseline:
   Run the following command:

```
cd TTRec
python train_TT_Rec.py
```

# Eff_TT/pure_GPU_training_with_test.py
6. Train EL-Rec (GPU):
   Run the following command:

```
cd EL_Rec_Single_GPU
python pure_GPU_training.py
```

# dynamic_cache/sequential_train.py 
# dynamic_cache/fast_train_with_cache.py
7. Train EL-Rec (CPU+GPU):
   Run the following command:

```
cd EL_Rec_CPU_GPU
python EL_Rec_CPU_GPU.py
```



### HugeCTR:
docker run --gpus=all --rm -it --cap-add SYS_NICE -v /home/zheng_wang/workspace/PipeDLRM:/workspace/PipeDLRM -w /workspace/PipeDLRM -it -u $(id -u):$(id -g) nvcr.io/nvidia/merlin/merlin-training:22.05

docker run --gpus=all -it --cap-add SYS_NICE -v /home/zheng_wang/workspace/PipeDLRM:/workspace/PipeDLRM -w /workspace/PipeDLRM nvcr.io/nvidia/merlin/merlin-training:22.05
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zheng_wang/workspace/PipeDLRM:/workspace/PipeDLRM -w /workspace/PipeDLRM nvcr.io/nvidia/merlin/merlin-training:22.03

docker run --gpus=all -it --cap-add SYS_NICE -v /home/zheng_wang/workspace/PipeDLRM:/workspace/PipeDLRM -w /workspace/PipeDLRM nvcr.io/nvidia/merlin/merlin-training:21.12

docker run --gpus=all -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-training:21.12

docker cp /home/ubuntu/zheng/PipeDLRM/Nvidia-Merlin/workspace/kaggle_ad_workspace e7dc50874689:/workspace/dataset
docker cp /home/ubuntu/zheng/PipeDLRM/Nvidia-Merlin/workspace/avazu_workspace e7dc50874689:/workspace/dataset

docker cp /home/ubuntu/zheng/PipeDLRM/HugeCTR e7dc50874689:/workspace



### Cache hit rate:
/usr/local/cuda-11.0/bin/nsys profile -t cuda,osrt,nvtx,cudnn,cublas -y 60 -d 20 -o output -f true -w true python backward_cache_hit_rate.py

Use this command:
sudo /usr/local/cuda-11.0/bin/ncu --csv --set full /home/ubuntu/anaconda3/envs/nvtabular/bin/python ./backward_cache_hit_rate.py
sudo /usr/local/cuda-11.0/bin/ncu --csv --set full -o ordered_cache_hit_rate /home/ubuntu/anaconda3/envs/nvtabular/bin/python ./backward_cache_hit_rate.py 


docker run --gpus=all -it --cap-add SYS_NICE -v /local/home/zheng_wang/workspace/Balance-Rec/:/workspace/Balance-Rec -w /workspace/Balance-Rec nvcr.io/nvidia/merlin/merlin-training:21.12



python3 criteo_script/preprocess.py --src_csv_path=$DST_DATA_DIR/day_1_shuf --dst_csv_path=$DST_DATA_DIR/day_1_shuf.out --normalize_dense=1 --feature_cross=0


### data_processing
python3 criteo_script/preprocess.py --src_csv_path=dcn_data/day_$1_shuf --dst_csv_path=$DST_DATA_DIR/day_1_shuf.out --normalize_dense=1 --feature_cross=0


python3 criteo_script/preprocess.py --src_csv_path=train_subset.txt --dst_csv_path=train_subset.out --normalize_dense=1 --feature_cross=0

bash preprocess.sh 1 dcn_data pandas 1 0
bash preprocess.sh 1 wdl_data pandas 1 1 100


criteo2hugectr train.txt criteo2hugectr/sparse_embedding criteo2hugectr/file_list.txt 

criteo2hugectr local_kaggle/train_subset.out local_kaggle/kaggle_ local_kaggle/file_list.txt 


criteo2hugectr train_subset.out new_kaggle/kaggle_ new_kaggle/file_list.txt 0 100


### torchrec install method
conda env create -f environment.yml
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
torchx run -s local_cwd dist.ddp -j 1x2 --script single_table_test.py