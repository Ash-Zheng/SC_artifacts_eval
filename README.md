# EL-Rec: Efficient Large-scale Recommendation Model Training via Tensor-train Embedding Table

This repo is for the SC 2022 artifacts evaluation.



## Environment Setup and Preprocessing

* If you are using the server that we provided, you don't need to setup the environment and prepross the dataset, please use the docker container `SC_artifacts`  and `SC_ae_hugectr` (only for running hugectr) to run the scripts.
* If you want to run the codes on your own server, please refer to the  [Docker Image](# Docker-Image) , [Data Process](# Data-Process) and [Generate Index Bijection](# Generate-Index-Bijection) sections.



## Running Expriment

We provoded some script to run the expriment and get the result in our paper, includes:

* Figure 11: 
  * We provide codes for running DLRM (CPU-GPU), TT-Rec and EL-Rec.
  * Please also refer to the `Figure11/README.md`.
* Figure 12:
  * We provide codes for running DLRM (1 GPU), DLRM (4 GPU), EL-Rec (1 GPU) and EL-Rec (4 GPU).
  * Please also refer to the `Figure12/README.md`.
* Figure 13:
  * We provide codes for running EL-Rec (1/2/4 GPU), TorchRec(2/4 GPU) and HugeCTR (2/4 GPU).
  * To run the EL-Rec, please use the docker container `SC_artifacts`.
  * To run the HugeCTR, please use the docker container `SC_ae_hugectr`.
  * To run the HugeCTR, please use the conda environment `new_torchrec`.
  * Please also refer to the `Figure13/README.md`.
* Figure 14:
  * We provide codes to get the breakdown study of EL-Rec.
  * Please also refer to the `Figure14/README.md`.
* Figure 15:
  * We provide codes to draw the loss convergence curve of DLRM, TT-Rec and EL-Rec.
  * Please also refer to the `Figure15/README.md`.
* Figure 16:
  * We provide codes for running DLRM, EL-Rec (Sequential) and EL-Rec (Pipeline).
  * Please also refer to the `Figure16/README.md`.
* Figure 17:
  * We provide codes to get the breakdown study of Efficient TT-table lookup.
  * Please also refer to the `Figure17/README.md`.
* Figure 18:
  * We provide codes to get the breakdown study of Efficient TT-table backward.
  * Please also refer to the `Figure18/README.md`.
* Table 4:
  * We provide codes to get the test accuracy of different frameworks.
  * Please also refer to the `Table4/README.md`.


## Docker Image 

We need two different dockers to reproduce the evaluation results.

```
docker pull happy233/zheng_dlrm:latest
docker pull zhengwang0122/dlrm_hugectr:latest
```

To run the docker images, please using the following commands:

```
docker run --gpus=all -it --cap-add SYS_NICE -v $$<your folder>$$:/workspace/SC_artifacts_eval -w /workspace/SC_artifacts_eval happy233/zheng_dlrm:latest
docker run --gpus=all -it --cap-add SYS_NICE -v $$<your folder>$$/Figure13:/workspace/HugeCTR -w /workspace/HugeCTR zhengwang0122/dlrm_hugectr:latest
```

Examples:

```
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval:/workspace/SC_artifacts_eval -w /workspace/SC_artifacts_eval happy233/zheng_dlrm:latest
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval/Figure13:/workspace/HugeCTR -w /workspace/HugeCTR zhengwang0122/dlrm_hugectr:latest
```

Please install the following packages in `happy233/zheng_dlrm` image:

```
pip install sympy
```



## Data Process

Download dataset:

* Kaggle: https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310
* Terabyte:  https://labs.criteo.com/2013/12/download-terabyte-click-logs/
* Avazu: https://www.kaggle.com/c/avazu-ctr-prediction/data



Unzip and save dataset to:

```
dlrm_dataset/
├── avazu
│   └── avazu
├── kaggle
│   ├── readme.txt
│   ├── test.txt
│   └── train.txt
└── terabyte
    ├── day_0
    ├── day_1
    └── day_2
```



First, make directory:

```
mkdir processed_data/avazu
mkdir processed_data/avazu/processed
mkdir processed_data/kaggle
mkdir processed_data/kaggle/processed
mkdir processed_data/terabyte/processed
mkdir processed_data/workspace
```

Run script to process data:

```
cd data_process
./run.sh
```



## Generate Index Bijection

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

### 