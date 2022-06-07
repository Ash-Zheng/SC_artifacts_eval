
#### If you are using the server we provided, I can just skip the Docker Image and Data Process setction ####

### Docker Image 
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
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval:/workspace/SC_artifacts_eval -w /workspace/SC_artifacts_eval zhengwang0122/dlrm_hugectr:latest
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

Examples:
```
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval:/workspace/SC_artifacts_eval -w /workspace/SC_artifacts_eval happy233/zheng_dlrm:latest
docker run --gpus=all -it --cap-add SYS_NICE -v /home/zhengw/workspace/SC_artifacts_eval/Figure13:/workspace/HugeCTR -w /workspace/HugeCTR zhengwang0122/dlrm_hugectr:latest
```

Please install the following packages in happy233/zheng_dlrm iamge:
```
pip install sympy
```


### Data Process
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



