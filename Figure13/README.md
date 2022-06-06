### Evaluation Result for Figure 13
We provide a script for running EL-Rec (1/2/4 GPU) and HugeCTR (2/4 GPU).

### Result of EL-Rec
To get the training throughput of EL-Rec, please make sure you are using the docker image: **happy233/zheng_dlrm:latest**.

Please run:
```
./run.sh
```
The result will be saved in **out.log**.

### Result of HugeCTR
To get the training throughput of HugeCTR, please make sure you are using the docker image: **zhengwang0122/dlrm_hugectr:latest**.


To get the result, please run:
```
./run_hugectr.sh
```
The result will be saved in **out.log**.