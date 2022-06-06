### Evaluation Result for Figure 14
We provide a script for running the breakdown study of EL-Rec.
To run this experiment, please make sure you are using the docker image: **happy233/zheng_dlrm:latest**.

In this experiment, we have four different setting:
* without Index Reordering
* without Intermediate Result Reuse
* without Gradient Aggregation
* EL-Rec

Please run:
```
./run.sh
```

The result will be saved in **out.log**.