### Evaluation Result for Figure 18
We provide a script for running the breakdown study of Efficient TT-table Backward.
To run this experiment, please make sure you are using the docker image: **happy233/zheng_dlrm:latest**.

In this experiment, we have four different setting:
* TT-Rec
* +Fused update
* +Gradient Aggregation
* +Index Reordering

Please run:
```
./run.sh
```

The result will be saved in **out.log**.