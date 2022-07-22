### Evaluation Result for Figure 17
We provide a script for running the breakdown study of Efficient TT-table Lookup.
To run this experiment, please make sure you are using the docker image: **happy233/zheng_dlrm:latest**.

In this experiment, we have four different setting:
* TT-Rec
* +Intermediate Result Reuse
* +Index Reordering

Please run:
```
./run.sh
```

The result will be saved in **out.log**.