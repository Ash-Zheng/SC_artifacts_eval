import hugectr
from mpi4py import MPI
import time

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nDev', type=int, default=1)

args = parser.parse_args()
nDev = args.nDev

if nDev == 2:
    vvgpu = [[0, 1]]
    workspace_size_per_gpu_in_mb = 14000
elif nDev == 4:
    vvgpu = [[0, 1, 2, 3]]
    workspace_size_per_gpu_in_mb = 7000


solver = hugectr.CreateSolver(max_eval_batches = 70,
                              batchsize_eval = 4096,
                              batchsize = 4096,
                              lr = 0.1,
                              warmup_steps = 5,
                              vvgpu = vvgpu,
                              repeat_dataset = True,
                              i64_input_key=True,
                              )
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                source = ["/workspace/HugeCTR/HugeCTR_data/file_list.txt"],
                                eval_source = "/workspace/HugeCTR/HugeCTR_data/file_list_test.txt",
                                check_type = hugectr.Check_t.Non)

optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                    update_type = hugectr.Update_t.Local,
                                    atomic_update = True)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=1,
        dense_name="dense",
        data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, False, 1)],
    )
)
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            slot_size_array=[
                                40000000
                            ],
                            workspace_size_per_gpu_in_mb = workspace_size_per_gpu_in_mb,  # 19532, 9766, 4833
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
#===========================================================================================================
 

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["interaction1"],
                            leading_dim=128,
                            ))

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["interaction1"],
                            top_names = ["fc1"],
                            num_output=1))
                                                                                         
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc1", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()

iters = 5000
start_time = time.time()
model.fit(max_iter = iters, display = 100, eval_interval = 10000000, snapshot = 10000000, snapshot_prefix = "dlrm")
end_time = time.time()

total = end_time - start_time
throughput = nDev * iters/total

print("time:{:.3f}, throughput:{:.3f}".format(total, throughput))
print("Result saved to out.log")
with open('out.log', 'a') as f:
    f.write("HugeCTR_single_large_table, {}GPU, throughput: {:.3f}\n".format(nDev, throughput))

