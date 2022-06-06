import os
import glob
import torch

from cpu_gpu_model import EMB_Tables, MLP_Layers, TT_Tables
import time

# import torch.multiprocessing as mp
# from tormultiprocessingch. import Process, Queue, Lock, Event
import threading
from threading import Thread, Event
from queue import Queue

from torch.utils.cpp_extension import load
cache_sync_cuda = load(name="cache_sync_cuda", sources=[
    "/workspace/SC_artifacts_eval/Figure16/cache_sync/cache_sync_wrap.cpp", 
    "/workspace/SC_artifacts_eval/Figure16/cache_sync/cache_sync_kernel.cu", 
    ], verbose=True)

import argparse
from tqdm import tqdm
import sys
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from random_dataloader import in_memory_dataloader_cpu_gpu
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 


def time_wrap():
    torch.cuda.synchronize()
    return time.time()


def consumer_mlp(input_queue, output_queue, model, TT_table, batch_num):
    # Synchronize access to the console
    batch_cnt = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCELoss(reduction="mean")
    tag = 0
    for i in range(0,batch_num):
        cnt = 0
        while True:
            cnt += 1
            if not input_queue.empty():
                # print("get something from queue")
                x_int, emb, unique_list, inverse_list, y = input_queue.get()

                # if tag > 0:
                    # for idx in range(len(emb)):
                    #     cache_sync_cuda.cache_sync(unique_list[idx], last_unique[idx], emb[idx], last_batch[idx])

                for idx in range(len(emb)):
                    emb[idx].requires_grad = True

                emb_optimizer = torch.optim.SGD(emb, lr=0.1)
                
                full_emb = []
                for idx in range(len(emb)):
                    embeddings = emb[idx][inverse_list[idx]].squeeze()
                    full_emb.append(embeddings)

                z = model(x_int, full_emb)
                E = loss_fn(z, y)

                optimizer.zero_grad()
                emb_optimizer.zero_grad()
                E.backward()
                optimizer.step()
                emb_optimizer.step()

                last_batch = emb
                last_unique = unique_list

                tag = 1
                output_queue.put(emb)

                break

            time.sleep(0.0001)
            # if cnt > 1000:
            #     print("cnt out of bound consumer_mlp")
            #     break
    # event.wait() # wait for consumer_emb

def consumer_emb(input_queue, index_queue, model, batch_num):
    global emb_lock
    for i in range(0,batch_num):
        cnt = 0
        while True:
            cnt += 1
            if not input_queue.empty() and not index_queue.empty():

                emb = input_queue.get()
                idx = index_queue.get()
                
                emb_lock = 1
                model.update(idx, emb)
                emb_lock = 0

                break

            time.sleep(0.0001)
            # if cnt > 1000:
            #     print("cnt out of bound consumer_emb")
            #     break



if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "avazu" or dataset == "avazu_reordered":
        table_num = 20
        feature_size = 16
        dense_num = 2
        top_num = 226
        table_length = [8, 4680, 7567, 27, 8380, 550, 36, 2512738, 6385037, 8165, 6, 5, 2621, 9, 10, 434, 5, 69, 173, 61]
    elif dataset == "kaggle" or dataset == "kaggle_reordered":
        table_num = 26
        feature_size = 16
        dense_num = 13
        top_num = 367
        table_length = [1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633, 7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16, 272266, 105, 138045]
    elif dataset == "terabyte" or dataset == "terabyte_reordered":
        table_num = 26
        feature_size = 64
        dense_num = 13
        top_num = 415
        table_length = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]

    batch_size = 4096
    train_iter = in_memory_dataloader_cpu_gpu(batch_size, 0, dataset=dataset)
    num_iters = 1000

    device = 'cuda:0'

    EMB_tables = EMB_Tables(
        feature_size,
        table_length,
        device,
        threshold=99999999,
    )

    MLP_layers = MLP_Layers(
        [dense_num, 512, 256, 64, feature_size],
        [top_num, 512, 256, 1],
        'dot',
        device
    )

    TT_Table = TT_Tables(
        feature_size,
        table_length,
        device,
        threshold=99999999,
    )

    print("finish init model")

    learning_rate = 0.1 # default 0.1
    parameters = MLP_layers.parameters()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    batch_num = 1000
    InputDataQue = Queue(1)
    IndexQue = Queue(1)
    OutputDataQue = Queue(1)
    event = Event()

    p_mlp = Thread(target=consumer_mlp, args=(InputDataQue, OutputDataQue, MLP_layers, TT_Table, batch_num))
    p_emb = Thread(target=consumer_emb, args=(OutputDataQue, IndexQue, EMB_tables, batch_num))
    p_mlp.start()
    p_emb.start()

    start = time_wrap()
    for i in tqdm(range(0,batch_num)):
        label, sparse, sparse_gpu, dense = train_iter.next()

        emb, unique_list, inverse_list = EMB_tables.unique_forward(sparse_gpu) # has been send to gpu 
        batch_input = (dense, emb, unique_list, inverse_list, label)

        cnt = 0
        while True:
            cnt += 1
            if not InputDataQue.full():
                # print("put something into queue")
                InputDataQue.put(batch_input)
                IndexQue.put(unique_list)
                break
            time.sleep(0.0001)
            # if cnt > 1000:
            #     print("cnt out of bound produce?")
            #     break

    p_mlp.join()
    p_emb.join()
    end = time_wrap()

    print("time:",end-start)
    print("Result saved to out.log")
    with open('out.log', 'a') as f:
        f.write("EL-Rec(pipeline), {}, training_time:{:.3f}\n".format(dataset, end-start))


