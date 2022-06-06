import torch
from tqdm import tqdm
import time

import sys
sys.path.append('/workspace/SC_artifacts_eval/rabbit_module')

from breakdown_efficient_TT.efficient_tt import Eff_TTEmbedding
from random_dataloader import in_memory_dataloader
from unique_generator import unique_generator

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--setting', type=int, default=0)
parser.add_argument('--length', type=int, default=2500000)


def time_wrap():
    torch.cuda.synchronize()
    return time.time()

if __name__ == "__main__":
    # size = [8, 4680, 7567, 27, 8380, 550, 36, 2512738, 6385037, 8165, 6, 5, 2621, 9, 10, 434, 5, 69, 173, 61]
    # size = [1461, 581, 9214729, 2031648, 306, 24, 12471, 634, 4, 90948, 5633, 7607629, 3183, 28, 14825, 4995567, 11, 5606, 2172, 4, 6431684, 18, 16, 272266, 105, 138045]
    # size = [33121475, 30875, 15297, 7296, 19902, 4, 6519, 1340, 63, 20388174, 945108, 253624, 11, 2209, 10074, 75, 4, 964, 15, 39991895, 7312994, 28182799, 347447, 11111, 98, 35]

    # n = [2500000, 5000000, 10000000]
    # eff_tag == 0: both efficient
    # eff_tag == 1: forward TT, backward efficient
    # eff_tag == 2: forward efficient, backward TT
    args = parser.parse_args()
    setting = args.setting
    n = args.length

    if setting == 0:
        exp = "EL-Rec"
        eff_tag = 0
        reordered = 1
    elif setting == 1:
        exp = "w/o-Index-Reordering"
        eff_tag = 0
        reordered = 0
    elif setting == 2:
        exp = "w/o-Intermediate-Result-Reuse"
        eff_tag = 1
        reordered = 1
    elif setting == 3:
        exp = "w/o-Gradient-Aggregation"
        eff_tag = 2
        reordered = 1

    if n == 2500000:
        dataset = "kaggle"
        t_id = 3
    elif n == 5000000:
        dataset = "kaggle"
        t_id = 15
    elif n == 10000000:
        dataset = "kaggle"
        t_id = 2
   
    if reordered == 1:
        dataset += "_reordered"

    device = 'cuda:0'
    batch_size = 8192
    train_iter = in_memory_dataloader(batch_size,0,dataset=dataset)
    unique_gen = unique_generator(batch_size=batch_size, nDev=1, table_num=26, rank=0, dataloader=train_iter, generate=False, dataset=dataset)
    offset = torch.tensor(range(batch_size)).to(device)  # int64
    offset1 = torch.tensor(range(batch_size+1)).to(device)  # int64

    feature_size = 64
    q_size = [4, 4, 4]
    tt_rank = [128, 128]

    eff_emb = Eff_TTEmbedding(
        num_embeddings=n,
        embedding_dim=feature_size,
        tt_p_shapes=None,
        tt_q_shapes=q_size,
        tt_ranks=tt_rank,
        weight_dist="uniform",
        batch_size=batch_size,
    ).to(device)

    loss_fn = torch.nn.BCELoss(reduction="mean")
    fc_layer = torch.nn.Linear(feature_size, 1).to(device)

    # warm up
    for i in range(100):
        label, sparse, dense = train_iter.next()
        unique, inverse = unique_gen.next()
        emb_1 = eff_emb(sparse[t_id], offset, unique[t_id], inverse[:,t_id], eff_tag)
        output1 = fc_layer(emb_1)
        Loss1 = loss_fn(label, output1)
        Loss1.backward()

    eff_time = 0
    iters = 5000
    t1 = time_wrap()
    for i in tqdm(range(iters)):
        label, sparse, dense = train_iter.next()
        unique, inverse = unique_gen.next()
        emb_1 = eff_emb(sparse[t_id], offset, unique[t_id], inverse[:,t_id], eff_tag)
        Loss1 = loss_fn(label.squeeze(), emb_1.sum(1))
        Loss1.backward()
    t2 = time_wrap()

    total = t2 -t1
    throughput = iters/total

    print("time:{:.3f}, throughput:{:.3f}".format(total, throughput))
    print("Result saved to out.log")
    with open('out.log', 'a') as f:
        f.write("table_length:{}, setting:{}, throughput: {:.3f}\n".format(n, exp, throughput))


    


