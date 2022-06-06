import torch
import torch.nn as nn
import numpy as np
from random_dataloader import random_dataloader
from torch.utils.cpp_extension import load

from Efficient_TT.efficient_tt import Eff_TTEmbedding

class data_parallel_EMT:
    # ================================== Multi Table =====================================
    def __init__(self, replicate_length, feature_dim, device, batch_size=4096, learning_rate=0.1, table_map=None):
        # super(data_parallel_EMT, self).__init__()

        self.replicate_length = replicate_length
        self.feature_dim = feature_dim
        self.table_num = len(replicate_length)

        self.device = device
        self.learning_rate = learning_rate
            
        # total_replicate_length = np.sum(replicate_length) + 1
        # self.emt = nn.EmbeddingBag(total_replicate_length, feature_dim, mode="sum", sparse=True, padding_idx=0).to(self.device)
        self.table_map = table_map
        self.emt_tag = []

        if feature_dim == 16:
            tt_q_shapes = [2, 4, 2]
        else:
            tt_q_shapes = [4, 4, 4]

        self.emt = nn.ModuleList()
        for i in range(self.table_num):
            # emt = nn.EmbeddingBag(replicate_length[i] + 1, feature_dim, mode="sum",sparse=True, padding_idx=0).to(device)

            if replicate_length[i] > 1000000: # 200000
                emt = Eff_TTEmbedding(
                    num_embeddings=replicate_length[i],
                    embedding_dim=feature_dim,
                    tt_p_shapes=None,
                    tt_q_shapes=tt_q_shapes,
                    tt_ranks=[64, 64],
                    weight_dist="uniform",
                    device=self.device,
                    batch_size=batch_size,
                    ).to(self.device)
                self.emt_tag.append(1)
            else:
                # emt = nn.EmbeddingBag(replicate_length[i], feature_dim, mode="sum", sparse=True).to(device)
                emt = nn.Embedding(replicate_length[i], feature_dim,sparse=True).to(self.device)
                emt.weight.requires_grad = False
                self.emt_tag.append(0)

            self.emt.append(emt)
        # print(self.emt_tag)
 

    def lookup(self, sparse_input):
        sparse_input = sparse_input.to(self.device)
        # batch_size = sparse_input.shape[1]

        sparse_feature = []
        for i in range(self.table_num):
            tag = self.emt_tag[i]
            if tag == 1:
                embedding = self.emt[i](sparse_input[i].view(-1))
            else:
                embedding = self.emt[i](sparse_input[i].view(-1))
                embedding.requires_grad = True
            sparse_feature.append(embedding)
        
        return sparse_feature

    def updata_emt(self, table_id, index, gradient):
        self.emt[table_id].weight.data.index_add_(0, index, gradient*self.learning_rate)