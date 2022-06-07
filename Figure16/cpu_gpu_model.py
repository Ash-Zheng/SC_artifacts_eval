import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parameter import Parameter
import numpy as np
import sys
from torch_scatter import scatter, segment_coo

sys.path.append('/workspace/SC_artifacts_eval/models/Efficient_TT')
from efficient_tt import Eff_TTEmbedding

import threading

class EMB_Tables(nn.Module):
    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()

        for i in range(0, len(ln)):
            n = ln[i]
            if n <= self.threshold:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

                # plan b
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=False)

                emb_l.append(EE)

        return emb_l

    def __init__(
        self,
        dim_feature=None,
        list_emb=None,
        device='cuda:0',
        threshold = 100000000
    ):
        super(EMB_Tables, self).__init__()

        # create operators
        self.emb_idx = []
        self.threshold = threshold
        for i in range(len(list_emb)):
            if list_emb[i] <= self.threshold:
                self.emb_idx.append(i)

        self.device = device
        self.dim_feature = dim_feature
        self.emb_l = self.create_emb(dim_feature, list_emb)
        # print("num_emb:", len(self.emb_idx), "idx:", self.emb_idx)
    
    def forward(self, lS_i, pick=None):
        ly = []
        idx = 0
        for i in self.emb_idx:
            sparse_index_group_batch = lS_i[i].cpu()
            if pick!=None:
                sparse_index_group_batch = sparse_index_group_batch[pick]
            E = self.emb_l[idx]

            V = E(
                sparse_index_group_batch
            )
            V = V.to(self.device).detach()
            # V = V.to(self.device)
            ly.append(V)
            idx += 1
        
        # ly = torch.stack(ly).to(self.device)
        return ly
    
    def to_cpu_forward(self, lS_i):
        ly = []
        idx = 0
        for i in self.emb_idx:
            sparse_index_group_batch = lS_i[i].cpu()
            E = self.emb_l[idx]

            V = E(
                sparse_index_group_batch
            )
            V = V.to(self.device)
            # V = V.to(self.device)
            ly.append(V)
            idx += 1
        
        # ly = torch.stack(ly).to(self.device)
        return ly

    def unique_forward(self, lS_i, pick=None):
        ly = []
        inverse_list = []
        unique_list = []
        idx = 0
        for i in self.emb_idx:
            index = lS_i[i]
            if pick!=None:
                index = index[pick]
            unique, inverse = index.unique(sorted=True, return_inverse=True)
            E = self.emb_l[idx]
            
            V = E.weight.data[unique].to(self.device)
            ly.append(V)
            # inverse_list.append(inverse.to(self.device))
            # unique_list.append(unique.to(self.device))
            inverse_list.append(inverse)
            unique_list.append(unique)
            idx += 1
        
        # ly = torch.stack(ly).to(self.device)
        return ly, unique_list, inverse_list
    
    def unique_get(self, unique_idx):
        ly = []
        idx = 0
        for i in self.emb_idx:
            index = unique_idx[i]
            E = self.emb_l[idx]
            V = E.weight.data[index].squeeze().to(self.device)
            ly.append(V)
            idx += 1

        return ly

    def update(self, index, embeddings):
        # for i in range(len(embeddings)):
        idx = 0
        for i in self.emb_idx:
            self.emb_l[idx].weight.data[index[i].squeeze()] = embeddings[idx].squeeze().cpu()
            idx += 1

    def scatter_update(self, inverse, unique, grad, receive_emb_list, learning_rate=0.1):
        idx = 0
        grad = grad * learning_rate
        for i in self.emb_idx:
            scatter(grad[i], inverse[:,i], dim=0, out=receive_emb_list[i],reduce="sum")
            self.emb_l[idx].weight.data[unique[i]] = receive_emb_list[i][0:len(unique[i])].cpu()
            receive_emb_list[i].zero_()
            idx += 1

    def scatter_update_list(self, inverse, unique, grad, receive_emb_list, learning_rate=0.1):
        idx = 0
        grad = grad * learning_rate
        for i in self.emb_idx:
            scatter(grad[i], inverse[i], dim=0, out=receive_emb_list[i],reduce="sum")
            self.emb_l[idx].weight.data[unique[i]] = receive_emb_list[i][0:len(unique[i])].cpu()
            receive_emb_list[i].zero_()
            idx += 1

    
    def print_size(self):
        _sum = 0
        size_list = []
        for table in self.emb_l:
            size = table.embedding_dim * table.num_embeddings * 4 # float32
            size_list.append(str(round(size/1024/1024, 2))+"MB")
            _sum += size
        # print("sum:",_sum, "~:",_sum/1024/1024,"MB")
        sum_size = str(round(_sum/1024/1024,2))+"MB"
        print(sum_size, size_list)


class MLP_Layers(nn.Module):
    def create_mlp(self, ln, top=False):
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
          
            layers.append(LL)

            if top and i == len(ln) - 2:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def __init__(
        self,
        list_bot=None,
        list_top=None,
        arch_interaction_op=None, #'dot'
        device='cuda:0'
    ):
        super(MLP_Layers, self).__init__()

        # save arguments
        self.arch_interaction_op = arch_interaction_op
        self.device = device

        # create operators
        self.bot_l = self.create_mlp(list_bot)
        self.top_l = self.create_mlp(list_top, top=True)

        # nn layers saved on GPU
        self.bot_l = self.bot_l.to(self.device)
        self.top_l = self.top_l.to(self.device)


    def apply_mlp(self, x, layers):
        return layers(x)
    

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape

            (emb_num,_,_) = ly.shape # tensor
            ly = ly.transpose(1,0).reshape(batch_size,emb_num*d) # tensor
            T = torch.cat((x,ly),dim=1).view((batch_size, -1, d)) # tensor

            # T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d)) # list

            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R


    def forward(self, dense_x, ly):
        return self.sequential_forward(dense_x, ly)


    def sequential_forward(self, dense_x, ly):
        # process dense features (using bottom mlp), resulting in a row vector
        # print("bottom mlp")
        x = self.apply_mlp(dense_x, self.bot_l)

        # print("interaction")
        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # print("top mlp")
        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)
   
        return p


class  TT_Tables(nn.Module):
    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()

        for i in range(0, len(ln)):
            n = ln[i]
            if n > self.threshold:
                eff_emb = Eff_TTEmbedding(
                    num_embeddings=n,
                    embedding_dim=m,
                    tt_p_shapes=None,
                    tt_q_shapes=[2,4,2],
                    tt_ranks=[128, 128],
                    weight_dist="uniform"
                ).to(self.device)

                # eff_emb = TTEmbeddingBag(
                #     num_embeddings=n,
                #     embedding_dim=m,
                #     tt_p_shapes=None,
                #     tt_q_shapes=[2,4,2],
                #     tt_ranks=[128, 128],
                #     sparse=True,
                #     use_cache=False,
                #     weight_dist="uniform"
                # ).to(self.device)

                emb_l.append(eff_emb)

        return emb_l

    def __init__(
        self,
        dim_feature=None,
        list_emb=None,
        device='cuda:0',
        threshold = 100000000
    ):
        super(TT_Tables, self).__init__()

        # create operators
        self.emb_idx = []
        self.normal_table_list = []
        self.threshold = threshold
        for i in range(len(list_emb)):
            if list_emb[i] > self.threshold:
                self.emb_idx.append(i)
            else:
                self.normal_table_list.append(i)

        self.device = device
        self.dim_feature = dim_feature
        self.emb_l = self.create_emb(dim_feature, list_emb)

    def forward(self, lS_i):
        ly = []
        idx = 0
        # offset = torch.tensor(range(lS_i[0].shape[0]+1)).to(self.device)
        for i in self.emb_idx:
            sparse_index_group_batch = lS_i[i]
            E = self.emb_l[idx]

            V = E(
                sparse_index_group_batch,
                # offset
            )
            # V = V.to(self.device)
            ly.append(V)
            idx += 1
        
        # ly = torch.stack(ly).to(self.device)
        return ly