import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from TTRec_kernel.tt_embedding import TTEmbeddingBag
import time

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

### define dlrm in PyTorch ###
class TT_DLRM_Net(nn.Module):
    def create_mlp(self, ln, top_mlp=False):
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)
            
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if top_mlp and i == len(ln) - 2:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            # if i == sigmoid_layer:
            #     layers.append(nn.Sigmoid())
            # else:
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, len(ln)):
            n = ln[i]

            if m == 16:
                q_shape = [2, 2, 4]
            elif m == 64:
                q_shape = [4, 4, 4]

            if n > 1000000:
                EE = TTEmbeddingBag(
                    num_embeddings=n,
                    embedding_dim=m,
                    tt_p_shapes=None,
                    tt_q_shapes=q_shape,
                    tt_ranks=[self.tt_rank, self.tt_rank],
                    sparse=False,
                    use_cache=False,
                    weight_dist="uniform",
                    learning_rate=0.1
                )
                self.emb_tag.append(1)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_tag.append(0)
            
            emb_l.append(EE)
        return emb_l

    def __init__(
        self,
        feature_size=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        device=None,
        tt_rank=128,
    ):
        super(TT_DLRM_Net, self).__init__()

        # create operators
        self.device = device
        self.emb_tag=[]
        self.tt_rank=tt_rank
        self.emb_l = self.create_emb(feature_size, ln_emb)
        self.bot_l = self.create_mlp(ln_bot)
        self.top_l = self.create_mlp(ln_top,top_mlp = True)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.bot_mlp_time = 0
        self.upper_mlp_time = 0
        self.emb_time = 0
        self.normal_emb_time = 0
        self.tt_emb_time = 0
        self.interact_time = 0 
        self.record_time = False
        self.ori_total = 0
        self.decom_total = 0
        self.unreduce_total = 0

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_i, emb_l, pick=None):
        ly = []
        length = lS_i[0].shape[0]
        if pick!=None:
            offset = torch.tensor(range(len(pick)+1)).to(self.device)
        else:
            offset = torch.tensor(range(lS_i[0].shape[0]+1)).to(self.device)
       
        for k, index in enumerate(lS_i):
            if pick!=None:
                index = index[pick]

            E = emb_l[k]
            tag = self.emb_tag[k]
            if tag == 1:
                V = E(
                    index,
                    offset,
                )
            else:
                index = index.view(-1,1)
                V = E(
                    index
                )
            # print(V.shape)
            ly.append(V)

        return ly

    def interact_features(self, x, ly):

        # concatenate dense and sparse features
        (batch_size, d) = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        li = torch.tensor([i for i in range(ni) for j in range(i)])
        lj = torch.tensor([j for i in range(nj) for j in range(i)])

        Zflat = Z[:, li, lj]
        R = torch.cat([x] + [Zflat], dim=1)

        return R

    def forward(self, dense_x, lS_i, pick=None):
        return self.sequential_forward(dense_x, lS_i, pick)

    def sequential_forward(self, dense_x, lS_i, pick=None):
        if pick != None:
            dense_x = dense_x[pick]

        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        ly = self.apply_emb(lS_i, self.emb_l, pick)
        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_l)
        z = p

        return z


