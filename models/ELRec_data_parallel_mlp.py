import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self,ln, top=False):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
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

            self.layers.append(LL)

            # construct sigmoid or relu operator
            if top and i == len(ln) - 2:
                self.layers.append(nn.Sigmoid())
            else:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            # z = self.layers[0](x)
            x = l(x)
        return x


class dlrm_hybrid(nn.Module):
    def __init__(
        self,
        list_bot=None,
        list_top=None,
        device=None,
        # arch_interaction_op='dot', #'dot'
    ):
        super(dlrm_hybrid, self).__init__()

        # save arguments
        # self.arch_interaction_op = arch_interaction_op
        # self.cuda_device = "cuda:{}".format(device)
        # self.device = device
        cuda_device = "cuda:{}".format(device)

        # create operators
        self.bot_l = MLP(list_bot).to(cuda_device)
        self.top_l = MLP(list_top, top=True).to(cuda_device)


    def interact_features(self, x, ly):
        # pair-pair wise dot product, then concatnate with dense feature
        # if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
        (batch_size, d) = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        offset = 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]
        R = torch.cat([x] + [Zflat], dim=1)
        # elif self.arch_interaction_op == "cat":
        #     R = torch.cat([x] + ly, dim=1)

        return R

    
    def apply_mlp(self, x, layers):
        return layers(x)


    def forward(self, dense_x, embedding):
        dense_feature = self.bot_l(dense_x)

        x = torch.stack(embedding)
        
        interacted_feature = self.interact_features(dense_feature, embedding)
        
        output = self.top_l(interacted_feature)

        return output