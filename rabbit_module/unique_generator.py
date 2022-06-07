import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
from tqdm import tqdm

import sys
from random_dataloader import random_dataloader, in_memory_dataloader
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 
parser.add_argument('--nDev', type=int, default=1) 
parser.add_argument('--batch_size', type=int, default=4096) 



class unique_generator:
    def __init__(self, batch_size, nDev, table_num, rank, dataloader,generate=False, dataset="kaggle"):
        self.batch_size = batch_size
        self.nDev = nDev
        self.total_batch_size = batch_size * nDev
        self.table_num = table_num
        self.rank = rank
        self.generate = generate
        self.dataset = dataset
        self.num_batch = int(dataloader.num_batch)
        self.inverse = None
        self.unique = []
        self.index = []
        self.unique_list = [[] for _ in range(self.num_batch)]
        self.cnt = 0

        if generate == True:
            self.generate_new_unique(dataloader)
        else:
            input_file = "/workspace/SC_artifacts_eval/Access_Index/{}/unique/inverse_{}_{}.pt".format(self.dataset, batch_size, nDev)
            self.inverse = torch.load(input_file).to(self.rank)
            for i in range(self.table_num):
                file_unique = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/unique_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
                file_index = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/index_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
                self.unique.append(torch.load(file_unique).to(self.rank))
                self.index.append(torch.load(file_index).to(self.rank))

            for i in range(self.num_batch):
                for j in range(self.table_num):
                    start = self.index[j][i]
                    end = self.index[j][i+1]
                    self.unique_list[i].append(self.unique[j][start:end])

    def generate_new_unique(self, dataloader):
        inverse_list = [[] for _ in range(self.table_num)]
        unique_list = [[] for _ in range(self.table_num)]
        unique_index_list = [[0] for _ in range(self.table_num)]
        unique_index = [0 for _ in range(self.table_num)]
        max_unique_num = [0 for _ in range(self.table_num)]

        for _ in tqdm(range(self.num_batch)):
        # for _ in range(2):
            _, sparse, _ = dataloader.next()
            for j in range(self.table_num):
                unique, inverse = torch.unique(sparse[j],sorted=True, return_inverse=True)
                inverse_list[j].append(inverse)
                unique_list[j].append(unique)
                unique_index[j] += len(unique)
                unique_index_list[j].append(unique_index[j])
                max_unique_num[j] = len(unique) if len(unique) > max_unique_num[j] else max_unique_num[j]
        
        for i in range(self.table_num):
            inverse_list[i] = torch.cat(inverse_list[i])
            unique_list[i] = torch.cat(unique_list[i])
            unique_index_list[i] = torch.tensor(unique_index_list[i])

        cat_inverse = torch.cat(inverse_list,dim=1)
        final_tensor = torch.cat((cat_inverse,torch.tensor(max_unique_num,device=self.rank).unsqueeze(0)))
        file_name = "/workspace/SC_artifacts_eval/Access_Index/{}/unique/inverse_{}_{}.pt".format(self.dataset, self.batch_size, self.nDev)
        torch.save(final_tensor.to('cpu'), file_name)

        for i in range(self.table_num):
            file_name_unique = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/unique_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
            file_name_index = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/index_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
            torch.save(unique_list[i].to('cpu'), file_name_unique)
            torch.save(unique_index_list[i].to('cpu'), file_name_index)

    def next(self):
        if self.cnt >= self.num_batch-1:
            self.cnt = 0

        start = self.cnt * self.total_batch_size
        end = start + self.total_batch_size
        inverse = self.inverse[start:end]
        unique = self.unique_list[self.cnt]

        self.cnt += 1

        return unique, inverse

    def max_length(self):
        max_length = self.inverse[-1]
        return max_length


class unique_generator_cpu:
    def __init__(self, batch_size, nDev, table_num, rank, dataloader, dataset="kaggle"):
        self.batch_size = batch_size
        self.nDev = nDev
        self.total_batch_size = batch_size * nDev
        self.table_num = table_num
        self.rank = rank
        self.dataset = dataset
        self.num_batch = int(dataloader.num_batch)
        self.inverse = None
        self.unique = []
        self.unique_cpu = []
        self.index = []
        self.unique_list = [[] for _ in range(self.num_batch)]
        self.unique_list_cpu = [[] for _ in range(self.num_batch)]
        self.cnt = 0

      
        input_file = "/workspace/SC_artifacts_eval/Access_Index/{}/unique/inverse_{}_{}.pt".format(self.dataset, batch_size, nDev)
        self.inverse = torch.load(input_file).to(self.rank)
        for i in range(self.table_num):
            file_unique = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/unique_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
            file_index = '/workspace/SC_artifacts_eval/Access_Index/{}/unique/index_{}_{}_{}.pt'.format(self.dataset, self.batch_size, self.nDev, i)
            self.unique.append(torch.load(file_unique).to(self.rank))
            self.unique_cpu.append(torch.load(file_unique).to('cpu'))
            self.index.append(torch.load(file_index).to(self.rank))

        for i in range(self.num_batch):
            for j in range(self.table_num):
                start = self.index[j][i]
                end = self.index[j][i+1]
                self.unique_list[i].append(self.unique[j][start:end])
                self.unique_list_cpu[i].append(self.unique_cpu[j][start:end])

    def next(self):
        if self.cnt >= self.num_batch-1:
            self.cnt = 0

        start = self.cnt * self.total_batch_size
        end = start + self.total_batch_size
        inverse = self.inverse[start:end]
        unique = self.unique_list[self.cnt]
        unique_cpu = self.unique_list_cpu[self.cnt]

        self.cnt += 1

        return unique, unique_cpu, inverse

    def max_length(self):
        max_length = self.inverse[-1]
        return max_length


if __name__ == "__main__":
    args = parser.parse_args()

    # kaggle avazu terabyte kaggle_reordered avazu_reordered terabyte_reordered
    dataset = args.dataset
    nDev = args.nDev
    batch_size = args.batch_size

    if dataset == "avazu" or dataset == "avazu_reordered":
        table_num = 20
    else:
        table_num = 26

    train_iter = in_memory_dataloader(batch_size*nDev,0,dataset=dataset)
    unique_gen = unique_generator(batch_size=batch_size, nDev=nDev, table_num=table_num, rank=0, dataloader=train_iter, generate=True, dataset=dataset)



