import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "kaggle":
        for idx in range(26):
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/kaggle/access_record/access_" + str(idx) + ".pt")
            sorted, indices = torch.sort(x,descending=True)

            true_index = torch.arange(1,x.shape[0]+1,dtype=torch.long)
            for i in range(x.shape[0]):
                true_index[indices[i]] = i

            torch.save(true_index, '/workspace/SC_artifacts_eval/Access_Index/kaggle/access_index/access_index_'+str(idx)+'.pt')

    elif dataset == "avazu":
        for idx in range(20):
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/avazu/access_record/access_" + str(idx) + ".pt")
            sorted, indices = torch.sort(x,descending=True)

            true_index = torch.arange(1,x.shape[0]+1,dtype=torch.long)
            for i in range(x.shape[0]):
                true_index[indices[i]] = i

            torch.save(true_index, '/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_'+str(idx)+'.pt')


    elif dataset == "terabyte":
        for idx in range(26):
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/terabyte/access_record/access_" + str(idx) + ".pt")
            sorted, indices = torch.sort(x,descending=True)

            true_index = torch.arange(1,x.shape[0]+1,dtype=torch.long)
            for i in range(x.shape[0]):
                true_index[indices[i]] = i

            torch.save(true_index, '/workspace/SC_artifacts_eval/Access_Index/terabyte/access_index/access_index_'+str(idx)+'.pt')
