import torch
import numpy as np

class random_dataloader:
    def __init__(self, batch_size, table_length, dense_num, device):
        # torch.manual_seed(1234)
        # np.random.seed(1234)
        self.batch_size = batch_size
        self.table_num = len(table_length)
        self.table_length = table_length
        self.dense_num = dense_num
        self.device = device

    def next(self):
        label = torch.randint(0, 2, (self.batch_size,1), dtype=torch.float, device=self.device)
        dense = torch.rand(self.batch_size, self.dense_num, device=self.device)
        sparse = []
        for i in range(self.table_num):
            tmp = torch.randint(0, self.table_length[i], (self.batch_size,1), device=self.device)
            sparse.append(tmp)

        return label, torch.stack(sparse), dense


class in_memory_dataloader:
    def __init__(self, batch_size, device, dataset="kaggle"):
        # torch.manual_seed(1234)
        # np.random.seed(1234)
        self.batch_size = batch_size
        base_dir = "/workspace/SC_artifacts_eval/Access_Index/"

        if dataset == "kaggle":
            self.dense = torch.load(base_dir+"kaggle/training_data/dense.pt").to(device)
            self.sparse = torch.load(base_dir+"kaggle/training_data/sparse.pt").to(device)
            self.label = torch.load(base_dir+"kaggle/training_data/label.pt").to(device)
        elif dataset == "avazu":
            self.dense = torch.load(base_dir+"avazu/training_data/dense.pt").to(device)
            self.sparse = torch.load(base_dir+"avazu/training_data/sparse.pt").to(device)
            self.label = torch.load(base_dir+"avazu/training_data/label.pt").to(device)
        elif dataset == "terabyte":
            self.dense = torch.load(base_dir+"terabyte/training_data/dense.pt").to(device)
            self.sparse = torch.load(base_dir+"terabyte/training_data/sparse.pt").to(device)
            self.label = torch.load(base_dir+"terabyte/training_data/label.pt").to(device)
        elif dataset == "kaggle_reordered":
            self.dense = torch.load(base_dir+"kaggle/training_data/reordered_dense.pt").to(device)
            self.sparse = torch.load(base_dir+"kaggle/training_data/reordered_sparse.pt").to(device)
            self.label = torch.load(base_dir+"kaggle/training_data/reordered_label.pt").to(device)
        elif dataset == "avazu_reordered":
            self.dense = torch.load(base_dir+"avazu/training_data/reordered_dense.pt").to(device)
            self.sparse = torch.load(base_dir+"avazu/training_data/reordered_sparse.pt").to(device)
            self.label = torch.load(base_dir+"avazu/training_data/reordered_label.pt").to(device)
        elif dataset == "terabyte_reordered":
            self.dense = torch.load(base_dir+"terabyte/training_data/reordered_dense.pt").to(device)
            self.sparse = torch.load(base_dir+"terabyte/training_data/reordered_sparse.pt").to(device)
            self.label = torch.load(base_dir+"terabyte/training_data/reordered_label.pt").to(device)
        else:
            print("Error dataset name")
            exit(0)

        self.num_batch = self.label.shape[0] / self.batch_size
        self.cnt = 0

    def next(self):
        if self.cnt >= self.num_batch-1:
            self.cnt = 0

        start = self.cnt * self.batch_size
        end = start + self.batch_size

        desnse = self.dense[start:end]
        label = self.label[start:end]
        sparse = self.sparse[:,start:end]
        self.cnt += 1

        return label, sparse, desnse
    
    def reset(self):
        self.cnt = 0

if __name__ == "__main__":
    # dataloader = random_dataloader(3, [10, 10], 4)

    # sparse, dense, label = dataloader.next()

    # print(sparse.shape)
    # print(dense)
    # print(label)

    dataloader = in_memory_dataloader(4096,0)
   
    label, sparse, dense = dataloader.next()

    print(sparse)
    print(sparse.shape)
