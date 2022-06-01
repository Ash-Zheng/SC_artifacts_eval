# Split dataset into train(85%) and valid(15%)
import os

if __name__ == "__main__":
    train_subset = open("/workspace/SC_artifacts_eval/processed_data/kaggle/train_subset.txt",'w')
    val_subset = open("/workspace/SC_artifacts_eval/processed_data/kaggle/val_subset.txt",'w')

    row_cnt = 0
    with open("/workspace/SC_artifacts_eval/dlrm_dataset/kaggle/train.txt", 'r') as file_to_read:
        for line in file_to_read:
            row_cnt += 1

    row_cnt = int(row_cnt * 0.9)

    with open("/workspace/SC_artifacts_eval/dlrm_dataset/kaggle/train.txt", 'r') as file_to_read:
        i = 0
        for line in file_to_read:
            if i<=row_cnt:
                train_subset.write(line)
            else:
                val_subset.write(line)
            i += 1
