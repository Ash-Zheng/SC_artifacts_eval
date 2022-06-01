import os
import pathlib

import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU

from nvtabular.utils import download_file
from sklearn.model_selection import train_test_split
import csv


if __name__ == "__main__":
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = "/workspace/SC_artifacts_eval/"
    INPUT_PATH  = os.path.join(BASE_DIR, "dlrm_dataset/avazu")
    total_length = 40428968
    input_file = os.path.join(INPUT_PATH, "avazu")

    
    train_subset = open("/workspace/SC_artifacts_eval/processed_data/avazu/train_subset.txt",'w')
    val_subset = open("/workspace/SC_artifacts_eval/processed_data/avazu/val_subset.txt",'w')
    # # add a column day
    # # ['0', '21', '00', 
    # # '1005', '0', '1fbe01fe', 
    # # 'f3845767', '28905ebd', 'ecad2386', 
    # # '7801e8d9', '07d7df22', 'a99f214a', 
    # # '96809ac8', '711ee120', '1', '0', 
    # # '15704', '320', '50', 
    # # '1722', '0', '35', '100084']

    print("start")

    with open(input_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        idx = 0
        for row in spamreader:
            if i == 0:
                i += 1
                continue
            x = row[0].split(',')[1:]
            time = x[1][6:]
            day = x[1][4:6]
            x[1] = time
            x.insert(1,day)
            y = "\t".join(x)

            if i < 10:
                i += 1
                train_subset.write(y)
                train_subset.write('\n')
            else:
                i = 1
                val_subset.write(y)
                val_subset.write('\n')
            
            idx += 1
            if idx % 1024 == 0:
                print(idx)
