torchx run -s local_cwd dist.ddp -j 1x2 --script torchrec/single_table_test.py

torchx run -s local_cwd dist.ddp -j 1x4 --script torchrec/single_table_test.py
