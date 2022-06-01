# ==================== kaggle ======================
python generate_index_bijection.py --dataset=kaggle --table_idx 2

python generate_index_bijection.py --dataset=kaggle --table_idx 3

python generate_index_bijection.py --dataset=kaggle --table_idx 11

python generate_index_bijection.py --dataset=kaggle --table_idx 15

python generate_index_bijection.py --dataset=kaggle --table_idx 20

# ==================== avazu ======================
python generate_index_bijection.py --dataset=avazu --table_idx 7

python small_generate_index_bijection.py --dataset=avazu --table_idx 8 --batch_num 65536

# ==================== terabyte ======================
python small_generate_index_bijection.py --dataset=terabyte --table_idx 0 --batch_num 65536

python small_generate_index_bijection.py --dataset=terabyte --table_idx 9 --batch_num 65536

python small_generate_index_bijection.py --dataset=terabyte --table_idx 10 --batch_num 65536

python small_generate_index_bijection.py --dataset=terabyte --table_idx 19 --batch_num 65536

python small_generate_index_bijection.py --dataset=terabyte --table_idx 20 --batch_num 65536

python small_generate_index_bijection.py --dataset=terabyte --table_idx 21 --batch_num 65536