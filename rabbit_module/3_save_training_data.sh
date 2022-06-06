# ==================== kaggle ======================
python save_dataset_kaggle.py --reorder=1

python save_dataset_kaggle.py --reorder=0

# ===================== avazu ======================
python save_dataset_avazu.py --reorder=1

python save_dataset_avazu.py --reorder=0

# ==================== terabyte ====================
python save_dataset_terabyte.py --reorder=1

python save_dataset_terabyte.py --reorder=0


# ==================== generate unique ====================
mkdir /workspace/SC_artifacts_eval/Access_Index/kaggle_reordered/
mkdir /workspace/SC_artifacts_eval/Access_Index/kaggle_reordered/unique/
mkdir /workspace/SC_artifacts_eval/Access_Index/avazu_reordered/
mkdir /workspace/SC_artifacts_eval/Access_Index/avazu_reordered/unique/
mkdir /workspace/SC_artifacts_eval/Access_Index/terabyte_reordered/
mkdir /workspace/SC_artifacts_eval/Access_Index/terabyte_reordered/unique/

python unique_generator.py --dataset=kaggle_reordered --nDev=1
python unique_generator.py --dataset=avazu_reordered --nDev=1
python unique_generator.py --dataset=terabyte_reordered --nDev=1

python unique_generator.py --dataset=kaggle_reordered --nDev=4
python unique_generator.py --dataset=avazu_reordered --nDev=4
python unique_generator.py --dataset=terabyte_reordered --nDev=4

python unique_generator.py --dataset=kaggle --nDev=1 --batch_size=8192
python unique_generator.py --dataset=kaggle_reordered --nDev=1 --batch_size=8192