# ======================= Kaggle =====================
python ELRec_multiGPU_train.py --dataset=kaggle --nDev=1

python ELRec_multiGPU_train.py --dataset=kaggle --nDev=4

python facebook_dlrm_multigpu.py --dataset=kaggle --nDev=1

python facebook_dlrm_multigpu.py --dataset=kaggle --nDev=4

# ======================= Avazu =====================
python ELRec_multiGPU_train.py --dataset=avazu --nDev=1

python ELRec_multiGPU_train.py --dataset=avazu --nDev=4

python facebook_dlrm_multigpu.py --dataset=avazu --nDev=1

python facebook_dlrm_multigpu.py --dataset=avazu --nDev=4

# ==================== terabyte =====================
python ELRec_multiGPU_train.py --dataset=terabyte --nDev=1

python ELRec_multiGPU_train.py --dataset=terabyte --nDev=4

python facebook_dlrm_multigpu.py --dataset=terabyte --nDev=4