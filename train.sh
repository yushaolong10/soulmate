
################
# SFT

# 小模型单卡
#CUDA_VISIBLE_DEVICES=0 python sft_gpu.py

# 单卡模型8bit量化
#CUDA_VISIBLE_DEVICES=0 python sft_gpu_8bit.py

# 多卡 - 方式2: 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sft_gpu_mc.py

################
# DPO

# 单卡模型8bit量化
#CUDA_VISIBLE_DEVICES=0 python dpo_gpu_8bit.py

# 多卡 - 方式2: 指定 GPU
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_gpu_mc.py
