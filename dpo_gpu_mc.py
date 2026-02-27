# dpo_gpu.py
# LoRA DPO for Qwen (chat style)
# 使用 DPO (Direct Preference Optimization) 进行偏好对齐训练
# 支持多卡分布式训练 (DDP)
#
# 架构:
#   - Policy: Base + SFT adapter (冻结) + DPO adapter (训练)
#   - Reference: 使用 precompute_ref_log_probs 预计算，无需额外加载 ref model
#
# 这种方式可以保留 SFT 的效果，同时只训练新的 DPO adapter
#
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft torch trl
#
# Data format (JSONL):
# {
#   "prompt": [...messages...],
#   "chosen": "正确回复",
#   "rejected": "负样本回复",
#   "rejected_type": "负样本类型"
# }
#
# Run (单卡):
#   CUDA_VISIBLE_DEVICES=0 python dpo_gpu_mc.py
#
# Run (多卡 - 推荐):
#   # 方式1: 使用 torchrun (推荐)
#   torchrun --nproc_per_node=2 dpo_gpu_mc.py
#
#   # 方式2: 指定 GPU
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_gpu_mc.py
#
#   # 方式3: 使用 accelerate
#   accelerate launch --num_processes=2 dpo_gpu_mc.py
#
# Output:
#   ./qwen_lora_dpo_0211  (LoRA adapter weights + tokenizer)
#
# 多卡训练说明:
#   - 使用 precompute_ref_log_probs=True 预计算 ref logprobs
#   - 无需加载额外的 ref model，节省显存
#   - 支持 DDP 分布式训练

import os
import math
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from datasets import load_dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig


def is_main_process() -> bool:
    """判断是否为主进程 (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


class NaNDetectorCallback(TrainerCallback):
    """训练过程中检测 NaN/Inf 的回调"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", None)
            if loss is not None:
                if math.isnan(loss) or math.isinf(loss):
                    print(f"\n❌ 检测到异常 loss: {loss}")
                    print(f"   Step: {state.global_step}")
                    print(f"   停止训练...")
                    control.should_training_stop = True
        return control


# -----------------------------
# User config
# -----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-14B")
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "qwen_lora_adapter_0227_1w_mc")  # SFT LoRA 模型路径
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets0211_train/dpo/dpo_1700.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_dpo_0227_1700_mc")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-5"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "20"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "2"))

# DPO hyperparameters
BETA = float(os.environ.get("BETA", "0.1"))  # DPO 温度参数

# DPO LoRA hyperparameters (新增的 DPO adapter)
DPO_LORA_R = int(os.environ.get("DPO_LORA_R", "8"))
DPO_LORA_ALPHA = int(os.environ.get("DPO_LORA_ALPHA", "16"))
DPO_LORA_DROPOUT = float(os.environ.get("DPO_LORA_DROPOUT", "0.05"))


def build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    将 messages 列表转换为 prompt 文本
    注意：enable_thinking=False 与推理时保持一致
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def preprocess_dpo(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    预处理 DPO 数据

    输入格式:
    {
        "prompt": [...messages...],
        "chosen": "正确回复",
        "rejected": "负样本回复"
    }

    输出格式 (DPOTrainer 需要):
    {
        "prompt": "prompt 文本",
        "chosen": "chosen 回复",
        "rejected": "rejected 回复"
    }
    """
    messages = example.get("prompt", [])
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # 构建 prompt 文本
    prompt_text = build_prompt_text(tokenizer, messages)

    return {
        "prompt": prompt_text,
        "chosen": chosen,
        "rejected": rejected,
    }


def main():
    # --------
    # Device check & 多卡初始化
    # --------
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow on CPU!")
    else:
        if is_main_process():
            print(f"✅ CUDA available: {num_gpus} GPU(s)")
            for i in range(num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
                )
            if num_gpus > 1:
                print(f"\n🚀 多卡训练模式: {num_gpus} GPUs")
                print(f"   使用 DDP + precompute_ref_log_probs 进行分布式训练")

    # --------
    # Tokenizer
    # --------
    print_rank0(f"🔹 Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO 训练时需要 left padding
    tokenizer.padding_side = "left"

    # --------
    # Policy Model: Base + SFT LoRA (冻结) + DPO LoRA (训练)
    # --------
    print_rank0(f"🔹 Loading base model from {BASE_MODEL}...")

    # 多卡训练: 不使用 device_map="auto"，让 Trainer 自动处理
    if num_gpus > 1:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,  # 让 Trainer 处理分布式
            # attn_implementation="flash_attention_2",
        )
        base_model = base_model.to(f"cuda:{local_rank}")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )

    base_model.config.use_cache = False

    # 加载 SFT LoRA adapter (冻结)
    print_rank0(f"🔹 Loading SFT LoRA adapter from {SFT_LORA_DIR} (frozen)...")
    if not os.path.exists(SFT_LORA_DIR):
        raise FileNotFoundError(f"SFT LoRA directory not found: {SFT_LORA_DIR}")

    model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR, is_trainable=False)
    print_rank0(f"   ✅ SFT LoRA loaded (frozen)")

    # 在 SFT LoRA 基础上添加新的 DPO LoRA adapter (训练)
    print_rank0(f"🔹 Adding DPO LoRA adapter (trainable)...")
    print_rank0(
        f"   r={DPO_LORA_R}, alpha={DPO_LORA_ALPHA}, scaling={DPO_LORA_ALPHA/DPO_LORA_R:.2f}"
    )
    dpo_lora_config = LoraConfig(
        r=DPO_LORA_R,
        lora_alpha=DPO_LORA_ALPHA,
        lora_dropout=DPO_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    # 添加新的 adapter，命名为 "dpo"
    model.add_adapter("dpo", dpo_lora_config)
    # 设置 "dpo" adapter 为活动 adapter 并可训练
    model.set_adapter("dpo")
    print_rank0(f"   ✅ DPO LoRA adapter added")

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    # --------
    # Reference Model: 使用 precompute_ref_log_probs，无需加载额外模型
    # --------
    # 注意：使用 precompute_ref_log_probs=True 时，DPOTrainer 会：
    # 1. 在训练前，使用当前模型（关闭 DPO adapter）计算 ref logprobs
    # 2. 将 ref logprobs 缓存到 dataset 中
    # 3. 训练时只使用 policy model，从缓存读取 ref logprobs
    # 这样就不需要额外加载 ref model，节省显存，也避免多卡同步问题
    print_rank0(f"\n🔹 Reference logprobs: 将使用 precompute_ref_log_probs 预计算")
    print_rank0(f"   无需加载额外的 reference model")
    print_rank0(f"   预计算时会临时禁用 DPO adapter，使用 SFT 状态计算 ref logprobs")

    # --------
    # Dataset
    # --------
    print_rank0(f"🔹 Loading dataset from {TRAIN_FILE}...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
    print_rank0(f"   Total samples: {len(ds)}")

    # 预处理数据
    ds = ds.map(
        lambda x: preprocess_dpo(x, tokenizer),
        remove_columns=[
            col
            for col in ds.column_names
            if col not in ["prompt", "chosen", "rejected"]
        ],
        num_proc=4,
    )

    # 过滤过短的样本（可能导致数值问题）
    original_len = len(ds)
    ds = ds.filter(lambda x: len(x["chosen"]) >= 5 and len(x["rejected"]) >= 5)
    filtered_len = len(ds)
    if filtered_len < original_len and is_main_process():
        print(
            f"   Filtered {original_len - filtered_len} samples with response < 5 chars"
        )
    print_rank0(f"   Total samples (after filtering): {filtered_len}")

    # 打印样本示例 (仅主进程)
    if is_main_process():
        print("\n📝 Sample example:")
        sample = ds[0]
        print(f"   Prompt (truncated): {sample['prompt'][:200]}...")
        print(f"   Chosen: {sample['chosen'][:100]}...")
        print(f"   Rejected: {sample['rejected'][:100]}...")

    # --------
    # Trainable parameters info
    # --------
    print_rank0("\n🔹 Trainable parameters (只训练 DPO adapter):")
    if is_main_process():
        model.print_trainable_parameters()

    # --------
    # DPO Config
    # --------
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        # DPO 参数
        beta=BETA,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        # 预计算 ref logprobs - 关键参数！
        # 这样就不需要额外的 ref_model，避免多卡同步问题
        precompute_ref_log_probs=True,
        # padding 设置
        label_pad_token_id=-100,
        # 精度设置
        bf16=True,
        fp16=False,
        # 优化器
        optim="adamw_torch_fused",
        # 梯度裁剪，防止梯度爆炸
        max_grad_norm=1.0,
        # 其他设置
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # --------
    # DPO Trainer
    # --------
    print_rank0("🔹 Initializing DPOTrainer...")
    print_rank0(f"   Base model: {BASE_MODEL}")
    print_rank0(f"   SFT LoRA (frozen): {SFT_LORA_DIR}")
    print_rank0(f"   DPO LoRA (trainable): r={DPO_LORA_R}, alpha={DPO_LORA_ALPHA}")
    print_rank0(f"   Beta (temperature): {BETA}")
    print_rank0(f"   Max sequence length: {MAX_SEQ_LEN}")
    print_rank0(f"   Max prompt length: {MAX_PROMPT_LEN}")
    print_rank0(f"   precompute_ref_log_probs: True (无需额外 ref model)")

    # 注意：ref_model=None，使用 precompute_ref_log_probs 预计算
    # DPOTrainer 会自动：
    # 1. 检测到 ref_model=None 且 precompute_ref_log_probs=True
    # 2. 临时禁用 trainable adapter (DPO adapter)
    # 3. 使用 frozen 状态 (SFT) 计算 ref logprobs
    # 4. 缓存到 dataset 中
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 不需要额外的 ref model
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[NaNDetectorCallback()],  # 添加 NaN 检测
    )

    # --------
    # Train
    # --------
    print_rank0("🚀 Starting DPO training...")
    if num_gpus > 1:
        print_rank0(f"   分布式训练: {num_gpus} GPUs")

    trainer.train()

    # Save adapter + tokenizer (仅主进程保存)
    if is_main_process():
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\n✅ Done. DPO LoRA adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
