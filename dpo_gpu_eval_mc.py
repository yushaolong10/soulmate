# dpo_gpu.py
# LoRA DPO for Qwen (chat style)
# 使用 DPO (Direct Preference Optimization) 进行偏好对齐训练
# 支持多卡分布式训练 (DDP)
#
# 架构:
#   - Policy: Merged(Base + SFT) + DPO adapter (训练)
#   - Reference: Merged(Base + SFT)，使用 precompute_ref_log_probs 预计算
#
# 先将 SFT LoRA 合并进基座（merge_and_unload），再在合并后的模型上叠加 DPO adapter。
# 这样 reference = Merged(Base+SFT)，policy = Merged(Base+SFT) + DPO，语义清晰正确。
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
from peft import PeftModel, LoraConfig, get_peft_model
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
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "qwen_lora_adapter_0311_48k_mc")  # SFT LoRA 模型路径
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets0305_train/dpo/dpo_data_rh_31h.jsonl")
EVAL_FILE = os.environ.get("EVAL_FILE", "datasets0305_train/dpo/dpo_data_rt_220.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_dpo_0311_31h_220_mc")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "3e-6"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "5"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "10"))
EVAL_SPLIT_RATIO = float(os.environ.get("EVAL_SPLIT_RATIO", "0.05"))
LOAD_BEST_MODEL_AT_END = os.environ.get("LOAD_BEST_MODEL_AT_END", "1").strip().lower() not in {"0", "false", "no"}


# DPO hyperparameters
BETA = float(os.environ.get("BETA", "0.1"))  # DPO 温度参数

# DPO LoRA hyperparameters (新增的 DPO adapter)
DPO_LORA_R = int(os.environ.get("DPO_LORA_R", "16"))
DPO_LORA_ALPHA = int(os.environ.get("DPO_LORA_ALPHA", "32"))
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


def load_and_prepare_dataset(data_file: str, tokenizer, split_name: str):
    """加载、预处理并过滤 DPO 数据集。"""
    print_rank0(f"🔹 Loading {split_name} dataset from {data_file}...")
    ds = load_dataset("json", data_files={split_name: data_file})[split_name]
    print_rank0(f"   {split_name} raw samples: {len(ds)}")

    ds = ds.map(
        lambda x: preprocess_dpo(x, tokenizer),
        remove_columns=[
            col
            for col in ds.column_names
            if col not in ["prompt", "chosen", "rejected"]
        ],
        num_proc=4,
    )

    original_len = len(ds)
    ds = ds.filter(lambda x: len(x["chosen"]) >= 5 and len(x["rejected"]) >= 5)
    filtered_len = len(ds)
    if filtered_len < original_len and is_main_process():
        print(
            f"   Filtered {original_len - filtered_len} {split_name} samples "
            f"with response < 5 chars"
        )
    print_rank0(f"   {split_name} samples (after filtering): {filtered_len}")
    return ds


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
    # Policy Model: Merged(Base + SFT) + DPO LoRA (训练)
    # 方案A: 先 merge SFT 权重进基座，再叠加 DPO adapter
    #   - reference = Merged(Base+SFT)  ← precompute_ref_log_probs 时禁用 DPO adapter 得到
    #   - policy    = Merged(Base+SFT) + DPO adapter (trainable)
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

    # Step 1: 加载 SFT LoRA 并合并进基座
    print_rank0(f"🔹 Loading & merging SFT LoRA from {SFT_LORA_DIR}...")
    if not os.path.exists(SFT_LORA_DIR):
        raise FileNotFoundError(f"SFT LoRA directory not found: {SFT_LORA_DIR}")

    sft_model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR)
    merged_model = sft_model.merge_and_unload()  # 合并 SFT 权重，返回普通 nn.Module
    merged_model.config.use_cache = False
    print_rank0(f"   ✅ SFT LoRA merged into base model")

    # Step 2: 在合并后的模型上添加 DPO LoRA adapter (仅此 adapter 参与训练)
    print_rank0(f"🔹 Adding DPO LoRA adapter on merged model (trainable)...")
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
    model = get_peft_model(merged_model, dpo_lora_config)
    print_rank0(f"   ✅ DPO LoRA adapter added")

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    # --------
    # Reference Model: 使用 precompute_ref_log_probs，无需加载额外模型
    # --------
    # 使用 precompute_ref_log_probs=True 时，DPOTrainer 会：
    # 1. 训练前临时禁用 DPO adapter（model.disable_adapter()）
    # 2. 用 Merged(Base+SFT) 状态前向计算所有样本的 ref logprobs
    # 3. 将 ref logprobs 缓存到 dataset，训练时直接读取
    # reference = Merged(Base+SFT)，policy = Merged(Base+SFT) + DPO，语义完全正确
    print_rank0(f"\n🔹 Reference logprobs: 将使用 precompute_ref_log_probs 预计算")
    print_rank0(f"   reference = Merged(Base+SFT)，禁用 DPO adapter 后自动得到")
    print_rank0(f"   无需额外加载 ref model，节省显存，避免多卡同步问题")

    # --------
    # Dataset
    # --------
    train_ds = load_and_prepare_dataset(TRAIN_FILE, tokenizer, "train")
    eval_ds = None

    if EVAL_FILE:
        if not os.path.exists(EVAL_FILE):
            raise FileNotFoundError(f"EVAL_FILE not found: {EVAL_FILE}")
        eval_ds = load_and_prepare_dataset(EVAL_FILE, tokenizer, "eval")
    elif EVAL_SPLIT_RATIO > 0:
        split = train_ds.train_test_split(
            test_size=EVAL_SPLIT_RATIO, seed=42, shuffle=True
        )
        train_ds = split["train"]
        eval_ds = split["test"]
        print_rank0(
            f"🔹 Split train/eval from TRAIN_FILE with ratio={EVAL_SPLIT_RATIO:.3f}: "
            f"train={len(train_ds)}, eval={len(eval_ds)}"
        )
    else:
        print_rank0("🔹 No eval dataset: evaluation disabled")

    # 打印样本示例 (仅主进程)
    if is_main_process():
        print("\n📝 Sample example:")
        sample = train_ds[0]
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
    use_eval = eval_ds is not None and len(eval_ds) > 0

    dpo_config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
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
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if use_eval:
        dpo_config_kwargs.update(
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        dpo_config_kwargs.update(
            evaluation_strategy="no",
            load_best_model_at_end=False,
        )

    dpo_config = DPOConfig(**dpo_config_kwargs)

    # --------
    # DPO Trainer
    # --------
    print_rank0("🔹 Initializing DPOTrainer...")
    print_rank0(f"   Base model: {BASE_MODEL}")
    print_rank0(f"   SFT LoRA merged: {SFT_LORA_DIR} → Merged(Base+SFT)")
    print_rank0(f"   DPO LoRA (trainable): r={DPO_LORA_R}, alpha={DPO_LORA_ALPHA}")
    print_rank0(f"   Policy  = Merged(Base+SFT) + DPO adapter")
    print_rank0(f"   Reference = Merged(Base+SFT)  [precompute_ref_log_probs]")
    print_rank0(f"   Beta (temperature): {BETA}")
    print_rank0(f"   Max sequence length: {MAX_SEQ_LEN}")
    print_rank0(f"   Max prompt length: {MAX_PROMPT_LEN}")
    print_rank0(f"   Train samples: {len(train_ds)}")
    if use_eval:
        print_rank0(f"   Eval samples: {len(eval_ds)}")
        print_rank0(
            f"   Eval every {EVAL_STEPS} steps, "
            f"load_best_model_at_end={LOAD_BEST_MODEL_AT_END}"
        )
    else:
        print_rank0("   Eval: disabled")

    # ref_model=None + precompute_ref_log_probs=True：
    # DPOTrainer 训练前调用 model.disable_adapter() 得到 Merged(Base+SFT)
    # 计算并缓存 ref logprobs，训练时恢复 DPO adapter 正常训练
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 不需要额外的 ref model
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
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
