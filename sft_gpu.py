# sft_gpu.py
# LoRA SFT for Qwen (chat style) - 只训练 label 部分
# 适用于 format_data.py 生成的数据格式
#
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft torch
#
# Data format (JSONL):
# {
#   "messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."},...,{"role":"user","content":"..."}],
#   "label": "AI回复内容"
# }
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 python sft_gpu.py
#
# Output:
#   ./qwen_lora_adapter_0211  (LoRA adapter weights + tokenizer)
#
# 重要修复 (2026-02-26):
#   - 降低 LoRA alpha 从 32 到 16，使 scaling=1，避免 bfloat16 数值溢出
#   - 降低学习率从 1e-4 到 3e-5
#   - 添加梯度裁剪 max_grad_norm=1.0
#   - 添加训练过程数值检查

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model


# -----------------------------
# User config
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-14B")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets0211_train/train/train_h_10000.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_adapter_0226_1h")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "3e-5"))  # 降低学习率，避免训练不稳定

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "5"))

# LoRA hyperparameters
# 关键：alpha/r 比例决定 LoRA 的 scaling factor
# - 旧配置 r=16, alpha=32 -> scaling=2.0 -> 在 bfloat16 下产生 NaN
# - 新配置 r=16, alpha=16 -> scaling=1.0 -> 数值稳定
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))  # 必须 <= r，避免 scaling > 1
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))


# -----------------------------
# Helper: build input_ids and labels
# -----------------------------
def build_input_and_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    label: str,
    max_seq_len: int,
) -> Dict[str, Any]:
    """
    构建 input_ids 和 labels。

    数据格式:
    - messages: 包含 system + 多轮 user/assistant + 最后一个 user
    - label: 最后一轮 AI 的回复

    训练策略:
    - messages 部分作为上下文，不计算 loss (labels = -100)
    - label 部分计算 loss
    """
    # 1. 构建完整的对话（包含 label 作为最后一个 assistant 回复）
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": label})

    # 2. 获取完整对话的文本和 token
    # 注意：enable_thinking=False 与推理时保持一致，避免 <think> 标签干扰
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # 3. 获取不含 label 的 messages 部分（带 generation prompt）
    # 注意：enable_thinking=False 与推理时保持一致
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    # 4. 构建 labels
    # prompt 部分不计算 loss，label 部分计算 loss
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # 确保 input_ids 和 labels 长度一致
    input_ids = full_ids
    if len(labels) != len(input_ids):
        # 如果长度不一致，使用 full_ids 作为 labels，只 mask prompt 部分
        labels = [-100] * prompt_len + input_ids[prompt_len:]

    # 5. 截断
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# -----------------------------
# 训练过程中检测 NaN/Inf 的回调
# -----------------------------
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

        
@dataclass
class DataCollatorForCausalLM:
    """
    将 batch 中的样本 padding 到相同长度
    """

    tokenizer: Any
    pad_to_multiple_of: int = 8  # 对齐到 8 的倍数，提升 GPU 效率

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        # 对齐到 pad_to_multiple_of 的倍数
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id

        input_ids_batch = []
        attn_batch = []
        labels_batch = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            input_ids = f["input_ids"] + [pad_id] * pad_len
            attn = f["attention_mask"] + [0] * pad_len
            labels = f["labels"] + [-100] * pad_len

            input_ids_batch.append(input_ids)
            attn_batch.append(attn)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def main():
    # --------
    # Device check
    # --------
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow on CPU!")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # --------
    # Tokenizer
    # --------
    print(f"🔹 Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------
    # Model
    # --------
    print(f"🔹 Loading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2",  # 需要安装 flash-attn
    )
    model.config.use_cache = False

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    # --------
    # Dataset
    # --------
    print(f"🔹 Loading dataset from {TRAIN_FILE}...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
    print(f"   Total samples: {len(ds)}")

    def preprocess(example):
        messages = example.get("messages", [])
        label = example.get("label", "")
        if not messages or not label:
            raise ValueError("Each example must have non-empty `messages` and `label`.")
        return build_input_and_labels(tokenizer, messages, label, MAX_SEQ_LEN)

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=4)

    # 打印样本统计
    total_tokens = sum(len(x["input_ids"]) for x in ds)
    train_tokens = sum(sum(1 for l in x["labels"] if l != -100) for x in ds)
    print(f"   Total tokens: {total_tokens:,}")
    print(
        f"   Trainable tokens (label only): {train_tokens:,} ({train_tokens/total_tokens*100:.1f}%)"
    )

    # --------
    # LoRA config
    # --------
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
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

    # 应用 LoRA
    print("🔹 Applying LoRA...")
    print(f"   r={LORA_R}, alpha={LORA_ALPHA}, scaling={LORA_ALPHA/LORA_R:.2f}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 验证模型初始状态
    print("🔹 Verifying model initialization...")
    with torch.no_grad():
        test_input = tokenizer("你好", return_tensors="pt").to(model.device)
        test_output = model(**test_input)
        test_logits = test_output.logits
        print(f"   Initial logits range: [{test_logits.min().item():.2f}, {test_logits.max().item():.2f}]")
        if torch.isnan(test_logits).any() or torch.isinf(test_logits).any():
            raise ValueError("❌ Model produces NaN/Inf before training! Check LoRA config.")
    print("   ✅ Model verification passed")

    # --------
    # Training args
    # --------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        # 精度设置
        bf16=True,
        fp16=False,
        # 优化器
        optim="adamw_torch_fused",
        # 其他设置
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # 梯度裁剪，防止梯度爆炸
        max_grad_norm=1.0,
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForCausalLM(tokenizer)

    # --------
    # Trainer
    # --------
    print("🔹 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        callbacks=[NaNDetectorCallback()],  # 添加 NaN 检测
    )

    # --------
    # Train
    # --------
    print("🚀 Starting training...")
    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Done. LoRA adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
