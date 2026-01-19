# finetune_gpu.py
# LoRA SFT for Qwen (chat style) with response-only loss (train only assistant tokens)
# Optimized for Linux CUDA training
#
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft torch
#
# Data format (JSONL):
# {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]}
# 支持多轮对话：一条 system + 多组 user/assistant
#
# Run:
#   CUDA_VISIBLE_DEVICES=1 python finetune_gpu.py
#
# Output:
#   ./qwen_lora_adapter  (LoRA adapter weights + tokenizer)

import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from modelscope import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -----------------------------
# User config
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets/train_0119_s.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_adapter_0119_s")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "1e-4"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "5"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "1"))

# LoRA hyperparameters
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))


# -----------------------------
# Helper: build chat text and response-only labels
# -----------------------------
def build_text_and_labels(
    tokenizer, messages: List[Dict[str, str]], max_seq_len: int
) -> Dict[str, Any]:
    """
    构建 input_ids 和 labels，只对 assistant 的回复计算 loss。

    策略：逐条消息构建，对每条消息计算其对应的 token 范围，
    只有 assistant 消息的 token 才会被设置为有效 label。
    """
    input_ids: List[int] = []
    labels: List[int] = []

    prefix_messages: List[Dict[str, str]] = []

    for i, msg in enumerate(messages):
        prefix_messages.append(msg)

        # 获取当前前缀的完整文本
        full_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # 计算新增的 token
        if i == 0:
            new_ids = full_ids
        else:
            prev_text = tokenizer.apply_chat_template(
                prefix_messages[:-1], tokenize=False, add_generation_prompt=False
            )
            prev_ids = tokenizer(prev_text, add_special_tokens=False).input_ids
            new_ids = full_ids[len(prev_ids) :]

        input_ids.extend(new_ids)

        # Labels 策略：只有 assistant 消息的 token 参与训练
        if msg["role"] == "assistant":
            labels.extend(new_ids)
        elif msg["role"] == "user":
            # 给 user 一点点 loss
            labels.extend(new_ids)  # 或部分
        else:
            labels.extend([-100] * len(new_ids))


    # 截断
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


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
        if not messages:
            raise ValueError("Each example must have a non-empty `messages` list.")
        return build_text_and_labels(tokenizer, messages, MAX_SEQ_LEN)

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=4)

    # 打印样本统计
    total_tokens = sum(len(x["input_ids"]) for x in ds)
    train_tokens = sum(sum(1 for l in x["labels"] if l != -100) for x in ds)
    print(f"   Total tokens: {total_tokens:,}")
    print(
        f"   Trainable tokens (assistant only): {train_tokens:,} ({train_tokens/total_tokens*100:.1f}%)"
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    # 应用 LoRA
    print("🔹 Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
