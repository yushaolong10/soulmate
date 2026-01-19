# finetune_gpu_weighted_loss.py
# LoRA SFT for Qwen (chat style) with assistant-weighted token loss
# - System/User tokens participate in loss with lower weights
# - Assistant tokens use higher weight (default 1.0)
#
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft torch modelscope
#
# Data format (JSONL):
# {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]}
#
# Run:
#   CUDA_VISIBLE_DEVICES=1 python finetune_gpu_weighted_loss.py
#
# Output:
#   ./qwen_lora_adapter_xxx  (LoRA adapter weights + tokenizer)

import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
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

# Loss weights (assistant-weighted)
SYS_W = float(os.environ.get("SYS_W", "0.05"))
USER_W = float(os.environ.get("USER_W", "0.2"))
ASSIST_W = float(os.environ.get("ASSIST_W", "1.0"))


# -----------------------------
# Helper: build chat text and weighted labels
# -----------------------------
def build_text_and_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    max_seq_len: int,
    sys_w: float = 0.05,
    user_w: float = 0.2,
    assist_w: float = 1.0,
) -> Dict[str, Any]:
    """
    构建 input_ids / labels / loss_weight
    - labels: 所有非 padding token 都参与 loss（padding 用 -100）
    - loss_weight: token 级别权重，system/user 低、assistant 高
    """
    input_ids: List[int] = []
    labels: List[int] = []
    loss_weight: List[float] = []

    prefix_messages: List[Dict[str, str]] = []

    def role_weight(role: str) -> float:
        if role == "assistant":
            return assist_w
        if role == "user":
            return user_w
        if role == "system":
            return sys_w
        return user_w  # 兜底

    for i, msg in enumerate(messages):
        prefix_messages.append(msg)

        # 当前前缀完整文本
        full_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # 新增 token（与上一前缀做 diff）
        if i == 0:
            new_ids = full_ids
        else:
            prev_text = tokenizer.apply_chat_template(
                prefix_messages[:-1], tokenize=False, add_generation_prompt=False
            )
            prev_ids = tokenizer(prev_text, add_special_tokens=False).input_ids
            new_ids = full_ids[len(prev_ids) :]

        w = role_weight(msg.get("role", "user"))

        input_ids.extend(new_ids)
        labels.extend(new_ids)  # 关键：不再把 user/system 全部 mask 掉
        loss_weight.extend([w] * len(new_ids))

    # truncate
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        loss_weight = loss_weight[:max_seq_len]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weight": loss_weight,
    }


@dataclass
class DataCollatorForCausalLM:
    """
    将 batch 中的样本 padding 到相同长度，并携带 loss_weight
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
        w_batch = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            input_ids = f["input_ids"] + [pad_id] * pad_len
            attn = f["attention_mask"] + [0] * pad_len

            # labels：pad 部分用 -100，确保不算 loss
            labels = f["labels"] + [-100] * pad_len

            # loss_weight：pad 部分给 0
            w = f["loss_weight"] + [0.0] * pad_len

            input_ids_batch.append(input_ids)
            attn_batch.append(attn)
            labels_batch.append(labels)
            w_batch.append(w)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "loss_weight": torch.tensor(w_batch, dtype=torch.float32),
        }


class WeightedLossTrainer(Trainer):
    """
    token-level weighted cross entropy
    - shift logits/labels by 1 (causal LM)
    - ignore_index = -100 (padding)
    - multiply token loss by loss_weight
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")  # [B, T]
        loss_weight = inputs.pop("loss_weight")  # [B, T]
        outputs = model(**inputs)

        logits = outputs.logits  # [B, T, V]

        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # 确保 loss_weight 在正确的设备上，并转换为与 logits 相同的精度
        shift_w = (
            loss_weight[:, 1:].to(device=logits.device, dtype=logits.dtype).contiguous()
        )

        vocab_size = shift_logits.size(-1)

        # token-level CE (no reduction)
        # 使用 float32 计算 loss 以提高数值稳定性
        ce = F.cross_entropy(
            shift_logits.view(-1, vocab_size).float(),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(
            shift_labels
        )  # [B, T-1]

        weighted = ce * shift_w.float()

        # denominator: sum of weights on valid tokens
        valid = (shift_labels != -100).float()
        denom = (valid * shift_w.float()).sum().clamp_min(1e-6)

        loss = weighted.sum() / denom

        return (loss, outputs) if return_outputs else loss


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

    print(f"🔧 Loss weights: SYS_W={SYS_W}, USER_W={USER_W}, ASSIST_W={ASSIST_W}")

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
        return build_text_and_labels(
            tokenizer, messages, MAX_SEQ_LEN, SYS_W, USER_W, ASSIST_W
        )

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=4)

    # 打印样本统计
    total_tokens = sum(len(x["input_ids"]) for x in ds)

    # 统计各 role 权重 token（基于 loss_weight 近似）
    # 注意：这是 token 数量，不是 loss。仅用于 sanity check
    assist_tokens = 0
    user_tokens = 0
    sys_tokens = 0
    for x in ds:
        for w in x["loss_weight"]:
            if abs(w - ASSIST_W) < 1e-9:
                assist_tokens += 1
            elif abs(w - USER_W) < 1e-9:
                user_tokens += 1
            elif abs(w - SYS_W) < 1e-9:
                sys_tokens += 1

    print(f"   Total tokens: {total_tokens:,}")
    if total_tokens > 0:
        print(
            f"   Token mix (approx): assistant={assist_tokens:,} ({assist_tokens/total_tokens*100:.1f}%), "
            f"user={user_tokens:,} ({user_tokens/total_tokens*100:.1f}%), "
            f"system={sys_tokens:,} ({sys_tokens/total_tokens*100:.1f}%)"
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
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForCausalLM(tokenizer)

    # --------
    # Trainer
    # --------
    print("🔹 Initializing WeightedLossTrainer...")
    trainer = WeightedLossTrainer(
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
