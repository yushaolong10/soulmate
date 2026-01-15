# finetune.py
# QLoRA SFT for Qwen (chat style) with response-only loss (train only assistant tokens)
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft trl bitsandbytes torch
#
# Data format (JSONL):
# {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
#
# Run:
#   python finetune.py
#
# Output:
#   ./qwen_lora_adapter  (LoRA adapter weights + tokenizer)

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset
from modelscope import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------------
# User config
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets/train_0115_x.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_adapter_0115_x")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "10"))

# If you have limited VRAM, keep these as-is.
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# -----------------------------
# Helper: build chat text and a "response-only" label mask
# -----------------------------


def _ensure_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs = example.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("Each example must have a non-empty `messages` list.")
    # basic sanity: each item has role/content
    for m in msgs:
        if "role" not in m or "content" not in m:
            raise ValueError("Each message must contain `role` and `content`.")
    return msgs


def build_text_and_labels(
    tokenizer: AutoTokenizer, messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Create:
      - input_ids/attention_mask for full conversation text
      - labels where only assistant tokens are trained, others set to -100
    Strategy:
      We build the conversation turn by turn. For each assistant message, we compute
      which token span corresponds to that assistant content and enable labels only there.
    """
    # We’ll tokenize incrementally to know boundaries.
    input_ids: List[int] = []
    labels: List[int] = []

    # Some tokenizers need this for chat template; Qwen usually supports it.
    # We'll rely on apply_chat_template to keep formatting consistent.
    # We'll append messages progressively and track token counts.

    # Build tokenized prefix for each step
    prefix_messages: List[Dict[str, str]] = []
    for i, msg in enumerate(messages):
        prefix_messages.append(msg)

        # Tokenize full prefix (up to i)
        full_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # Determine newly added tokens from previous step
        if i == 0:
            new_ids = full_ids
            prev_len = 0
        else:
            prev_text = tokenizer.apply_chat_template(
                prefix_messages[:-1], tokenize=False, add_generation_prompt=False
            )
            prev_ids = tokenizer(prev_text, add_special_tokens=False).input_ids
            prev_len = len(prev_ids)
            new_ids = full_ids[prev_len:]

        # Append tokens
        input_ids.extend(new_ids)

        # Labels policy:
        # - If current msg is assistant: labels for NEW tokens are enabled (same as token ids)
        # - Else: labels for NEW tokens are -100
        if msg["role"] == "assistant":
            labels.extend(new_ids)
        else:
            labels.extend([-100] * len(new_ids))

    # Truncate
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLMWithLabels:
    """
    Pad input_ids/attention_mask/labels to batch max length.
    """

    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids_batch, attn_batch, labels_batch = [], [], []
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
    # Tokenizer
    # --------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Ensure pad token exists (some chat models don’t set it)
    if tokenizer.pad_token is None:
        # Common choice: use eos as pad for causal LM
        tokenizer.pad_token = tokenizer.eos_token

    # --------
    # Model (QLoRA 4-bit)
    # --------
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     trust_remote_code=True,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    # )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.to("mps")
    model.config.use_cache = False  # important for training

    # --------
    # Dataset
    # --------
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

    def preprocess(example):
        messages = _ensure_messages(example)
        return build_text_and_labels(tokenizer, messages)

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    # --------
    # LoRA config
    # --------
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # --------
    # Training args
    # --------
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,  # we provide exact tensors
        dataloader_num_workers=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer)

    # --------
    # Trainer
    # --------
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=True)

    # Save adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Done. LoRA adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
