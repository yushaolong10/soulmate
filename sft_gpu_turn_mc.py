# sft_gpu_mc_turn.py
# LoRA SFT for Qwen (chat style) - 多 assistant turn 监督
#
# 数据格式 (JSONL):
# {
#   "messages": [
#     {"role":"system","content":"..."},
#     {"role":"user","content":"..."},
#     {"role":"assistant","content":"..."},
#     ...
#   ],
#   "meta": {...}
# }
#
# 相比 `sft_gpu_mc.py`，这里不再只训练最后一个 `label`，
# 而是直接对样本中的多个 assistant turn 打 loss。

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from datasets import load_dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments


def is_main_process() -> bool:
    """判断是否为主进程 (rank 0)。"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """只在主进程打印。"""
    if is_main_process():
        print(*args, **kwargs)


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-14B")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets0305_train/train/train_turn_bh_48k.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_adapter_0311_48k_mc")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "4e-5"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "20"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "10"))

# 0 表示监督所有 assistant turn；正整数表示只监督最后 N 个 assistant turn
SUPERVISE_LAST_N_ASSISTANTS = int(os.environ.get("SUPERVISE_LAST_N_ASSISTANTS", "3"))

LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))


def _encode_messages(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def build_input_and_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    max_seq_len: int,
    supervise_last_n_assistants: int,
) -> Dict[str, Any]:
    """
    构建完整多轮对话的 input_ids 和 labels。

    监督策略：
    - system / user token 全部 mask
    - assistant token 参与 loss
    - 可选只监督最后 N 个 assistant turn
    """
    full_ids = _encode_messages(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full_ids)

    assistant_indices = [
        idx for idx, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("Each example must contain at least one assistant turn.")

    if supervise_last_n_assistants > 0:
        target_indices = assistant_indices[-supervise_last_n_assistants:]
    else:
        target_indices = assistant_indices

    supervised_turn_count = 0
    for assistant_idx in target_indices:
        prompt_ids = _encode_messages(
            tokenizer,
            messages[:assistant_idx],
            add_generation_prompt=True,
        )
        upto_ids = _encode_messages(
            tokenizer,
            messages[: assistant_idx + 1],
            add_generation_prompt=False,
        )

        start = len(prompt_ids)
        end = min(len(upto_ids), len(full_ids))
        if start >= end:
            continue

        labels[start:end] = full_ids[start:end]
        supervised_turn_count += 1

    if len(full_ids) > max_seq_len:
        keep_from = len(full_ids) - max_seq_len
        input_ids = full_ids[keep_from:]
        labels = labels[keep_from:]
    else:
        input_ids = full_ids

    attention_mask = [1] * len(input_ids)
    loss_token_count = sum(1 for token in labels if token != -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_token_count": loss_token_count,
        "supervised_turn_count": supervised_turn_count,
        "total_assistant_turns": len(assistant_indices),
        "seq_len": len(input_ids),
    }


class NaNDetectorCallback(TrainerCallback):
    """训练过程中检测 NaN/Inf 的回调。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", None)
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                print(f"\n❌ 检测到异常 loss: {loss}")
                print(f"   Step: {state.global_step}")
                print("   停止训练...")
                control.should_training_stop = True
        return control


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
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

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids_batch.append(feature["input_ids"] + [pad_id] * pad_len)
            attn_batch.append(feature["attention_mask"] + [0] * pad_len)
            labels_batch.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def main():
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow on CPU!")
    elif is_main_process():
        print(f"✅ CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )
        if num_gpus > 1:
            print(f"\n🚀 多卡训练模式: {num_gpus} GPUs")
            print("   使用 DDP (DistributedDataParallel) 进行分布式训练")

    print_rank0(f"🔹 Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank0(f"🔹 Loading model from {MODEL_NAME}...")
    if num_gpus > 1:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        model = model.to(f"cuda:{local_rank}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print_rank0(f"🔹 Loading dataset from {TRAIN_FILE}...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
    print_rank0(f"   Total samples: {len(ds)}")

    def preprocess(example):
        messages = example.get("messages", [])
        if not messages:
            raise ValueError("Each example must have non-empty `messages`.")
        return build_input_and_labels(
            tokenizer,
            messages,
            MAX_SEQ_LEN,
            SUPERVISE_LAST_N_ASSISTANTS,
        )

    ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: x["loss_token_count"] > 0, num_proc=4)

    if is_main_process():
        total_tokens = sum(int(x["seq_len"]) for x in ds)
        train_tokens = sum(int(x["loss_token_count"]) for x in ds)
        supervised_turns = sum(int(x["supervised_turn_count"]) for x in ds)
        total_assistant_turns = sum(int(x["total_assistant_turns"]) for x in ds)
        print(f"   Total tokens: {total_tokens:,}")
        print(
            f"   Trainable tokens (assistant turns): "
            f"{train_tokens:,} ({train_tokens / total_tokens * 100:.1f}%)"
        )
        print(
            f"   Supervised assistant turns: {supervised_turns:,} / "
            f"{total_assistant_turns:,}"
        )
        if SUPERVISE_LAST_N_ASSISTANTS > 0:
            print(
                f"   Supervision mode: last {SUPERVISE_LAST_N_ASSISTANTS} assistant turns"
            )
        else:
            print("   Supervision mode: all assistant turns")

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

    print_rank0("🔹 Applying LoRA...")
    print_rank0(f"   r={LORA_R}, alpha={LORA_ALPHA}, scaling={LORA_ALPHA / LORA_R:.2f}")
    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()

    if is_main_process():
        print("🔹 Verifying model initialization...")
        with torch.no_grad():
            test_input = tokenizer("你好", return_tensors="pt").to(model.device)
            test_output = model(**test_input)
            test_logits = test_output.logits
            print(
                f"   Initial logits range: [{test_logits.min().item():.2f}, {test_logits.max().item():.2f}]"
            )
            if torch.isnan(test_logits).any() or torch.isinf(test_logits).any():
                raise ValueError(
                    "❌ Model produces NaN/Inf before training! Check LoRA config."
                )
        print("   ✅ Model verification passed")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_drop_last=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForCausalLM(tokenizer)

    print_rank0("🔹 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        callbacks=[NaNDetectorCallback()],
    )

    print_rank0("🚀 Starting training...")
    if num_gpus > 1:
        print_rank0(f"   分布式训练: {num_gpus} GPUs")

    trainer.train()

    if is_main_process():
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\n✅ Done. LoRA adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
