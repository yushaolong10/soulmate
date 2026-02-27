# dpo_gpu_8bit.py
# LoRA DPO for Qwen (chat style) with 8-bit quantization
# 使用 DPO (Direct Preference Optimization) 进行偏好对齐训练
#
# 架构:
#   - Policy: Base (8-bit) + SFT adapter (冻结) + DPO adapter (训练)
#   - Reference: 使用 precompute_ref_log_probs 预计算，无需额外加载 ref model
#
# 优点:
#   - 使用 8-bit 量化，显存占用更少 (~15GB)
#   - 与 sft_gpu_8bit.py 训练的模型兼容
#   - 数值稳定，避免 bfloat16 的溢出问题
#   - 使用 precompute_ref_log_probs=True 预计算 ref logprobs，
#     避免 8-bit 模型跨设备训练的限制
#
# Requirements:
#   pip install -U "transformers>=4.41" datasets accelerate peft torch trl bitsandbytes
#
# Data format (JSONL):
# {
#   "prompt": [...messages...],
#   "chosen": "正确回复",
#   "rejected": "负样本回复",
#   "rejected_type": "负样本类型"
# }
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 python dpo_gpu_8bit.py
#
# Output:
#   ./qwen_lora_dpo_0226  (LoRA adapter weights + tokenizer)

import os
import math
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TrainerCallback
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# -----------------------------
# User config
# -----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-14B")
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "qwen_lora_adapter_0226_1w_8bit")  # SFT LoRA 模型路径
TRAIN_FILE = os.environ.get("TRAIN_FILE", "datasets0211_train/dpo/dpo_1700.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen_lora_dpo_0226_1700_8bit")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-5"))

PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "1"))

# DPO hyperparameters
BETA = float(os.environ.get("BETA", "0.05"))  # DPO 温度参数

# DPO LoRA hyperparameters (新增的 DPO adapter)
DPO_LORA_R = int(os.environ.get("DPO_LORA_R", "8"))
DPO_LORA_ALPHA = int(os.environ.get("DPO_LORA_ALPHA", "16"))
DPO_LORA_DROPOUT = float(os.environ.get("DPO_LORA_DROPOUT", "0.05"))


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


def load_model_8bit(model_name: str, device_map: str = "auto"):
    """
    使用 8-bit 量化加载模型
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    return model


def main():
    # --------
    # Device check
    # --------
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow on CPU!")
        device_map = "auto"
    else:
        num_gpus = torch.cuda.device_count()
        print(f"✅ CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )
        # 8-bit 量化模型只能使用单一设备，使用 precompute_ref_log_probs 避免多卡限制
        device_map = "auto"
        print(f"\n📍 使用 precompute_ref_log_probs=True 方案")
        print(f"   Policy model device: {device_map}")
        print(f"   Reference logprobs: 预计算后缓存，无需额外 ref model")

    # --------
    # Tokenizer
    # --------
    print(f"🔹 Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO 训练时需要 left padding
    tokenizer.padding_side = "left"

    # --------
    # Policy Model: Base (8-bit) + SFT LoRA (冻结) + DPO LoRA (训练)
    # --------
    print(f"\n🔹 Loading base model (8-bit) from {BASE_MODEL}...")
    print(f"   Device map: {device_map}")
    base_model = load_model_8bit(BASE_MODEL, device_map=device_map)
    base_model.config.use_cache = False

    # 为量化训练准备模型
    base_model = prepare_model_for_kbit_training(base_model)

    # 验证 base model
    print(f"🔹 Verifying base model...")
    with torch.no_grad():
        test_input = tokenizer("你好", return_tensors="pt").to(base_model.device)
        test_output = base_model(**test_input)
        base_logits = test_output.logits
        print(
            f"   Base model logits: [{base_logits.min().item():.2f}, {base_logits.max().item():.2f}]"
        )
        if torch.isnan(base_logits).any() or torch.isinf(base_logits).any():
            raise ValueError("❌ Base model produces NaN/Inf!")
        print(f"   ✅ Base model OK")

    # 加载 SFT LoRA adapter (冻结)
    print(f"🔹 Loading SFT LoRA adapter from {SFT_LORA_DIR} (frozen)...")
    if not os.path.exists(SFT_LORA_DIR):
        raise FileNotFoundError(f"SFT LoRA directory not found: {SFT_LORA_DIR}")

    model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR, is_trainable=False)

    # 验证 SFT model
    print(f"🔹 Verifying SFT model...")
    with torch.no_grad():
        test_output = model(**test_input)
        sft_logits = test_output.logits
        print(
            f"   SFT model logits: [{sft_logits.min().item():.2f}, {sft_logits.max().item():.2f}]"
        )
        if torch.isnan(sft_logits).any() or torch.isinf(sft_logits).any():
            raise ValueError("❌ SFT model produces NaN/Inf! Please retrain SFT model.")
        print(f"   ✅ SFT LoRA loaded (frozen)")

    # 在 SFT LoRA 基础上添加新的 DPO LoRA adapter (训练)
    print(f"🔹 Adding DPO LoRA adapter (trainable)...")
    print(
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
    print(f"   ✅ DPO LoRA adapter added")

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    # --------
    # Reference Model: 使用 precompute_ref_log_probs，无需加载额外模型
    # --------
    # 注意：使用 precompute_ref_log_probs=True 时，DPOTrainer 会：
    # 1. 在训练前，使用当前模型（关闭 DPO adapter）计算 ref logprobs
    # 2. 将 ref logprobs 缓存到 dataset 中
    # 3. 训练时只使用 policy model，从缓存读取 ref logprobs
    # 这样就不需要额外加载 ref model，避免了 8-bit 模型跨设备的问题
    print(f"\n🔹 Reference logprobs: 将使用 precompute_ref_log_probs 预计算")
    print(f"   无需加载额外的 reference model")
    print(f"   预计算时会临时禁用 DPO adapter，使用 SFT 状态计算 ref logprobs")

    # --------
    # Dataset
    # --------
    print(f"🔹 Loading dataset from {TRAIN_FILE}...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
    print(f"   Total samples: {len(ds)}")

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
    if filtered_len < original_len:
        print(
            f"   Filtered {original_len - filtered_len} samples with response < 5 chars"
        )
    print(f"   Total samples (after filtering): {filtered_len}")

    # 打印样本示例
    print("\n📝 Sample example:")
    sample = ds[0]
    print(f"   Prompt (truncated): {sample['prompt'][:200]}...")
    print(f"   Chosen: {sample['chosen'][:100]}...")
    print(f"   Rejected: {sample['rejected'][:100]}...")

    # --------
    # Trainable parameters info
    # --------
    print("\n🔹 Trainable parameters (只训练 DPO adapter):")
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
        # 这样就不需要额外的 ref_model，避免 8-bit 模型跨设备限制
        precompute_ref_log_probs=True,
        # padding 设置
        label_pad_token_id=-100,
        # 精度设置 - 使用 fp16 与 8-bit 量化配合
        bf16=False,
        fp16=True,
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
    print("🔹 Initializing DPOTrainer...")
    print(f"   Base model: {BASE_MODEL} (8-bit quantized)")
    print(f"   SFT LoRA (frozen): {SFT_LORA_DIR}")
    print(f"   DPO LoRA (trainable): r={DPO_LORA_R}, alpha={DPO_LORA_ALPHA}")
    print(f"   Beta (temperature): {BETA}")
    print(f"   Max sequence length: {MAX_SEQ_LEN}")
    print(f"   Max prompt length: {MAX_PROMPT_LEN}")
    print(f"   precompute_ref_log_probs: True (无需额外 ref model)")

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
    print("🚀 Starting DPO training...")
    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --------
    # 训练后验证
    # --------
    print("\n🔹 Verifying model after training...")
    model.eval()
    with torch.no_grad():
        test_input = tokenizer("你好", return_tensors="pt").to(model.device)
        test_output = model(**test_input)
        final_logits = test_output.logits
        print(
            f"   Final logits: [{final_logits.min().item():.2f}, {final_logits.max().item():.2f}]"
        )
        if torch.isnan(final_logits).any() or torch.isinf(final_logits).any():
            print(f"   ❌ 训练后模型产生 NaN/Inf!")
        else:
            print(f"   ✅ 无 NaN/Inf")

    print(f"\n✅ Done. DPO LoRA adapter saved to: {OUTPUT_DIR}")

    # --------
    # 测试文本生成
    # --------
    print(f"\n🔹 Testing text generation...")
    try:
        messages = [
            {"role": "system", "content": "你是一个友好的男生"},
            {"role": "user", "content": "我想你了"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"   Input: 我想你了")
        print(f"   Output: {response}")

        if len(response.strip()) > 0:
            print(f"   ✅ 文本生成正常")
        else:
            print(f"   ⚠️ 生成结果为空，可能有问题")
    except Exception as e:
        print(f"   ❌ 文本生成失败: {e}")

    print(f"\n✅ All done!\n")


if __name__ == "__main__":
    main()
