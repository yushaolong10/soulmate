# check_sft_model.py
# 检查 SFT LoRA 模型是否正常
#
# Usage:
#   python check_sft_model.py --sft_dir qwen_lora_adapter_0226_1h --load_mode bf16
#   python check_sft_model.py --sft_dir qwen_lora_adapter_0226_1h --load_mode 8bit
#   python check_sft_model.py --sft_dir qwen_lora_adapter_0226_1h --skip_inference
#

import os
import sys
import json
import argparse
import torch
from transformers import BitsAndBytesConfig


def check_directory(sft_dir: str) -> bool:
    """检查目录是否存在及文件完整性"""
    print(f"\n{'='*60}")
    print(f"📁 检查 SFT LoRA 目录: {sft_dir}")
    print(f"{'='*60}")

    if not os.path.exists(sft_dir):
        print(f"❌ 目录不存在: {sft_dir}")
        print(f"\n💡 可能的目录:")
        # 列出当前目录下的 qwen_lora* 目录
        for item in os.listdir("."):
            if "lora" in item.lower() or "adapter" in item.lower():
                print(f"   - {item}")
        return False

    print(f"✅ 目录存在")

    # 检查必要文件
    files_found = os.listdir(sft_dir)
    print(f"\n📄 目录内容:")
    for f in sorted(files_found):
        size = os.path.getsize(os.path.join(sft_dir, f))
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"   - {f} ({size_str})")

    # 检查 adapter_config.json
    config_path = os.path.join(sft_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"\n❌ 缺少 adapter_config.json")
        return False

    print(f"\n✅ adapter_config.json 存在")

    # 读取并显示配置
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\n📋 LoRA 配置:")
    print(f"   - r: {config.get('r', 'N/A')}")
    print(f"   - lora_alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"   - lora_dropout: {config.get('lora_dropout', 'N/A')}")
    print(f"   - target_modules: {config.get('target_modules', 'N/A')}")
    print(
        f"   - base_model_name_or_path: {config.get('base_model_name_or_path', 'N/A')}"
    )

    # 计算 scaling
    r = config.get("r", 16)
    alpha = config.get("lora_alpha", 32)
    scaling = alpha / r
    print(f"   - scaling (alpha/r): {scaling}")

    if scaling > 2:
        print(f"   ⚠️ scaling 较高 ({scaling})，可能导致数值不稳定")

    # 检查模型权重文件
    if "adapter_model.safetensors" in files_found:
        print(f"\n✅ adapter_model.safetensors 存在")
    elif "adapter_model.bin" in files_found:
        print(f"\n✅ adapter_model.bin 存在")
    else:
        print(f"\n❌ 缺少模型权重文件 (adapter_model.safetensors 或 adapter_model.bin)")
        return False

    return True


def check_weights(sft_dir: str) -> bool:
    """检查权重文件数值是否正常"""
    print(f"\n{'='*60}")
    print(f"🔍 检查 LoRA 权重数值")
    print(f"{'='*60}")

    # 加载权重
    weights_path = os.path.join(sft_dir, "adapter_model.safetensors")
    if os.path.exists(weights_path):
        from safetensors.torch import load_file

        weights = load_file(weights_path)
    else:
        weights_path = os.path.join(sft_dir, "adapter_model.bin")
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location="cpu")
        else:
            print("❌ 无法加载权重文件")
            return False

    print(f"   加载了 {len(weights)} 个参数张量")

    # 检查每个权重
    has_nan = False
    has_inf = False
    extreme_values = []

    for name, tensor in weights.items():
        tensor = tensor.float()  # 转换为 float32 进行检查

        if torch.isnan(tensor).any():
            print(f"   ❌ NaN 发现于: {name}")
            has_nan = True

        if torch.isinf(tensor).any():
            print(f"   ❌ Inf 发现于: {name}")
            has_inf = True

        # 检查极端值
        max_val = tensor.abs().max().item()
        if max_val > 100:
            extreme_values.append((name, max_val))

    if has_nan:
        print(f"\n❌ 权重中包含 NaN 值!")
        return False

    if has_inf:
        print(f"\n❌ 权重中包含 Inf 值!")
        return False

    print(f"\n✅ 权重数值检查通过 (无 NaN/Inf)")

    if extreme_values:
        print(f"\n⚠️ 发现 {len(extreme_values)} 个极端值 (>100):")
        for name, val in extreme_values[:5]:  # 只显示前 5 个
            print(f"   - {name}: max={val:.2f}")
        if len(extreme_values) > 5:
            print(f"   ... 还有 {len(extreme_values) - 5} 个")

    # 统计权重分布
    all_values = []
    for name, tensor in weights.items():
        all_values.append(tensor.float().flatten())
    all_values = torch.cat(all_values)

    print(f"\n📊 权重统计:")
    print(f"   - 总参数数: {len(all_values):,}")
    print(f"   - 均值: {all_values.mean().item():.6f}")
    print(f"   - 标准差: {all_values.std().item():.6f}")
    print(f"   - 最小值: {all_values.min().item():.6f}")
    print(f"   - 最大值: {all_values.max().item():.6f}")

    return True


def check_model_inference(
    sft_dir: str, base_model: str = "Qwen/Qwen3-14B", load_mode: str = "8bit"
) -> bool:
    """尝试加载模型并进行推理测试"""
    print(f"\n{'='*60}")
    print(f"🧪 模型推理测试")
    print(f"{'='*60}")

    try:
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print(f"   请安装: pip install modelscope peft transformers torch")
        return False

    print(f"   Base model: {base_model}")
    print(f"   SFT LoRA: {sft_dir}")
    print(f"   Load mode: {load_mode}")

    # 加载 tokenizer
    print(f"\n🔹 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer 加载完成")

    # 加载 base model
    print(f"\n🔹 加载 base model ({load_mode})...")
    try:
        if load_mode == "8bit":
            # 8-bit 量化加载（推荐用于 8-bit 训练的模型）
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="cuda:0",
            )
        elif load_mode == "4bit":
            # 4-bit 量化加载
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="cuda:0",
            )
        else:
            # bf16 + device_map="auto"（可能导致 CPU offload）
            print(f"   ⚠️ bf16 模式可能导致 CPU offload 和设备不一致问题")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        print(f"   ✅ Base model 加载完成")
    except Exception as e:
        print(f"   ❌ Base model 加载失败: {e}")
        return False

    # 测试 base model
    print(f"\n🔹 测试 base model 推理...")
    try:
        with torch.no_grad():
            test_input = tokenizer("你好", return_tensors="pt").to(model.device)
            test_output = model(**test_input)
            base_logits = test_output.logits
            print(
                f"   Base model logits: [{base_logits.min().item():.4f}, {base_logits.max().item():.4f}]"
            )

            if torch.isnan(base_logits).any() or torch.isinf(base_logits).any():
                print(f"   ❌ Base model 产生 NaN/Inf!")
                return False
            print(f"   ✅ Base model 推理正常")
    except Exception as e:
        print(f"   ❌ Base model 推理失败: {e}")
        return False

    # 加载 SFT LoRA
    print(f"\n🔹 加载 SFT LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(model, sft_dir)
        print(f"   ✅ SFT LoRA 加载完成")
    except Exception as e:
        print(f"   ❌ SFT LoRA 加载失败: {e}")
        return False

    # 测试 SFT model
    print(f"\n🔹 测试 SFT model 推理...")
    try:
        with torch.no_grad():
            # 重新构建输入（因为 model 可能已经改变）
            test_input = tokenizer("你好", return_tensors="pt").to(model.device)
            test_output = model(**test_input)
            sft_logits = test_output.logits
            print(
                f"   SFT model logits: [{sft_logits.min().item():.4f}, {sft_logits.max().item():.4f}]"
            )

            if torch.isnan(sft_logits).any() or torch.isinf(sft_logits).any():
                print(f"   ❌ SFT model 产生 NaN/Inf!")
                print(f"\n💡 这说明 SFT LoRA 权重有问题，需要重新训练 SFT")
                return False

            # 检查极小值
            if sft_logits.abs().max().item() < 1e-10:
                print(f"   ⚠️ SFT model logits 值过小 (可能接近 0)")
                print(f"   这可能导致数值不稳定")

            print(f"   ✅ SFT model 推理正常")
    except Exception as e:
        print(f"   ❌ SFT model 推理失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 测试生成
    print(f"\n🔹 测试文本生成...")
    try:
        messages = [
            {"role": "system", "content": "你是一个友好的助手"},
            {"role": "user", "content": "你好"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"   生成结果: {response[:100]}...")
        print(f"   ✅ 文本生成正常")
    except Exception as e:
        print(f"   ❌ 文本生成失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="检查 SFT LoRA 模型")
    parser.add_argument(
        "--sft_dir",
        type=str,
        default="qwen_lora_adapter_0211_1w",
        help="SFT LoRA 目录路径",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3-14B", help="Base model 名称"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="跳过模型推理测试（节省时间和显存）",
    )
    parser.add_argument(
        "--load_mode",
        type=str,
        default="bf16",
        choices=["8bit", "4bit", "bf16"],
        help="模型加载模式: bf16 (默认), 4bit, bf16",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 SFT LoRA 模型检查工具")
    print("=" * 60)

    # 1. 检查目录
    if not check_directory(args.sft_dir):
        print(f"\n❌ 目录检查失败")
        sys.exit(1)

    # 2. 检查权重
    if not check_weights(args.sft_dir):
        print(f"\n❌ 权重检查失败")
        sys.exit(1)

    # 3. 推理测试
    if not args.skip_inference:
        if not check_model_inference(args.sft_dir, args.base_model, args.load_mode):
            print(f"\n❌ 推理测试失败")
            sys.exit(1)
    else:
        print(f"\n⏭️ 跳过推理测试")

    # 总结
    print(f"\n{'='*60}")
    print(f"✅ 所有检查通过!")
    print(f"{'='*60}")
    print(f"\n💡 SFT LoRA 模型看起来正常，可以用于 DPO 训练")
    print(f"   运行: SFT_LORA_DIR={args.sft_dir} python dpo_gpu.py")


if __name__ == "__main__":
    main()
