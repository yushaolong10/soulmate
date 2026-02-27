#!/usr/bin/env python3
"""
调试脚本：对比新旧方法中 label 参与 loss 的情况
运行: python debug_labels.py
"""
import json
from modelscope import AutoTokenizer

# 可以换成你实际使用的模型
MODEL_NAME = "Qwen/Qwen3-14B"
DATA_FILE = "datasets0211_train/train/train_h_10000.jsonl"


def old_method(tokenizer, messages, label):
    """旧方法：不带 enable_thinking=False（可能默认启用 thinking）"""
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": label})

    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # 调整长度
    if len(labels) < len(full_ids):
        labels = labels + [-100] * (len(full_ids) - len(labels))
    elif len(labels) > len(full_ids):
        labels = labels[: len(full_ids)]

    valid_count = sum(1 for l in labels if l != -100)
    return {
        "full_ids_len": len(full_ids),
        "prompt_ids_len": len(prompt_ids),
        "valid_labels": valid_count,
        "prompt_text_tail": prompt_text[-300:],
        "full_text_tail": full_text[-300:],
    }


def new_method(tokenizer, messages, label):
    """新方法：使用 enable_thinking=False，与推理保持一致"""
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": label})

    # 完整文本（与推理一致：enable_thinking=False）
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # prompt 部分
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # 调整长度
    if len(labels) < len(full_ids):
        labels = labels + [-100] * (len(full_ids) - len(labels))
    elif len(labels) > len(full_ids):
        labels = labels[: len(full_ids)]

    valid_count = sum(1 for l in labels if l != -100)
    return {
        "full_ids_len": len(full_ids),
        "prompt_ids_len": len(prompt_ids),
        "valid_labels": valid_count,
        "prompt_text_tail": prompt_text[-300:],
        "full_text_tail": full_text[-300:],
    }


def new_method_manual(tokenizer, messages, label):
    """备选方法：手动拼接（如果 enable_thinking 不可用）"""
    messages_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    assistant_text = f"<|im_start|>assistant\n{label}<|im_end|>"
    full_text = messages_text + "\n" + assistant_text

    messages_ids = tokenizer(messages_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    prompt_len = len(messages_ids)
    labels = [-100] * len(full_ids)

    for i in range(prompt_len, len(full_ids)):
        labels[i] = full_ids[i]

    valid_count = sum(1 for l in labels if l != -100)

    # 备选方案验证
    if valid_count == 0:
        label_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
        label_len = len(label_ids)
        for i in range(max(0, len(full_ids) - label_len), len(full_ids)):
            labels[i] = full_ids[i]
        valid_count = sum(1 for l in labels if l != -100)

    return {
        "full_ids_len": len(full_ids),
        "messages_ids_len": len(messages_ids),
        "valid_labels": valid_count,
        "full_text_tail": full_text[-300:],
    }


def main():
    print("=" * 70)
    print("🔍 对比新旧方法：检查 label 是否参与 loss 计算")
    print("=" * 70)
    print("   🔴 旧方法: 不带 enable_thinking=False（可能有 <think> 标签）")
    print("   🟢 新方法: 带 enable_thinking=False（与推理一致）")
    print("=" * 70)

    print(f"\n🔹 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"🔹 读取数据: {DATA_FILE}")
    with open(DATA_FILE, "r") as f:
        samples = [json.loads(line) for line in f.readlines()[:5]]

    old_total_valid = 0
    new_total_valid = 0
    old_total_tokens = 0
    new_total_tokens = 0

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        label = sample["label"]

        old_result = old_method(tokenizer, messages, label)
        new_result = new_method(tokenizer, messages, label)

        print(f"\n{'='*70}")
        print(f"📝 样本 {i+1}: label = {label[:50]}...")

        print(f"\n   🔴 旧方法 (无 enable_thinking):")
        print(
            f"      full_ids: {old_result['full_ids_len']}, prompt_ids: {old_result['prompt_ids_len']}"
        )
        print(
            f"      差值: {old_result['full_ids_len'] - old_result['prompt_ids_len']}"
        )
        print(f"      有效 labels: {old_result['valid_labels']}")
        if old_result["valid_labels"] == 0:
            print(f"      ❌ 问题: 所有 labels 都是 -100！")
        elif old_result["valid_labels"] < 10:
            print(f"      ⚠️ 警告: 有效 labels 过少！")

        print(f"\n   🟢 新方法 (enable_thinking=False):")
        print(
            f"      full_ids: {new_result['full_ids_len']}, prompt_ids: {new_result['prompt_ids_len']}"
        )
        print(
            f"      差值: {new_result['full_ids_len'] - new_result['prompt_ids_len']}"
        )
        print(f"      有效 labels: {new_result['valid_labels']}")
        if new_result["valid_labels"] > 0:
            print(f"      ✅ 正常")

        old_total_valid += old_result["valid_labels"]
        new_total_valid += new_result["valid_labels"]
        old_total_tokens += old_result["full_ids_len"]
        new_total_tokens += new_result["full_ids_len"]

        # 显示 prompt 结尾对比（当有问题时）
        if old_result["valid_labels"] <= 0 or old_result["valid_labels"] != new_result["valid_labels"]:
            print(f"\n   📜 旧方法 prompt 结尾 (检查是否有 <think>):")
            print(f"      {repr(old_result['prompt_text_tail'][-250:])}")
            print(f"\n   📜 新方法 prompt 结尾:")
            print(f"      {repr(new_result['prompt_text_tail'][-250:])}")

    print(f"\n{'='*70}")
    print(f"📈 总结 (前 {len(samples)} 条样本):")
    print(
        f"   🔴 旧方法: {old_total_valid}/{old_total_tokens} 有效 labels ({old_total_valid/old_total_tokens*100:.1f}%)"
    )
    print(
        f"   🟢 新方法: {new_total_valid}/{new_total_tokens} 有效 labels ({new_total_valid/new_total_tokens*100:.1f}%)"
    )

    if old_total_valid == 0 and new_total_valid > 0:
        print(f"\n✅ 确认问题: 旧方法 label 完全被 mask，新方法已修复！")
    elif old_total_valid < new_total_valid:
        print(f"\n⚠️ 旧方法有效 label 比新方法少，问题已确认！")
        print(f"   差异: {new_total_valid - old_total_valid} 个 tokens")
    elif old_total_valid / old_total_tokens < 0.05:
        print(f"\n⚠️ 旧方法有效 label 比例过低，建议检查数据！")
    else:
        print(f"\n✅ 两种方法结果一致，template 应该没问题")


if __name__ == "__main__":
    main()
