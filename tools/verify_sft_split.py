# verify_sft_split.py
# 验证 SFT 训练中 prompt_len 切分是否正确
#
# 核心检查：full_ids[:prompt_len] == prompt_ids 是否成立
# 如果不成立，说明 label 的起点切分不可靠
#
# Run:
#   python verify_sft_split.py

import json
from typing import Dict, List, Any
from modelscope import AutoTokenizer

# 配置
MODEL_NAME = "Qwen/Qwen3-14B"
DATA_FILE = "datasets0211_train/train/train_h_10000.jsonl"
NUM_SAMPLES = 10000  # 检查的样本数


def check_split(
    tokenizer,
    messages: List[Dict[str, str]],
    label: str,
    sample_idx: int,
) -> Dict[str, Any]:
    """
    检查 prompt_len 切分是否正确
    """
    # 1. 构建完整对话
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": label})

    # 2. 获取 full_text 和 full_ids
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # 3. 获取 prompt_text 和 prompt_ids
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    prompt_len = len(prompt_ids)

    # 4. 核心检查：full_ids[:prompt_len] == prompt_ids
    full_prefix = full_ids[:prompt_len]
    match = full_prefix == prompt_ids

    # 5. 找出不匹配的位置
    mismatch_positions = []
    if not match:
        for i in range(min(len(full_prefix), len(prompt_ids))):
            if full_prefix[i] != prompt_ids[i]:
                mismatch_positions.append(i)
        # 长度不同也算不匹配
        if len(full_prefix) != len(prompt_ids):
            mismatch_positions.append(
                f"len_diff: {len(full_prefix)} vs {len(prompt_ids)}"
            )

    # 6. 检查 label 部分
    label_ids = full_ids[prompt_len:]
    label_text_decoded = tokenizer.decode(label_ids, skip_special_tokens=False)

    # 7. 检查 label 是否正确开始
    # label 应该以实际的 label 内容开始，而不是 assistant 标记
    label_starts_correctly = (
        label.strip()[:20] in label_text_decoded[:50] if label_ids else False
    )

    return {
        "sample_idx": sample_idx,
        "prompt_len": prompt_len,
        "full_len": len(full_ids),
        "label_len": len(label_ids),
        "prefix_match": match,
        "mismatch_positions": (
            mismatch_positions[:5] if mismatch_positions else []
        ),  # 只显示前5个
        "label_starts_correctly": label_starts_correctly,
        "label_preview": label,
        "decoded_label_preview": label_text_decoded if label_ids else "(empty)",
        "prompt_text_tail": prompt_text[-200:],
        "full_text_tail": full_text[-200:],
    }


def main():
    print("=" * 70)
    print("🔍 验证 SFT 训练中 prompt_len 切分是否正确")
    print("=" * 70)
    print(f"   Model: {MODEL_NAME}")
    print(f"   Data: {DATA_FILE}")
    print(f"   Samples: {NUM_SAMPLES}")
    print("=" * 70)

    print(f"\n🔹 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"🔹 读取数据...")
    with open(DATA_FILE, "r") as f:
        samples = [json.loads(line) for line in f.readlines()[:NUM_SAMPLES]]

    match_count = 0
    mismatch_count = 0
    label_correct_count = 0

    results = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        label = sample["label"]

        result = check_split(tokenizer, messages, label, i)
        results.append(result)

        if result["prefix_match"]:
            match_count += 1
        else:
            mismatch_count += 1

        if result["label_starts_correctly"]:
            label_correct_count += 1

    # 打印详细结果
    print(f"\n{'='*70}")
    print("📊 详细结果:")
    print("=" * 70)

    for r in results:
        status = "✅" if r["prefix_match"] else "❌"
        label_status = "✅" if r["label_starts_correctly"] else "⚠️"

        print(f"\n样本 {r['sample_idx']+1}:")
        print(f"   {status} prefix_match: {r['prefix_match']}")
        print(
            f"   {label_status} label_starts_correctly: {r['label_starts_correctly']}"
        )
        print(
            f"   prompt_len: {r['prompt_len']}, full_len: {r['full_len']}, label_len: {r['label_len']}"
        )

        if not r["prefix_match"]:
            print(f"   ❌ 不匹配位置: {r['mismatch_positions']}")

        print(f"   label 预览: {r['label_preview']}")
        print(f"   decoded label: {r['decoded_label_preview']}")

        # 如果不匹配，显示更多细节
        if not r["prefix_match"] or not r["label_starts_correctly"]:
            print(f"\n   📜 prompt_text 结尾:")
            print(f"      {repr(r['prompt_text_tail'])}")
            print(f"\n   📜 full_text 结尾:")
            print(f"      {repr(r['full_text_tail'])}")

    # 打印汇总
    print(f"\n{'='*70}")
    print("📈 汇总统计:")
    print("=" * 70)
    print(f"   总样本数: {NUM_SAMPLES}")
    print(
        f"   ✅ prefix 完全匹配: {match_count}/{NUM_SAMPLES} ({match_count/NUM_SAMPLES*100:.1f}%)"
    )
    print(
        f"   ❌ prefix 不匹配: {mismatch_count}/{NUM_SAMPLES} ({mismatch_count/NUM_SAMPLES*100:.1f}%)"
    )
    print(
        f"   ✅ label 起始正确: {label_correct_count}/{NUM_SAMPLES} ({label_correct_count/NUM_SAMPLES*100:.1f}%)"
    )

    print(f"\n{'='*70}")
    if mismatch_count == 0:
        print("✅ 结论: prompt_len 切分可靠，所有样本 prefix 完全匹配")
    else:
        print("❌ 结论: prompt_len 切分不可靠！")
        print("   建议: 需要调整切分策略，或使用其他方法定位 label 起点")

    # 额外检查：直接比较 token 序列
    print(f"\n{'='*70}")
    print("🔬 深入分析第一个样本的 token 边界:")
    print("=" * 70)

    if samples:
        sample = samples[0]
        messages = sample["messages"]
        label = sample["label"]

        full_messages = messages.copy()
        full_messages.append({"role": "assistant", "content": label})

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        prompt_len = len(prompt_ids)

        print(f"\n   prompt_ids 最后 10 个 token:")
        for i, tid in enumerate(prompt_ids[-10:]):
            text = tokenizer.decode([tid])
            print(f"      [{prompt_len-10+i}] {tid}: {repr(text)}")

        print(f"\n   full_ids 在 prompt_len 附近的 token:")
        start = max(0, prompt_len - 5)
        end = min(len(full_ids), prompt_len + 10)
        for i in range(start, end):
            tid = full_ids[i]
            text = tokenizer.decode([tid])
            marker = " <-- prompt_len 切分点" if i == prompt_len else ""
            marker2 = " (label 开始)" if i == prompt_len else ""
            print(f"      [{i}] {tid}: {repr(text)}{marker}{marker2}")


if __name__ == "__main__":
    main()
