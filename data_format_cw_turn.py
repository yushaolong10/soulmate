#!/usr/bin/env python3
"""
多 assistant turn SFT 数据格式化脚本。

参考 `data_format_cw.py` 的清洗与窗口逻辑，但输出格式改为：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "meta": {
    "lang": "zh",
    "turn_count": 8,
    "assistant_turns": 8,
    "loss_mode": "assistant_turns"
  }
}

训练时不再单独提供 `label`，而是对样本中的多个 assistant turn 直接监督。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_format_cw import (
    DST_DIR as BASE_DST_DIR,
    LANGUAGES as BASE_LANGUAGES,
    MAX_TURNS_PER_FILE as BASE_MAX_TURNS_PER_FILE,
    WINDOW_SIZE as BASE_WINDOW_SIZE,
    FilterStats,
    DialogTurn,
    parse_file,
    resolve_system_prompt,
    window_has_duplicate_label,
    window_has_loop,
    window_has_no_punct_streak,
)


SRC_DIR = os.environ.get("SRC_DIR", "datasets0305_clean")
DST_DIR = os.environ.get("DST_DIR", BASE_DST_DIR)
LANGUAGES = os.environ.get("LANGUAGES", ",".join(BASE_LANGUAGES)).split(",")
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", str(BASE_WINDOW_SIZE)))
MAX_TURNS_PER_FILE = int(
    os.environ.get("MAX_TURNS_PER_FILE", str(BASE_MAX_TURNS_PER_FILE))
)
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_turn")

# 采样策略：
# - 短段（<= SHORT_SEGMENT_MAX_TURNS）直接整段保留
# - 长段使用变长窗口采样，减少固定 8 轮 + stride=1 的高重复问题
SHORT_SEGMENT_MAX_TURNS = int(os.environ.get("SHORT_SEGMENT_MAX_TURNS", "6"))
MIXED_WINDOW_SIZES = [
    int(x)
    for x in os.environ.get("MIXED_WINDOW_SIZES", "8,12,16").split(",")
    if x.strip()
]
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "4"))
ALWAYS_INCLUDE_TAIL = os.environ.get("ALWAYS_INCLUDE_TAIL", "1") == "1"


def create_training_sample(window: List[DialogTurn], lang: str) -> Dict[str, Any]:
    """将完整多轮窗口转成 messages-only 训练样本。"""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": resolve_system_prompt(window, lang)}
    ]
    for turn in window:
        messages.append({"role": "user", "content": turn.user_content})
        messages.append({"role": "assistant", "content": turn.assistant_content})

    return {
        "messages": messages,
    }


def clamp_segments(segments: List[List[DialogTurn]]) -> List[List[DialogTurn]]:
    """按段顺序累计截断到 MAX_TURNS_PER_FILE 轮。"""
    clamped_segments: List[List[DialogTurn]] = []
    remaining = MAX_TURNS_PER_FILE
    for seg in segments:
        if remaining <= 0:
            break
        if len(seg) >= remaining:
            clamped_segments.append(seg[:remaining])
            remaining = 0
        else:
            clamped_segments.append(seg)
            remaining -= len(seg)
    return clamped_segments


def unique_window_ranges(seg_len: int) -> List[Tuple[int, int]]:
    """
    为单个连续段生成窗口范围：
    - 短段：整段保留
    - 长段：按变长窗口 + stride 采样
    - 尾段：可选强制补一个对齐段尾的窗口
    """
    if seg_len <= 0:
        return []

    if seg_len <= SHORT_SEGMENT_MAX_TURNS:
        return [(0, seg_len)]

    ranges: List[Tuple[int, int]] = []
    seen = set()

    for window_size in MIXED_WINDOW_SIZES:
        if seg_len < window_size:
            continue

        last_start = seg_len - window_size
        for start in range(0, last_start + 1, max(1, WINDOW_STRIDE)):
            item = (start, start + window_size)
            if item not in seen:
                ranges.append(item)
                seen.add(item)

        if ALWAYS_INCLUDE_TAIL:
            tail_item = (last_start, seg_len)
            if tail_item not in seen:
                ranges.append(tail_item)
                seen.add(tail_item)

    if not ranges:
        return [(0, seg_len)]
    return ranges


def validate_window(window: List[DialogTurn], stats: FilterStats) -> bool:
    """复用原有窗口级过滤逻辑。"""
    stats.candidate_windows += 1

    if window_has_loop(window):
        stats.drop_loop_window += 1
        return False

    if window_has_no_punct_streak(window):
        stats.drop_no_punct_window += 1
        return False

    if len(window) >= WINDOW_SIZE and window_has_duplicate_label(window):
        stats.drop_dup_label_window += 1
        return False

    return True


def process_file(
    file_path: Path, lang: str, stats: FilterStats
) -> Tuple[List[Dict[str, Any]], int]:
    """
    处理单个文件：
    解析（按连续段）→ 截断 → 各段独立滑动窗口 → 窗口级过滤 → 完整多轮样本。
    """
    segments = parse_file(file_path, stats)
    samples: List[Dict[str, Any]] = []

    original_count = sum(len(s) for s in segments)
    if original_count < WINDOW_SIZE:
        print(
            f"  ⚠️  {file_path.name}: 有效对话仅 {original_count} 轮（< {WINDOW_SIZE}），跳过"
        )
        return samples, 0

    clamped_segments = clamp_segments(segments)
    total_used = sum(len(s) for s in clamped_segments)
    total_supervised_turns = 0
    total_windows = 0

    for seg in clamped_segments:
        for start, end in unique_window_ranges(len(seg)):
            window = seg[start:end]
            total_windows += 1
            if not validate_window(window, stats):
                continue

            samples.append(create_training_sample(window, lang))
            stats.valid_samples += 1
            total_supervised_turns += len(window)

    clamp_note = (
        f" ✂截断自{original_count}" if original_count > MAX_TURNS_PER_FILE else ""
    )
    seg_note = f" {len(clamped_segments)}段" if len(clamped_segments) > 1 else ""
    print(
        f"  ✓  {file_path.name}: "
        f"{total_used}轮{clamp_note}{seg_note} → {len(samples)}样本 / "
        f"{total_supervised_turns}个assistant轮 / {total_windows}个候选窗口"
    )
    return samples, total_supervised_turns


def process_directory(
    src_dir: Path, lang: str
) -> Tuple[List[Dict[str, Any]], FilterStats, int]:
    """处理目录下所有 jsonl 文件。"""
    stats = FilterStats()
    all_samples: List[Dict[str, Any]] = []
    total_supervised_turns = 0

    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    print(f"\n📂 处理 {lang.upper()} 目录: {src_dir}")
    print(f"   找到 {len(jsonl_files)} 个 JSONL 文件")

    for file_path in jsonl_files:
        samples, supervised_turns = process_file(file_path, lang, stats)
        all_samples.extend(samples)
        total_supervised_turns += supervised_turns

    return all_samples, stats, total_supervised_turns


def save_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)
    dst_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("📝 数据格式化脚本 turn 版 — 多 assistant turn SFT")
    print("=" * 64)
    print(f"数据源目录:         {src_dir}/")
    print(f"处理语言:           {LANGUAGES}")
    print(f"短段整段阈值:       <= {SHORT_SEGMENT_MAX_TURNS} 轮")
    print(f"变长窗口集合:       {MIXED_WINDOW_SIZES}")
    print(f"窗口步长:           {WINDOW_STRIDE}")
    print(f"保留段尾窗口:       {ALWAYS_INCLUDE_TAIL}")
    print(f"每文件最大轮数:     {MAX_TURNS_PER_FILE} 轮")
    print(f"输出后缀:           {OUTPUT_SUFFIX}")
    print("样本格式:           连续段 + 变长窗口 messages（assistant turn 参与监督）")
    print("=" * 64)

    lang_dirs = {
        "zh": src_dir / "zh",
        "tw": src_dir / "tw",
    }

    total_samples: List[Dict[str, Any]] = []
    total_supervised_turns = 0
    summary: Dict[str, Any] = {}

    for lang in LANGUAGES:
        d = lang_dirs.get(lang)
        if not d or not d.exists():
            print(f"\n⚠️  语言 [{lang}] 目录不存在: {d}，跳过")
            continue

        samples, stats, supervised_turns = process_directory(d, lang)

        if samples:
            out = dst_dir / f"train_{lang}{OUTPUT_SUFFIX}.jsonl"
            save_jsonl(samples, out)
            print(
                f"\n  💾 保存: {out}  ({len(samples)} 个样本, {supervised_turns} 个assistant轮)"
            )

        total_samples.extend(samples)
        total_supervised_turns += supervised_turns
        summary[lang] = (len(samples), stats, supervised_turns)

    if total_samples:
        out_all = dst_dir / f"train_all{OUTPUT_SUFFIX}.jsonl"
        save_jsonl(total_samples, out_all)
        print(
            f"\n  💾 合并保存: {out_all}  "
            f"({len(total_samples)} 个样本, {total_supervised_turns} 个assistant轮)"
        )

    print("\n" + "=" * 64)
    print("📊 过滤统计报告")
    print("=" * 64)
    grand_total = 0
    for lang, (n, stats, supervised_turns) in summary.items():
        print(f"\n[{lang.upper()}]")
        print(stats.summary())
        print(f"  监督 assistant 轮数:  {supervised_turns}")
        grand_total += n

    print("\n" + "─" * 64)
    print(f"✅ 完成！最终有效样本总计: {grand_total} 条")
    print(f"✅ 总监督 assistant 轮数: {total_supervised_turns}")
    print("─" * 64)
    print("=" * 64)


if __name__ == "__main__":
    main()
