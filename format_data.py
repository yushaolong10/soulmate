#!/usr/bin/env python3
"""
数据格式化脚本
将清洗后的对话数据转换为训练集格式

输入: datasets0211_clean/zh/ 和 datasets0211_clean/tw/
输出: datasets0211_train/train_zh.jsonl 和 datasets0211_train/train_tw.jsonl

格式化规则:
1. 以每个文件为基准进行训练集合构造
2. system_prompt 根据目录 zh/tw 改为简体中文或繁体中文
3. system_prompt 中 ts 使用对话中的时间替换
4. messages 中包含 6 轮对话 (user/assistant)
5. label 是第 6 轮 AI 的回复
6. 使用滑动窗口构造数据
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 配置
SRC_DIR = "datasets0211_clean"
DST_DIR = "datasets0211_train"
WINDOW_SIZE = 8  # 8 轮对话


@dataclass
class DialogTurn:
    """单轮对话"""

    ts: str
    user_content: str
    assistant_content: str


def parse_line(line: str) -> Optional[DialogTurn]:
    """
    解析单行数据
    格式: ts,request_content ➡️ ,response_content
    """
    line = line.strip()
    if not line or line.startswith("ts,"):
        return None

    # 找到时间戳
    first_comma = line.find(",")
    if first_comma == -1:
        return None

    ts = line[:first_comma]
    rest = line[first_comma + 1 :]

    # 使用 ➡️ 分隔 request 和 response
    separator = " ➡️ ,"
    sep_idx = rest.find(separator)

    if sep_idx == -1:
        separator = "➡️,"
        sep_idx = rest.find(separator)

    if sep_idx == -1:
        return None

    user_content = rest[:sep_idx].strip()
    assistant_content = rest[sep_idx + len(separator) :].strip()

    if not user_content or not assistant_content:
        return None

    return DialogTurn(
        ts=ts, user_content=user_content, assistant_content=assistant_content
    )


def parse_file(file_path: Path) -> List[DialogTurn]:
    """解析文件，返回对话列表"""
    dialogs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            turn = parse_line(line)
            if turn:
                dialogs.append(turn)

    return dialogs


def create_system_prompt(lang: str, ts: str) -> str:
    """
    创建 system prompt
    lang: 'zh' (简体中文) 或 'tw' (繁体中文)
    ts: 当前对话时间
    """
    lang_name = "简体中文" if lang == "zh" else "繁体中文"

    return f"""你需要扮演一个虚拟的角色。
- 性別：男生
- 对话对象：女生
- 输出语言：{lang_name}
- 风格：口语、亲密、自然"""


def create_training_sample(
    dialogs: List[DialogTurn], start_idx: int, lang: str, window_size: int = WINDOW_SIZE
) -> Optional[Dict[str, Any]]:
    """
    创建单个训练样本

    Args:
        dialogs: 对话列表
        start_idx: 起始索引
        lang: 语言类型 ('zh' 或 'tw')
        window_size: 窗口大小（对话轮数）

    Returns:
        训练样本字典，如果数据不足则返回 None
    """
    # 检查是否有足够的对话
    if start_idx + window_size > len(dialogs):
        return None

    # 获取窗口内的对话
    window_dialogs = dialogs[start_idx : start_idx + window_size]

    # 使用最后一轮对话（label）的时间作为 system prompt 的时间
    last_ts = window_dialogs[window_size - 1].ts

    # 构建 messages
    messages = [{"role": "system", "content": create_system_prompt(lang, last_ts)}]

    # 添加前 (window_size - 1) 轮完整对话
    for i in range(window_size - 1):
        turn = window_dialogs[i]
        messages.append({"role": "user", "content": turn.user_content})
        messages.append({"role": "assistant", "content": turn.assistant_content})

    # 添加最后一轮的 user 消息
    last_turn = window_dialogs[window_size - 1]
    messages.append({"role": "user", "content": last_turn.user_content})

    # label 是最后一轮 AI 的回复
    label = last_turn.assistant_content

    return {"messages": messages, "label": label}


def process_file(file_path: Path, lang: str) -> List[Dict[str, Any]]:
    """
    处理单个文件，使用滑动窗口生成训练样本

    Args:
        file_path: 文件路径
        lang: 语言类型

    Returns:
        训练样本列表
    """
    dialogs = parse_file(file_path)
    samples = []

    if len(dialogs) < WINDOW_SIZE:
        print(f"  ⚠️ {file_path.name}: 对话数不足 {WINDOW_SIZE} 轮，跳过")
        return samples

    # 滑动窗口
    for i in range(len(dialogs) - WINDOW_SIZE + 1):
        sample = create_training_sample(dialogs, i, lang)
        if sample:
            samples.append(sample)

    print(f"  ✓ {file_path.name}: {len(dialogs)} 轮对话 → {len(samples)} 个样本")
    return samples


def process_directory(src_dir: Path, lang: str) -> List[Dict[str, Any]]:
    """
    处理目录下的所有文件

    Args:
        src_dir: 源目录
        lang: 语言类型

    Returns:
        所有训练样本
    """
    all_samples = []

    txt_files = sorted(src_dir.glob("*.txt"))
    print(f"\n📂 处理 {lang.upper()} 目录: {src_dir}")
    print(f"   找到 {len(txt_files)} 个文件")

    for file_path in txt_files:
        samples = process_file(file_path, lang)
        all_samples.extend(samples)

    return all_samples


def save_jsonl(samples: List[Dict[str, Any]], output_path: Path):
    """保存为 JSONL 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)

    # 检查源目录
    zh_dir = src_dir / "zh"
    tw_dir = src_dir / "tw"

    if not zh_dir.exists() and not tw_dir.exists():
        print(f"❌ 源目录不存在: {zh_dir} 和 {tw_dir}")
        print("   请先运行 clean_data.py 清洗数据")
        return

    # 创建输出目录
    dst_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("📝 数据格式化脚本")
    print("=" * 60)
    print(f"窗口大小: {WINDOW_SIZE} 轮对话")
    print(f"输出目录: {dst_dir}/")

    total_zh = 0
    total_tw = 0

    # 处理简体中文
    if zh_dir.exists():
        zh_samples = process_directory(zh_dir, "zh")
        if zh_samples:
            output_path = dst_dir / "train_zh.jsonl"
            save_jsonl(zh_samples, output_path)
            total_zh = len(zh_samples)
            print(f"\n   💾 保存: {output_path} ({total_zh} 个样本)")

    # 处理繁体中文
    if tw_dir.exists():
        tw_samples = process_directory(tw_dir, "tw")
        if tw_samples:
            output_path = dst_dir / "train_tw.jsonl"
            save_jsonl(tw_samples, output_path)
            total_tw = len(tw_samples)
            print(f"\n   💾 保存: {output_path} ({total_tw} 个样本)")

    # 合并所有样本
    all_samples = []
    if zh_dir.exists():
        all_samples.extend(
            process_directory(zh_dir, "zh") if not total_zh else zh_samples
        )
    if tw_dir.exists():
        all_samples.extend(
            process_directory(tw_dir, "tw") if not total_tw else tw_samples
        )

    if all_samples:
        output_path = dst_dir / "train_all.jsonl"
        save_jsonl(all_samples, output_path)
        print(f"\n   💾 保存: {output_path} ({len(all_samples)} 个样本)")

    print("\n" + "=" * 60)
    print("✅ 完成!")
    print(f"   简体中文样本: {total_zh}")
    print(f"   繁体中文样本: {total_tw}")
    print(f"   总计: {total_zh + total_tw}")
    print("=" * 60)


if __name__ == "__main__":
    main()
