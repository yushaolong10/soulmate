#!/usr/bin/env python3
"""
DPO 数据生成脚本
从 SFT 格式数据生成 DPO 训练数据

输入: datasets0211_train/dpo_src/base/*.jsonl (SFT 格式)
输出: datasets0211_train/dpo/*.jsonl (DPO 格式)

DPO 数据格式:
{
    "prompt": [...messages...],
    "chosen": "正确回复",
    "rejected": "负样本回复",
    "rejected_type": "负样本类型"
}

负样本类型:
1. formal_long: 书面语、变长、变说教
2. fake_name: 乱叫名字/乱编昵称
3. too_long: 太长
4. repeat_no_emotion: 复读用户/没回应情绪
5. repetitive_phrase: 口头禅重复率高
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# ============ 配置区 ============
API_BASE_URL = "http://172.20.69.79:8026/v1/chat/completions"
MODEL_NAME = "geek"

SRC_DIR = "datasets0211_train/dpo_src"
DST_DIR = "datasets0211_train/dpo"

# 每种负样本类型的权重（用于随机选择）
REJECT_TYPE_WEIGHTS = {
    "formal_long": 0.25,      # 书面语、变长、变说教
    "fake_name": 0.15,        # 乱叫名字/乱编昵称
    "too_long": 0.20,         # 太长
    "repeat_no_emotion": 0.20, # 复读用户/没回应情绪
    "repetitive_phrase": 0.20, # 口头禅重复率高
}

# 常见的虚假昵称（用于 fake_name 类型）
FAKE_NAMES = [
    "親愛的", "小可愛", "甜心",
    "小仙女", "小公主", "小寶貝", "乖乖", "小甜甜", "小美女",
    "我的小可爱", "小天使", "美女", "小姐姐",
]

# 常见的口头禅（用于 repetitive_phrase 类型）
REPETITIVE_PHRASES = [
    "嗚嗚嗚～", "嘿嘿嘿～", "哈哈哈～", "呵呵呵～",
    "你真的太會了啦", "好喜歡你喔", "你好可愛喔",
    "嗯嗯嗯", "對呀對呀", "真的假的", "天啊天啊",
    "哎呀哎呀", "好啦好啦", "是喔是喔",
    "你說的對", "我也是", "我也覺得",
]


def call_llm(messages: List[Dict[str, str]], temperature: float = 0.9) -> Optional[str]:
    """调用 LLM API"""
    try:
        response = requests.post(
            API_BASE_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠️ API 调用失败: {e}")
        return None


def get_last_user_message(messages: List[Dict[str, str]]) -> str:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def generate_formal_long(messages: List[Dict[str, str]], chosen: str) -> Optional[str]:
    """生成书面语、变长、变说教的负样本"""
    system_prompt = """你是一个文字改写助手。请将下面的回复改写成：
1. 更书面化、正式的语气
2. 更啰嗦
3. 带有说教意味
4. 去掉可爱的语气词和表情

只输出改写后的内容，不要任何解释。"""
    
    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"原文：{chosen}\n\n请改写："},
    ]
    
    result = call_llm(rewrite_messages, temperature=0.7)
    if result and len(result) > len(chosen):
        return result
    return None


def generate_fake_name(messages: List[Dict[str, str]], chosen: str) -> str:
    """生成乱叫名字/乱编昵称的负样本"""
    fake_name = random.choice(FAKE_NAMES)
    
    # 在回复开头或中间插入假昵称
    if random.random() < 0.5:
        # 开头插入
        return f"{fake_name}，{chosen}"
    else:
        # 替换或插入
        parts = chosen.split("，")
        if len(parts) > 1:
            insert_pos = random.randint(0, len(parts) - 1)
            parts.insert(insert_pos, fake_name)
            return "，".join(parts)
        return f"{chosen}，{fake_name}"


def generate_too_long(messages: List[Dict[str, str]], chosen: str) -> Optional[str]:
    """生成太长的负样本"""
    system_prompt = """你是一个文字扩写助手。请将下面的回复扩写成更长的版本：
1. 添加更多细节描述
2. 重复表达类似的意思
3. 加入更多语气词和感叹
4. 保持相同的语言风格

只输出扩写后的内容，不要任何解释。至少扩写到原文的2-3倍长度。"""
    
    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"原文：{chosen}\n\n请扩写："},
    ]
    
    result = call_llm(rewrite_messages, temperature=0.8)
    if result and len(result) > len(chosen) * 1.5:
        return result
    return None


def generate_repeat_no_emotion(messages: List[Dict[str, str]], chosen: str) -> str:
    """生成复读用户/没回应情绪的负样本"""
    last_user = get_last_user_message(messages)
    
    if random.random() < 0.5:
        # 复读用户
        variations = [
            f"嗯，{last_user}",
            f"哦，{last_user}嗎",
            f"你說{last_user}啊",
            f"{last_user}，好的",
            f"嗯嗯，{last_user}",
        ]
        return random.choice(variations)
    else:
        # 没回应情绪的冷淡回复
        cold_responses = [
            "嗯",
            "哦",
            "好",
            "知道了",
            "嗯嗯",
            "好的",
            "了解",
            "收到",
            "OK",
            "嗯哼",
        ]
        return random.choice(cold_responses)


def generate_repetitive_phrase(messages: List[Dict[str, str]], chosen: str) -> str:
    """生成口头禅重复率高的负样本"""
    phrase = random.choice(REPETITIVE_PHRASES)
    
    # 多次重复口头禅
    if random.random() < 0.5:
        # 开头重复
        return f"{phrase}{phrase}{chosen}"
    else:
        # 穿插重复
        parts = chosen.split("，")
        result = []
        for i, part in enumerate(parts):
            result.append(part)
            if i < len(parts) - 1 and random.random() < 0.5:
                result.append(phrase)
        return "，".join(result) + phrase


def generate_rejected(
    reject_type: str,
    messages: List[Dict[str, str]],
    chosen: str,
) -> Optional[str]:
    """根据类型生成负样本"""
    if reject_type == "formal_long":
        return generate_formal_long(messages, chosen)
    elif reject_type == "fake_name":
        return generate_fake_name(messages, chosen)
    elif reject_type == "too_long":
        return generate_too_long(messages, chosen)
    elif reject_type == "repeat_no_emotion":
        return generate_repeat_no_emotion(messages, chosen)
    elif reject_type == "repetitive_phrase":
        return generate_repetitive_phrase(messages, chosen)
    return None


def select_reject_type() -> str:
    """根据权重随机选择负样本类型"""
    types = list(REJECT_TYPE_WEIGHTS.keys())
    weights = list(REJECT_TYPE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def process_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """处理单个样本，生成 DPO 数据"""
    messages = sample.get("messages", [])
    chosen = sample.get("label", "")
    
    if not messages or not chosen:
        return None
    
    # 随机选择负样本类型
    reject_type = select_reject_type()
    
    # 生成负样本
    rejected = generate_rejected(reject_type, messages, chosen)
    
    if not rejected:
        # 如果生成失败，使用备用类型
        backup_types = ["fake_name", "repeat_no_emotion", "repetitive_phrase"]
        for backup_type in backup_types:
            rejected = generate_rejected(backup_type, messages, chosen)
            if rejected:
                reject_type = backup_type
                break
    
    if not rejected:
        return None
    
    # 确保 rejected 和 chosen 不同
    if rejected.strip() == chosen.strip():
        return None
    
    return {
        "prompt": messages,
        "chosen": chosen,
        "rejected": rejected,
        "rejected_type": reject_type,
    }


def process_file(src_path: Path, dst_path: Path) -> int:
    """处理单个文件"""
    print(f"\n📄 处理文件: {src_path.name}")
    
    samples = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"   共 {len(samples)} 个样本")
    
    dpo_samples = []
    success_count = 0
    fail_count = 0
    
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"   进度: {i + 1}/{len(samples)} (成功: {success_count}, 失败: {fail_count})")
        
        dpo_sample = process_sample(sample)
        if dpo_sample:
            dpo_samples.append(dpo_sample)
            success_count += 1
        else:
            fail_count += 1
        
        # 避免请求过快
        time.sleep(0.1)
    
    # 保存结果
    with open(dst_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"   ✅ 完成: {success_count} 个 DPO 样本")
    return success_count


def main():
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)
    
    if not src_dir.exists():
        print(f"❌ 源目录不存在: {src_dir}")
        return
    
    # 创建输出目录
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 jsonl 文件
    jsonl_files = list(src_dir.glob("*.jsonl"))
    
    print("=" * 60)
    print("🔄 DPO 数据生成脚本")
    print("=" * 60)
    print(f"源目录: {src_dir}")
    print(f"输出目录: {dst_dir}")
    print(f"找到 {len(jsonl_files)} 个文件")
    print(f"API: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print()
    print("负样本类型分布:")
    for reject_type, weight in REJECT_TYPE_WEIGHTS.items():
        print(f"  - {reject_type}: {weight * 100:.0f}%")
    
    total_samples = 0
    
    for src_path in jsonl_files:
        dst_path = dst_dir / f"dpo_{src_path.stem}.jsonl"
        count = process_file(src_path, dst_path)
        total_samples += count
    
    print("\n" + "=" * 60)
    print(f"✅ 完成! 共生成 {total_samples} 个 DPO 样本")
    print(f"   输出目录: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
