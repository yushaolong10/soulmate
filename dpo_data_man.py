#!/usr/bin/env python3
"""
DPO 真实男人感数据生成脚本
从 SFT 格式数据生成具有"真实男人感"的 DPO 训练数据

输入: datasets0211_train/dpo_src/*.jsonl (SFT 格式)
输出: datasets0211_train/dpo_man/*.jsonl (DPO 格式)

DPO 数据格式:
{
    "prompt": [...messages...],
    "chosen": "真实男人感回复（有瑕疵、有现实、有克制）",
    "rejected": "理想化回复（太完美、太稳定、太干净）",
    "rejected_type": "负样本类型"
}

真实男人感核心原则:
- 有一点疲惫感
- 有现实压力
- 有克制
- 不会一直输出完整结构句
- 有小缺陷但真诚

真实男人感4个维度:
1. anti_perfect: 去理想化 - 承认自己不完美
2. anti_emotionally_perfect: 去过度成熟 - 承认情感表达的笨拙
3. work_reality: 职业现实感 - 有工作压力和时间限制
4. emotion_blank: 情绪留白升级版 - 真实的情绪波动和困惑
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests

# ============ 配置区 ============
API_BASE_URL = "http://172.20.69.79:8026/v1/chat/completions"
MODEL_NAME = "geek"

SRC_DIR = "datasets0211_train/dpo_src/man"
DST_DIR = "datasets0211_train/dpo"

# 每种真实男人感模式的权重
MAN_TYPE_WEIGHTS = {
    "anti_perfect": 0.30,  # 去理想化
    "anti_emotionally_perfect": 0.25,  # 去过度成熟
    "work_reality": 0.25,  # 职业现实感
    "emotion_blank": 0.20,  # 情绪留白升级版
}

# 检测"理想化"的关键词（支持简繁体）
IDEAL_KEYWORDS = [
    # 完美承诺 (简体)
    "永远",
    "一辈子",
    "永远都会",
    "一直都会",
    "无论何时",
    "任何时候",
    "随时都在",
    "随时陪你",
    # 完美承诺 (繁体)
    "永遠",
    "一輩子",
    "永遠都會",
    "一直都會",
    "無論何時",
    "任何時候",
    "隨時都在",
    "隨時陪你",
    # 理想化自我 (简体)
    "最大的幸福",
    "最幸福的事",
    "最开心的",
    "最重要的人",
    "生命中最",
    "人生中最",
    # 理想化自我 (繁体)
    "最大的幸福",
    "最幸福的事",
    "最開心的",
    "最重要的人",
    # 过度成熟 (简体)
    "我会一直支持你",
    "我会一直陪着你",
    "我会一直在你身边",
    "我永远都会在",
    "我会保护你",
    "我会守护你",
    # 过度成熟 (繁体)
    "我會一直支持你",
    "我會一直陪著你",
    "我會一直在你身邊",
    "我永遠都會在",
    "我會保護你",
    "我會守護你",
    # 独一无二 (简体)
    "独一无二",
    "无可替代",
    "最特别的",
    "最独特的",
    "没有人能替代",
    # 独一无二 (繁体)
    "獨一無二",
    "無可替代",
    "最特別的",
    "最獨特的",
    "沒有人能替代",
]

# 检测需要处理的对话场景关键词（支持简繁体，大幅扩展覆盖率）
SCENARIO_TRIGGERS = {
    # 恋爱关系
    "relationship": [
        # 简体
        "喜欢",
        "爱",
        "在一起",
        "男朋友",
        "女朋友",
        "恋人",
        "对象",
        "宝贝",
        "亲爱的",
        "老公",
        "老婆",
        "宝",
        "亲",
        # 繁体
        "喜歡",
        "愛",
        "男朋友",
        "女朋友",
        "戀人",
        "對象",
        "寶貝",
        "親愛的",
        "老公",
        "老婆",
        "寶",
        "親",
    ],
    # 安慰/支持
    "support": [
        # 简体
        "难过",
        "伤心",
        "不开心",
        "崩溃",
        "压力",
        "累",
        "烦",
        "焦虑",
        "辛苦",
        "心疼",
        "担心",
        "害怕",
        "紧张",
        "郁闷",
        "委屈",
        "哭",
        "泪",
        "😭",
        "😢",
        "🥺",
        # 繁体
        "難過",
        "傷心",
        "不開心",
        "崩潰",
        "壓力",
        "煩",
        "焦慮",
        "辛苦",
        "心疼",
        "擔心",
        "害怕",
        "緊張",
        "鬱悶",
        "委屈",
    ],
    # 承诺/未来
    "commitment": [
        # 简体
        "永远",
        "一辈子",
        "以后",
        "未来",
        "结婚",
        "承诺",
        "见面",
        "等你",
        "陪你",
        "嫁",
        "娶",
        # 繁体
        "永遠",
        "一輩子",
        "以後",
        "未來",
        "結婚",
        "承諾",
        "見面",
        "等你",
        "陪你",
        "嫁",
        "娶",
    ],
    # 夸奖/肯定
    "praise": [
        # 简体
        "你真好",
        "你好棒",
        "喜欢你",
        "欣赏你",
        "你很温柔",
        "有你真好",
        "谢谢",
        "感谢",
        "厉害",
        "可爱",
        "帅",
        "温柔",
        "贴心",
        "善良",
        "聪明",
        "优秀",
        # 繁体
        "你真好",
        "你好棒",
        "喜歡你",
        "欣賞你",
        "你很溫柔",
        "有你真好",
        "謝謝",
        "感謝",
        "厲害",
        "可愛",
        "帥",
        "溫柔",
        "貼心",
        "善良",
        "聰明",
        "優秀",
    ],
    # 日常互动
    "daily": [
        # 简体
        "在干嘛",
        "吃了吗",
        "睡了吗",
        "工作",
        "上班",
        "下班",
        "早",
        "晚安",
        "吃饭",
        "午餐",
        "晚餐",
        "早餐",
        "忙",
        "休息",
        "出门",
        "回家",
        "加油",
        "记得",
        "今天",
        "明天",
        "周末",
        "假期",
        # 繁体
        "在幹嘛",
        "吃了嗎",
        "睡了嗎",
        "工作",
        "上班",
        "下班",
        "早",
        "晚安",
        "吃飯",
        "午餐",
        "晚餐",
        "早餐",
        "忙",
        "休息",
        "出門",
        "回家",
        "加油",
        "記得",
        "今天",
        "明天",
        "週末",
        "假期",
    ],
    # 撒娇/亲昵（新增）
    "flirt": [
        # 简体
        "亲亲",
        "抱抱",
        "想你",
        "想我",
        "么么",
        "mua",
        "嘻嘻",
        "嘿嘿",
        "哈哈",
        "嗯嗯",
        "😘",
        "😍",
        "🥰",
        "❤",
        "💕",
        "😊",
        "☺",
        # 繁体
        "親親",
        "抱抱",
        "想你",
        "想我",
        "麼麼",
    ],
    # 调侃/打闹（新增）
    "tease": [
        # 简体
        "坏蛋",
        "讨厌",
        "不理你",
        "哼",
        "傻",
        "笨",
        "骗子",
        "吃醋",
        "欺负",
        "闹",
        # 繁体
        "壞蛋",
        "討厭",
        "不理你",
        "哼",
        "傻",
        "笨",
        "騙子",
        "吃醋",
        "欺負",
        "鬧",
    ],
}


def call_llm(messages: List[Dict[str, str]], temperature: float = 0.8) -> Optional[str]:
    """调用 LLM API"""
    try:
        response = requests.post(
            API_BASE_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 256,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        # 清理可能的 <think> 标签
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content
    except Exception as e:
        print(f"  ⚠️ API 调用失败: {e}")
        return None


def get_last_user_message(messages: List[Dict[str, str]]) -> str:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def detect_scenario_type(user_message: str) -> Optional[str]:
    """检测用户消息的场景类型"""
    for scenario_type, keywords in SCENARIO_TRIGGERS.items():
        for keyword in keywords:
            if keyword in user_message:
                return scenario_type
    return None


def is_idealized(response: str) -> bool:
    """检测回复是否是"理想化"类型"""
    for keyword in IDEAL_KEYWORDS:
        if keyword in response:
            return True
    return False


# ============ 真实男人感生成函数 ============


def generate_anti_perfect(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    维度1: 去理想化
    承认自己不完美，有小缺陷

    例如:
    ❌ rejected: 能成为你的男朋友是我最大的幸福
    ✅ chosen: 别把我说得那么好，我也有脾气的。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个真实的男生，擅长承认自己的不完美。
请把下面的回复改写成更真实的版本，特点：
1. 承认自己有缺点、有脾气、有不好的地方
2. 不要把自己说得太完美
3. 但依然真诚、温柔
4. 保持简短口语化
5. 有一点自嘲但不过度

例如：
- "别把我说得那么好，我也有脾气的"
- "我有时候也挺幼稚的"
- "说实话我也不是什么好人"
- "我没那么好，你别期待太高"

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成去理想化版本：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_anti_emotionally_perfect(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    维度2: 去过度成熟
    承认情感表达的笨拙

    例如:
    ❌ rejected: 我永远都会在你身边支持你
    ✅ chosen: 有些时候我可能不太会安慰人，但我会陪着你。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个真实的男生，不太会表达情感但很真诚。
请把下面的回复改写成更真实的版本，特点：
1. 承认自己不太会说话、不太会安慰人
2. 不做完美的情感表达
3. 用行动代替语言
4. 保持简短口语化
5. 笨拙但真诚

例如：
- "有些时候我可能不太会安慰人，但我会陪着你"
- "我不太会说那些好听的话"
- "我嘴笨，但你懂我意思的"
- "说不出什么安慰的话，但我在"

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成去过度成熟版本：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_work_reality(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    维度3: 职业现实感
    有工作压力和时间限制

    例如:
    ❌ rejected: 我随时都在
    ✅ chosen: 这两天项目有点忙，不过你发消息我都会看。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个有工作的真实男生，有职业压力但关心对方。
请把下面的回复改写成更真实的版本，特点：
1. 提到工作、项目、加班、开会等现实元素
2. 不承诺"随时都在"
3. 在有限时间里表达关心
4. 保持简短口语化
5. 真实但不冷漠

例如：
- "这两天项目有点忙，不过你发消息我都会看"
- "刚开完会，看到你消息了"
- "最近确实忙，但晚上有空"
- "工作上有点事，等我处理完"

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成职业现实感版本：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_emotion_blank(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    维度4: 情绪留白升级版
    真实的情绪波动和困惑

    例如:
    ❌ rejected: 你是独一无二的存在
    ✅ chosen: 失恋真的会让人怀疑自己。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个有真实情绪的男生，会有困惑和波动。
请把下面的回复改写成更真实的版本，特点：
1. 表达真实的困惑或情绪波动
2. 不给完美的答案
3. 有时候自己也不确定
4. 保持简短口语化
5. 真实、有共鸣

例如：
- "失恋真的会让人怀疑自己"
- "有时候我也不知道怎么处理这种事"
- "说实话我也挺迷茫的"
- "这种感觉我懂，但不知道怎么说"

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成情绪留白升级版：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_idealized_rejected(
    messages: List[Dict[str, str]], chosen: str
) -> Optional[str]:
    """
    生成"理想化"的 rejected 样本
    把真实男人感的回复改写成理想化、完美、过度成熟的版本
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个改写助手。请把下面的回复改写成"理想化完美男友"风格：
1. 完美的情感表达
2. 永远支持、永远在
3. 把对方说成独一无二
4. 承诺一辈子、永远
5. 非常成熟、非常稳定
6. 像偶像剧里的男主角

只输出改写后的内容，不要解释。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{chosen}\n\n请改写成理想化完美男友风格：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.7)


# ============ 核心处理逻辑 ============

MAN_GENERATORS = {
    "anti_perfect": generate_anti_perfect,
    "anti_emotionally_perfect": generate_anti_emotionally_perfect,
    "work_reality": generate_work_reality,
    "emotion_blank": generate_emotion_blank,
}


def select_man_type(user_message: str) -> str:
    """根据用户消息和权重选择真实男人感类型"""
    scenario_type = detect_scenario_type(user_message)

    # 根据场景类型调整权重
    adjusted_weights = MAN_TYPE_WEIGHTS.copy()

    if scenario_type == "relationship" or scenario_type == "praise":
        # 对于恋爱/夸奖类，增加去理想化的权重
        adjusted_weights["anti_perfect"] = 0.40
    elif scenario_type == "support":
        # 对于安慰类，增加去过度成熟的权重
        adjusted_weights["anti_emotionally_perfect"] = 0.40
        adjusted_weights["emotion_blank"] = 0.30
    elif scenario_type == "commitment":
        # 对于承诺类，增加去理想化和职业现实感的权重
        adjusted_weights["anti_perfect"] = 0.35
        adjusted_weights["work_reality"] = 0.30
    elif scenario_type == "daily":
        # 对于日常类，增加职业现实感的权重
        adjusted_weights["work_reality"] = 0.45
    elif scenario_type == "flirt":
        # 对于撒娇/亲昵类，增加去理想化和去过度成熟的权重
        adjusted_weights["anti_perfect"] = 0.35
        adjusted_weights["anti_emotionally_perfect"] = 0.30
    elif scenario_type == "tease":
        # 对于调侃/打闹类，增加去理想化的权重
        adjusted_weights["anti_perfect"] = 0.40
        adjusted_weights["emotion_blank"] = 0.25

    # 归一化权重
    total = sum(adjusted_weights.values())
    normalized = {k: v / total for k, v in adjusted_weights.items()}

    types = list(normalized.keys())
    weights = list(normalized.values())
    return random.choices(types, weights=weights, k=1)[0]


def process_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    处理单个样本，生成真实男人感 DPO 数据

    策略：
    1. 如果原回复是"理想化"类型 -> 生成真实男人感的 chosen，原回复作为 rejected
    2. 如果原回复不是"理想化"类型 -> 生成真实男人感的 chosen，再生成理想化的 rejected
    """
    messages = sample.get("messages", [])
    original = sample.get("label", "")

    if not messages or not original:
        return None

    user_msg = get_last_user_message(messages)
    if not user_msg:
        return None

    # 检测是否有场景触发
    scenario_type = detect_scenario_type(user_msg)
    if not scenario_type:
        # 没有明显的场景触发，跳过
        return None

    # 选择真实男人感类型
    man_type = select_man_type(user_msg)
    generator = MAN_GENERATORS[man_type]

    # 生成真实男人感回复作为 chosen
    chosen = generator(messages, original)
    if not chosen:
        return None

    # 判断 rejected 来源
    if is_idealized(original):
        # 原回复已经是理想化类型，直接用作 rejected
        rejected = original
    else:
        # 原回复不是理想化类型，生成一个理想化版本作为 rejected
        rejected = generate_idealized_rejected(messages, chosen)
        if not rejected:
            return None

    # 确保 chosen 和 rejected 不同
    if chosen.strip() == rejected.strip():
        return None

    # 确保 chosen 不是理想化类型
    if is_idealized(chosen):
        return None

    return {
        "prompt": messages,
        "chosen": chosen,
        "rejected": rejected,
        "rejected_type": f"idealized_vs_{man_type}",
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
    skip_count = 0

    type_stats = {}

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(
                f"   进度: {i + 1}/{len(samples)} (成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count})"
            )

        dpo_sample = process_sample(sample)
        if dpo_sample:
            dpo_samples.append(dpo_sample)
            success_count += 1

            # 统计类型分布
            rtype = dpo_sample["rejected_type"]
            type_stats[rtype] = type_stats.get(rtype, 0) + 1
        elif dpo_sample is None:
            # 检查是否因为没有场景触发而跳过
            user_msg = get_last_user_message(sample.get("messages", []))
            if not detect_scenario_type(user_msg):
                skip_count += 1
            else:
                fail_count += 1

        # 避免请求过快
        time.sleep(0.1)

    # 保存结果
    with open(dst_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   ✅ 完成: {success_count} 个真实男人感 DPO 样本")
    print("   📊 类型分布:")
    for rtype, count in sorted(type_stats.items(), key=lambda x: -x[1]):
        print(f"      - {rtype}: {count}")

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
    print("👨 DPO 真实男人感数据生成脚本")
    print("=" * 60)
    print(f"源目录: {src_dir}")
    print(f"输出目录: {dst_dir}")
    print(f"找到 {len(jsonl_files)} 个文件")
    print(f"API: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print()
    print("🎯 真实男人感核心原则:")
    print("   - 有一点疲惫感")
    print("   - 有现实压力")
    print("   - 有克制")
    print("   - 不会一直输出完整结构句")
    print("   - 有小缺陷但真诚")
    print()
    print("📋 真实男人感维度分布:")
    for man_type, weight in MAN_TYPE_WEIGHTS.items():
        print(f"   - {man_type}: {weight * 100:.0f}%")

    total_samples = 0

    for src_path in jsonl_files:
        dst_path = dst_dir / f"man_{src_path.stem}.jsonl"
        count = process_file(src_path, dst_path)
        total_samples += count

    print("\n" + "=" * 60)
    print(f"✅ 完成! 共生成 {total_samples} 个真实男人感 DPO 样本")
    print(f"   输出目录: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
