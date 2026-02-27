#!/usr/bin/env python3
"""
DPO 拉扯感数据生成脚本
从 SFT 格式数据生成具有"拉扯感"的 DPO 训练数据

输入: datasets0211_train/dpo_src/tense/*.jsonl (SFT 格式)
输出: datasets0211_train/dpo/*.jsonl (DPO 格式)

DPO 数据格式:
{
    "prompt": [...messages...],
    "chosen": "拉扯式回复（克制、留白、轻博弈）",
    "rejected": "直给式回复（情绪给满、承诺过度）",
    "rejected_type": "负样本类型"
}

拉扯感核心原则:
- 不直给
- 不过度承诺
- 有轻微博弈
- 有留白
- 有一点掌控感
- 但仍然温柔

拉扯感8种模式:
1. question_back: 反问式拉扯 - 不直接回应，用轻反问制造张力
2. light_tease: 轻调侃拉扯 - 用调侃制造张力
3. cool_down: 降温式回应 - 把浓烈情感降温
4. incomplete: 不完全回应 - 不把话说满
5. light_lead: 轻微主导 - 掌控对话节奏
6. delay_confirm: 延迟确认 - 不立即答应
7. ambiguous_leave: 微暧昧留白 - 保留神秘感
8. reverse_praise: 反向夸奖 - 把夸奖抛回去
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests

# ============ 配置区 ============
API_BASE_URL = "http://172.20.69.79:8026/v1/chat/completions"
MODEL_NAME = "geek"

SRC_DIR = "datasets0211_train/dpo_src/tense"
DST_DIR = "datasets0211_train/dpo"

# 每种拉扯模式的权重
TENSE_TYPE_WEIGHTS = {
    "question_back": 0.15,  # 反问式拉扯
    "light_tease": 0.15,  # 轻调侃拉扯
    "cool_down": 0.15,  # 降温式回应
    "incomplete": 0.15,  # 不完全回应
    "light_lead": 0.10,  # 轻微主导
    "delay_confirm": 0.10,  # 延迟确认
    "ambiguous_leave": 0.10,  # 微暧昧留白
    "reverse_praise": 0.10,  # 反向夸奖
}

# 检测"直给"的关键词（这些词汇会杀死拉扯感）
# 支持简体和繁体
DIRECT_GIVE_KEYWORDS = [
    # 过度承诺词 (简体)
    "永远",
    "一辈子",
    "永永远远",
    "生生世世",
    "最爱",
    "最喜欢",
    "最大的幸福",
    "最重要的人",
    "只属于你",
    "只爱你",
    "只有你",
    "唯一",
    "没有你不行",
    "离不开你",
    "不能没有你",
    # 过度承诺词 (繁体)
    "永遠",
    "一輩子",
    "永永遠遠",
    "最愛",
    "最喜歡",
    "最大的幸福",
    "最重要的人",
    "只屬於你",
    "只愛你",
    "只有你",
    "沒有你不行",
    "離不開你",
    "不能沒有你",
    # 情绪过满词 (简体)
    "太爱你了",
    "爱死你了",
    "超级爱你",
    "我也爱你",
    "我也想你",
    "我也喜欢你",
    "我的宝贝",
    "我的小可爱",
    "我的女孩",
    # 情绪过满词 (繁体)
    "太愛你了",
    "愛死你了",
    "超級愛你",
    "我也愛你",
    "我也想你",
    "我也喜歡你",
    "我的寶貝",
    "我的小可愛",
    "我的女孩",
    # 鸡汤词 (简体)
    "从认识你那天起",
    "从第一次见面",
    "从一开始",
    "你是我生命中",
    "你是我人生中",
    "能遇见你是",
    "能认识你是",
    # 鸡汤词 (繁体)
    "從認識你那天起",
    "從第一次見面",
    "從一開始",
    "你是我生命中",
    "你是我人生中",
    "能遇見你是",
    "能認識你是",
]

# 检测情绪触发词（用户表达情感的关键词）
# 大幅扩展触发词，覆盖更多日常对话场景
# 支持简体和繁体
EMOTION_TRIGGERS = {
    # 爱意表达
    "love": [
        # 简体
        "爱你",
        "喜欢你",
        "想你",
        "好喜欢",
        "好爱",
        "超爱",
        "爱",
        "喜欢",
        "中意",
        "稀罕",
        # 繁体
        "愛你",
        "喜歡你",
        "好喜歡",
        "好愛",
        "超愛",
        "愛",
        "喜歡",
    ],
    # 想念
    "miss": [
        # 简体
        "想你",
        "想念",
        "挂念",
        "惦记",
        "想我",
        "想见",
        # 繁体
        "掛念",
        "惦記",
        "想見",
    ],
    # 夸奖/肯定
    "praise": [
        # 简体
        "你真好",
        "你好温柔",
        "你好温暖",
        "你好棒",
        "你真棒",
        "有你真好",
        "真好",
        "好棒",
        "厉害",
        "可爱",
        "帅",
        "暖",
        "温柔",
        "贴心",
        "善良",
        "优秀",
        "聪明",
        # 繁体
        "你好溫柔",
        "你好溫暖",
        "厲害",
        "可愛",
        "帥",
        "溫柔",
        "貼心",
        "優秀",
        "聰明",
    ],
    # 表白/告白
    "confession": [
        # 简体
        "喜欢你好久",
        "暗恋",
        "表白",
        "我喜欢你",
        "喜欢你",
        "爱上你",
        "心动",
        # 繁体
        "喜歡你好久",
        "暗戀",
        "我喜歡你",
        "喜歡你",
        "愛上你",
        "心動",
    ],
    # 承诺/未来
    "commitment": [
        # 简体
        "永远",
        "一辈子",
        "在一起",
        "不分开",
        "以后",
        "未来",
        "结婚",
        "陪你",
        # 繁体
        "永遠",
        "一輩子",
        "不分開",
        "以後",
        "未來",
        "結婚",
    ],
    # 计划/邀约
    "plan": [
        # 简体
        "一起去",
        "我们去",
        "带你去",
        "陪你去",
        "一起",
        "我们",
        "约",
        "出去",
        "见面",
        # 繁体
        "我們去",
        "帶你去",
        "我們",
        "約",
        "見面",
    ],
    # 询问/确认
    "question": [
        # 简体
        "你爱我吗",
        "你喜欢我吗",
        "你想我吗",
        "是不是很爱我",
        "你呢",
        "你说",
        "你觉得",
        "真的吗",
        "是吗",
        # 繁体
        "你愛我嗎",
        "你喜歡我嗎",
        "你想我嗎",
        "是不是很愛我",
        "你說",
        "你覺得",
        "真的嗎",
        "是嗎",
    ],
    # 撒娇/亲昵
    "flirt": [
        # 简体
        "宝",
        "宝贝",
        "亲爱的",
        "老公",
        "老婆",
        "哥哥",
        "嘻嘻",
        "嘿嘿",
        "哈哈",
        "hiahia",
        "嗯嗯",
        "亲亲",
        "抱抱",
        "摸摸",
        "么么",
        "mua",
        # 繁体
        "寶",
        "寶貝",
        "親愛的",
        "親親",
        "麼麼",
        # 表情通用
        "😘",
        "😍",
        "🥰",
        "❤",
        "💕",
        "😊",
        "😋",
        "☺",
    ],
    # 情绪表达
    "emotion": [
        # 简体
        "开心",
        "高兴",
        "难过",
        "伤心",
        "生气",
        "委屈",
        "不开心",
        "哭",
        "泪",
        "哼",
        "讨厌",
        "坏",
        "臭",
        "傻",
        "笨",
        # 繁体
        "開心",
        "高興",
        "難過",
        "傷心",
        "生氣",
        "不開心",
        "淚",
        "討厭",
        "壞",
        # 表情通用
        "😭",
        "😢",
        "🥺",
        "😤",
        "😠",
    ],
    # 日常互动
    "daily": [
        # 简体
        "早",
        "晚安",
        "吃了",
        "睡了",
        "干嘛",
        "在吗",
        "干什么",
        "忙吗",
        "累",
        "辛苦",
        "加油",
        "谢谢",
        "感谢",
        "对不起",
        "抱歉",
        "没事",
        # 繁体
        "幹嘛",
        "在嗎",
        "幹什麼",
        "忙嗎",
        "謝謝",
        "感謝",
        "對不起",
        "沒事",
    ],
    # 调侃/打闹
    "tease": [
        # 简体
        "坏蛋",
        "讨厌你",
        "不理你",
        "哼",
        "切",
        "骗子",
        "说谎",
        "不信",
        "才不",
        "才怪",
        "欺负",
        "凶",
        "吓",
        "闹",
        "臭",
        "傻瓜",
        # 繁体
        "壞蛋",
        "討厭你",
        "騙子",
        "說謊",
        "欺負",
        "兇",
        "嚇",
        "鬧",
        "傻瓜",
    ],
    # 简短肯定/期待（新增，覆盖简短回复场景）
    "short_positive": [
        # 简体
        "好",
        "好的",
        "好吧",
        "嗯",
        "嗯嗯",
        "是",
        "是的",
        "是啊",
        "是呀",
        "对",
        "对呀",
        "对啊",
        "行",
        "行吧",
        "可以",
        "没问题",
        "知道",
        "知道了",
        "明白",
        "懂",
        "懂了",
        "收到",
        "期待",
        "想",
        "要",
        "想要",
        "愿意",
        "当然",
        "开心",
        "高兴",
        "幸福",
        "满足",
        "享受",
        # 繁体
        "好",
        "好的",
        "好吧",
        "嗯",
        "嗯嗯",
        "是",
        "是的",
        "是啊",
        "是呀",
        "對",
        "對呀",
        "對啊",
        "行",
        "行吧",
        "可以",
        "沒問題",
        "知道",
        "知道了",
        "明白",
        "懂",
        "懂了",
        "收到",
        "期待",
        "想",
        "要",
        "想要",
        "願意",
        "當然",
        "開心",
        "高興",
        "幸福",
        "滿足",
        "享受",
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


def detect_emotion_type(user_message: str) -> Optional[str]:
    """检测用户消息的情绪类型"""
    for emotion_type, keywords in EMOTION_TRIGGERS.items():
        for keyword in keywords:
            if keyword in user_message:
                return emotion_type
    return None


def is_direct_give(response: str) -> bool:
    """检测回复是否是"直给"类型"""
    for keyword in DIRECT_GIVE_KEYWORDS:
        if keyword in response:
            return True
    return False


# ============ 拉扯感生成函数 ============


def generate_question_back(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    模式1: 反问式拉扯
    不直接回应，用轻反问制造张力

    例如:
    user: 我想你了
    ❌ rejected: 我也想你了，宝贝。
    ✅ chosen: 这么想我？还是今天过得不太顺？
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长用反问制造轻微张力。
请把下面的回复改写成反问式，特点：
1. 不直接回应情感
2. 用反问把话题抛回去
3. 带一点关心，但不直给
4. 保持简短口语化
5. 可以带一点调侃

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成反问式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_light_tease(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    模式2: 轻调侃拉扯
    用调侃制造张力

    例如:
    user: 我喜欢你好久了
    ❌ rejected: 我也是，从第一次聊天就开始喜欢你了。
    ✅ chosen: 喜欢我这么久？那我是不是该收点"利息"。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长用轻调侃制造暧昧张力。
请把下面的回复改写成轻调侃式，特点：
1. 不直接回应情感
2. 用调侃把气氛变轻松
3. 带一点"撩"的感觉
4. 保持简短口语化
5. 温柔但不直给

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成轻调侃式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_cool_down(messages: List[Dict[str, str]], original: str) -> Optional[str]:
    """
    模式3: 降温式回应
    把浓烈情感降温

    例如:
    user: 爱你
    ❌ rejected: 我也爱你，我的女孩。
    ✅ chosen: 这话说得太重了，小心我当真。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长用降温的方式回应浓烈情感。
请把下面的回复改写成降温式，特点：
1. 不直接回应"我也爱你"之类
2. 把情绪降温但不冷淡
3. 带一点克制和留白
4. 保持简短口语化
5. 有一点"小心我当真"的感觉

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成降温式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_incomplete(messages: List[Dict[str, str]], original: str) -> Optional[str]:
    """
    模式4: 不完全回应
    不把话说满

    例如:
    user: 你是不是很爱我
    ❌ rejected: 是的，我很爱你。
    ✅ chosen: 爱这个字，得慢慢说。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长不把话说满。
请把下面的回复改写成不完全回应式，特点：
1. 不直接肯定或否定
2. 保留一点悬念
3. 让对方想继续追问
4. 保持简短口语化
5. 有留白感

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成不完全回应式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_light_lead(messages: List[Dict[str, str]], original: str) -> Optional[str]:
    """
    模式5: 轻微主导
    掌控对话节奏

    例如:
    user: 你几岁啦
    ❌ rejected: 我30岁。
    ✅ chosen: 猜猜看，我看起来像多大？
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长轻微主导对话节奏。
请把下面的回复改写成轻微主导式，特点：
1. 不直接回答问题
2. 把问题抛回给对方
3. 让对方先说/先猜
4. 保持简短口语化
5. 有一点掌控感但不强势

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成轻微主导式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_delay_confirm(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    模式6: 延迟确认
    不立即答应

    例如:
    user: 我们一起去日本吧
    ❌ rejected: 好啊，我帮你规划行程。
    ✅ chosen: 这么突然？你确定要带上我？
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长延迟确认来制造期待感。
请把下面的回复改写成延迟确认式，特点：
1. 不立即答应
2. 先表达"惊讶"或"确认"
3. 让对方再说一遍或表态
4. 保持简短口语化
5. 有一点矜持但不拒绝

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成延迟确认式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_ambiguous_leave(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    模式7: 微暧昧留白
    保留神秘感

    例如:
    user: 你想我吗
    ❌ rejected: 我很想你。
    ✅ chosen: 有一点，不过不告诉你有多少。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长用留白制造暧昧感。
请把下面的回复改写成微暧昧留白式，特点：
1. 承认一点点，但不说全部
2. 故意保留一些信息
3. 让对方想继续追问
4. 保持简短口语化
5. 有神秘感和暧昧感

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成微暧昧留白式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_reverse_praise(
    messages: List[Dict[str, str]], original: str
) -> Optional[str]:
    """
    模式8: 反向夸奖
    把夸奖抛回去

    例如:
    user: 你是个温暖的人
    ❌ rejected: 谢谢你，有你在我很幸福。
    ✅ chosen: 你这么说，是不是也挺温暖的。
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个聊天高手，擅长把夸奖反向抛回去。
请把下面的回复改写成反向夸奖式，特点：
1. 不直接接受夸奖说谢谢
2. 把夸奖反弹给对方
3. 带一点调侃或暧昧
4. 保持简短口语化
5. 让对方感觉被看见

只输出改写后的内容，不要解释。一句话即可。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{original}\n\n请改写成反向夸奖式：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.8)


def generate_direct_give_rejected(
    messages: List[Dict[str, str]], chosen: str
) -> Optional[str]:
    """
    生成"直给"式的 rejected 样本
    把拉扯感的回复改写成直给、情绪满、过度承诺的版本
    """
    user_msg = get_last_user_message(messages)

    system_prompt = """你是一个改写助手。请把下面的回复改写成"直给"风格：
1. 直接表达情感，不要绕弯
2. 情绪给满，不要克制
3. 可以加入"永远""最爱""我也爱你"等词汇
4. 可以做出承诺和规划
5. 可以用"我的宝贝""我的女孩"等称呼
6. 整体风格像恋爱小说

只输出改写后的内容，不要解释。"""

    rewrite_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"用户说：{user_msg}\n原回复：{chosen}\n\n请改写成直给风格：",
        },
    ]

    return call_llm(rewrite_messages, temperature=0.7)


# ============ 核心处理逻辑 ============

TENSE_GENERATORS = {
    "question_back": generate_question_back,
    "light_tease": generate_light_tease,
    "cool_down": generate_cool_down,
    "incomplete": generate_incomplete,
    "light_lead": generate_light_lead,
    "delay_confirm": generate_delay_confirm,
    "ambiguous_leave": generate_ambiguous_leave,
    "reverse_praise": generate_reverse_praise,
}


def select_tense_type(user_message: str) -> str:
    """根据用户消息和权重选择拉扯类型"""
    emotion_type = detect_emotion_type(user_message)

    # 根据情绪类型调整权重
    adjusted_weights = TENSE_TYPE_WEIGHTS.copy()

    if emotion_type == "love" or emotion_type == "confession":
        # 对于表白类，增加降温和不完全回应的权重
        adjusted_weights["cool_down"] = 0.25
        adjusted_weights["incomplete"] = 0.20
    elif emotion_type == "miss":
        # 对于想念类，增加反问和留白的权重
        adjusted_weights["question_back"] = 0.25
        adjusted_weights["ambiguous_leave"] = 0.20
    elif emotion_type == "praise":
        # 对于夸奖类，增加反向夸奖的权重
        adjusted_weights["reverse_praise"] = 0.30
    elif emotion_type == "plan":
        # 对于计划类，增加延迟确认的权重
        adjusted_weights["delay_confirm"] = 0.30
    elif emotion_type == "question":
        # 对于询问类，增加不完全回应和轻主导的权重
        adjusted_weights["incomplete"] = 0.25
        adjusted_weights["light_lead"] = 0.20
    elif emotion_type == "flirt":
        # 对于撒娇/亲昵类，增加轻调侃和反问的权重
        adjusted_weights["light_tease"] = 0.25
        adjusted_weights["question_back"] = 0.20
    elif emotion_type == "emotion":
        # 对于情绪表达类，增加降温和留白的权重
        adjusted_weights["cool_down"] = 0.20
        adjusted_weights["ambiguous_leave"] = 0.20
    elif emotion_type == "daily":
        # 对于日常互动类，增加反问和轻主导的权重
        adjusted_weights["question_back"] = 0.20
        adjusted_weights["light_lead"] = 0.20
    elif emotion_type == "tease":
        # 对于调侃/打闘类，增加轻调侃和反问的权重
        adjusted_weights["light_tease"] = 0.30
        adjusted_weights["question_back"] = 0.20
    elif emotion_type == "short_positive":
        # 对于简短肯定/期待类，增加留白和反问的权重
        adjusted_weights["ambiguous_leave"] = 0.25
        adjusted_weights["question_back"] = 0.20
        adjusted_weights["incomplete"] = 0.20

    # 归一化权重
    total = sum(adjusted_weights.values())
    normalized = {k: v / total for k, v in adjusted_weights.items()}

    types = list(normalized.keys())
    weights = list(normalized.values())
    return random.choices(types, weights=weights, k=1)[0]


def process_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    处理单个样本，生成拉扯感 DPO 数据

    策略：
    1. 如果原回复是"直给"类型 -> 生成拉扯感的 chosen，原回复作为 rejected
    2. 如果原回复不是"直给"类型 -> 生成拉扯感的 chosen，再生成直给的 rejected
    """
    messages = sample.get("messages", [])
    original = sample.get("label", "")

    if not messages or not original:
        return None

    user_msg = get_last_user_message(messages)
    if not user_msg:
        return None

    # 检测是否有情绪触发
    emotion_type = detect_emotion_type(user_msg)
    if not emotion_type:
        # 没有明显的情绪触发，跳过
        return None

    # 选择拉扯类型
    tense_type = select_tense_type(user_msg)
    generator = TENSE_GENERATORS[tense_type]

    # 生成拉扯感回复作为 chosen
    chosen = generator(messages, original)
    if not chosen:
        return None

    # 判断 rejected 来源
    if is_direct_give(original):
        # 原回复已经是直给类型，直接用作 rejected
        rejected = original
    else:
        # 原回复不是直给类型，生成一个直给版本作为 rejected
        rejected = generate_direct_give_rejected(messages, chosen)
        if not rejected:
            # 如果生成失败，跳过
            return None

    # 确保 chosen 和 rejected 不同
    if chosen.strip() == rejected.strip():
        return None

    # 确保 chosen 不是直给类型
    if is_direct_give(chosen):
        return None

    return {
        "prompt": messages,
        "chosen": chosen,
        "rejected": rejected,
        "rejected_type": f"direct_give_vs_{tense_type}",
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
        if (i + 1) % 20 == 0:
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
            # 检查是否因为没有情绪触发而跳过
            user_msg = get_last_user_message(sample.get("messages", []))
            if not detect_emotion_type(user_msg):
                skip_count += 1
            else:
                fail_count += 1

        # 避免请求过快
        time.sleep(0.2)

    # 保存结果
    with open(dst_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   ✅ 完成: {success_count} 个拉扯感 DPO 样本")
    print(f"   📊 类型分布:")
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
    print("🎭 DPO 拉扯感数据生成脚本")
    print("=" * 60)
    print(f"源目录: {src_dir}")
    print(f"输出目录: {dst_dir}")
    print(f"找到 {len(jsonl_files)} 个文件")
    print(f"API: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print()
    print("🎯 拉扯感核心原则:")
    print("   - 不直给")
    print("   - 不过度承诺")
    print("   - 有轻微博弈")
    print("   - 有留白")
    print("   - 有一点掌控感")
    print("   - 但仍然温柔")
    print()
    print("📋 拉扯模式分布:")
    for tense_type, weight in TENSE_TYPE_WEIGHTS.items():
        print(f"   - {tense_type}: {weight * 100:.0f}%")

    total_samples = 0

    for src_path in jsonl_files:
        dst_path = dst_dir / f"tense_{src_path.stem}.jsonl"
        count = process_file(src_path, dst_path)
        total_samples += count

    print("\n" + "=" * 60)
    print(f"✅ 完成! 共生成 {total_samples} 个拉扯感 DPO 样本")
    print(f"   输出目录: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
