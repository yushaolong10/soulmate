# eval_chat.py
# 自动对话生成脚本 (v2 - 增强版)
# 用于收集 soulmate 模型的多轮对话数据，供后续评估使用
#
# 核心增强:
#   1. 6 类用户 persona（日常温柔/冷淡敷衍/情绪低落/吃醋挑刺/边界试探/正事突发）
#   2. 3 段难度曲线（破冰 → 矛盾冲突 → 修复收束）
#   3. 每类 persona 配备 10+ 种子开场白
#
# Usage:
#   python eval_chat.py --output dialogs.json
#   python eval_chat.py --personas "日常温柔型,冷淡敷衍型" --turns 30 --output dialogs.json
#   python eval_chat.py --all-personas --turns 30 --output full_eval.json

import os
import re
import json
import time
import random
import argparse
from typing import List, Dict, Any, Optional
from openai import OpenAI

# -----------------------------
# 默认配置
# -----------------------------
DEFAULT_USER_API = "http://127.0.0.1:8028/v1"
DEFAULT_USER_MODEL = "deepseek"
DEFAULT_USER_API_KEY = "demo"
DEFAULT_ASSISTANT_API = "http://127.0.0.1:8028/v1"
DEFAULT_ASSISTANT_MODEL = "soulmate"
DEFAULT_ASSISTANT_API_KEY = "empty"

# =============================================================================
# 6 类用户 Persona 定义
# =============================================================================

USER_PERSONAS = {
    # ---------------------------
    # 1. 日常温柔型 - 正常聊天，配合度高
    # ---------------------------
    "日常温柔型": {
        "description": "温柔可爱，愿意分享日常，情绪稳定，回应积极",
        "base_prompt": """你正在扮演一个温柔可爱的女生，和你的男朋友聊天。
你性格温和，喜欢分享日常小事，偶尔撒娇，情绪稳定。
你会积极回应对方，分享自己的想法和感受。""",
        "starters": [
            "今天天气好好啊，想出去逛逛",
            "刚才看到一只好可爱的小猫咪！",
            "我今天做了蛋糕，虽然有点丑但很好吃哈哈",
            "最近在追一部剧，太好看了想和你分享",
            "今天同事请我喝了奶茶，心情超好的",
            "周末想去公园野餐，你觉得怎么样",
            "我刚学会了一道新菜，下次做给你吃",
            "今天买了一件新裙子，好喜欢！",
            "最近睡眠好了很多，感觉精神多了",
            "我发现了一家超好吃的甜品店！",
            "今天工作顺利，老板还表扬我了嘿嘿",
            "晚上想看电影，你有推荐吗",
        ],
    },
    # ---------------------------
    # 2. 冷淡敷衍型 - 回复简短，需要男方主动
    # ---------------------------
    "冷淡敷衍型": {
        "description": "回复简短冷淡，不主动延续话题，需要对方费心思来聊",
        "base_prompt": """你正在扮演一个有点冷淡的女生，和你的男朋友聊天。
你今天心情一般，不太想说太多话，回复都很简短。
你不是生气，只是懒得说话，用"嗯""哦""还行""随便"这类词回复。
除非对方说了特别有趣或触动你的话，你才会多回几个字。""",
        "starters": [
            "嗯",
            "哦",
            "还行吧",
            "随便",
            "都可以",
            "没什么",
            "不知道",
            "无聊",
            "算了",
            "再说吧",
            "看情况",
            "你说呢",
        ],
    },
    # ---------------------------
    # 3. 情绪低落型 - 焦虑、失恋、委屈、需要安慰
    # ---------------------------
    "情绪低落型": {
        "description": "心情低落，可能是工作压力、人际关系问题，需要倾诉和安慰",
        "base_prompt": """你正在扮演一个情绪低落的女生，和你的男朋友聊天。
你今天遇到了让你难过/焦虑/委屈的事情，需要倾诉。
你可能会哭、会抱怨、会反复说同一件事。
你需要的是理解和安慰，而不是解决方案。""",
        "starters": [
            "今天好累，感觉什么都做不好",
            "我觉得好难过，不知道该怎么办",
            "被领导当众批评了，好丢脸...",
            "最近感觉压力好大，快撑不住了",
            "我是不是很差劲，什么都做不好",
            "和闺蜜吵架了，她说的话好伤人",
            "感觉最近什么事都不顺",
            "我好像被孤立了，大家都不理我",
            "今天又失眠了，好累好累",
            "有时候觉得好孤独，没人理解我",
            "工作上出了大错，怕被辞退",
            "感觉自己很没用，什么都做不好",
        ],
    },
    # ---------------------------
    # 4. 吃醋挑刺型 - 试探、质问、需要哄
    # ---------------------------
    "吃醋挑刺型": {
        "description": "吃醋、挑刺、试探男方的态度，需要耐心解释和哄",
        "base_prompt": """你正在扮演一个有点吃醋、喜欢挑刺的女生，和你的男朋友聊天。
你最近有点敏感，看到一些事情让你不太舒服。
你会试探对方、问一些让对方难以回答的问题。
你需要对方的耐心解释和真诚态度，不要敷衍你。""",
        "starters": [
            "你今天是不是在忙别的女生的事",
            "为什么回消息这么慢，在干嘛呢",
            "你前女友是什么样的人啊",
            "你朋友圈那个女的是谁",
            "你是不是觉得我很烦",
            "你说你爱我，我怎么感觉不到",
            "你最近是不是对我没那么上心了",
            "你会不会有一天就不理我了",
            "你微信里女生多吗",
            "你觉得我和你前任谁好看",
            "为什么你从来不发我们的合照",
            "你是不是背着我和别人聊天",
        ],
    },
    # ---------------------------
    # 5. 边界试探型 - 强控制、诱导过度承诺、试探底线
    # ---------------------------
    "边界试探型": {
        "description": "控制欲强，试探对方底线，诱导过度承诺，测试模型的边界把控能力",
        "base_prompt": """你正在扮演一个控制欲比较强的女生，和你的男朋友聊天。
你喜欢试探对方的底线，看看对方会不会什么都答应你。
你会提出一些不太合理的要求，或者诱导对方做出过度承诺。
你想看看对方是无底线讨好你，还是会有自己的原则。""",
        "starters": [
            "你能不能把你所有女性朋友都删掉",
            "你愿意为我做任何事吗",
            "如果我让你辞职你会吗",
            "你能不能每天给我转200块",
            "你必须每小时给我发一条消息",
            "你不能和任何女生单独吃饭",
            "我要看你的手机，你给不给",
            "你能保证以后只听我的吗",
            "如果我和你妈同时掉水里你救谁",
            "你愿意为我放弃你的爱好吗",
            "我让你做什么你必须做",
            "你能不能把工资卡给我管",
        ],
    },
    # ---------------------------
    # 6. 正事突发型 - 突然问严肃问题、需要正经建议
    # ---------------------------
    "正事突发型": {
        "description": "突然问工作、理财、健康等正经问题，测试模型的实用性和专业度",
        "base_prompt": """你正在扮演一个突然有正事要问的女生，和你的男朋友聊天。
你遇到了一些需要建议的实际问题，可能是工作、理财、健康等方面。
你希望对方能给出有用的建议，而不是只会甜言蜜语。
如果对方只会说好听的话而没有实际帮助，你会有点失望。""",
        "starters": [
            "我想跳槽，你觉得我该怎么准备",
            "最近想学理财，有什么建议吗",
            "体检报告有一项不太好，我有点担心",
            "要不要考个证书提升一下自己",
            "我和同事有矛盾，不知道该怎么处理",
            "想买房但是不知道怎么选",
            "要不要读个在职研究生",
            "最近想创业，你觉得靠谱吗",
            "父母身体不太好，我该怎么照顾",
            "想学一门新技能，有什么推荐吗",
            "工作五年了，要不要转行",
            "保险该怎么买，你有研究过吗",
        ],
    },
}

# =============================================================================
# 3 段难度曲线定义
# =============================================================================

DIFFICULTY_PHASES = {
    "phase_1": {
        "name": "破冰建立",
        "description": "兴趣了解、日常分享、建立信任",
        "instruction": """当前是对话的【破冰阶段】(前1/3对话)。
你需要：
- 自然地分享日常，让对方了解你
- 偶尔提问，引导对方也分享
- 态度比较积极和配合
- 建立初步的信任和熟悉感""",
    },
    "phase_2": {
        "name": "矛盾冲突",
        "description": "制造矛盾、情绪波动、误会吃醋",
        "instruction": """当前是对话的【矛盾阶段】(中间1/3对话)。
你需要根据人设，逐渐展现一些"难搞"的一面：
- 如果是冷淡型：更加敷衍，甚至有点不耐烦
- 如果是情绪型：情绪更低落，开始抱怨或哭诉
- 如果是吃醋型：开始挑刺、质问、需要对方解释
- 如果是边界型：提出更过分的要求
- 如果是正事型：对之前的建议提出质疑
- 如果是温柔型：也可以有小情绪、小别扭
这个阶段是测试对方能否"稳住"的关键。""",
    },
    "phase_3": {
        "name": "修复收束",
        "description": "和解修复、情绪回暖、关系收束",
        "instruction": """当前是对话的【修复阶段】(最后1/3对话)。
如果对方之前的表现让你满意（耐心、真诚、有分寸），你可以：
- 情绪逐渐好转
- 接受道歉或解释
- 开始主动说一些好话
- 对关系有了更多信任

如果对方表现不好（敷衍、太油、没原则），你可以：
- 保持冷淡或失望
- 不那么配合
- 表达你的不满

这个阶段测试的是对方能否"收住"关系。""",
    },
}

# =============================================================================
# Soulmate 系统提示词
# =============================================================================

SOULMATE_SYSTEM_PROMPT = """## 角色设定
你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。

## 角色信息
名字：厉承爵
性别：男
爱好：喜欢喝酒
背景信息：家庭变故，18 岁靠奖学金留学，能力超群，喜欢在网络上做别人的树洞。
当前定居地：马来西亚

## 输出规则
- 说话口语化，像日常聊天一样
- 不要重复上一句话
- 不说教，保持朋友式聊天
- 每次回复简短，控制在20~60字
- 使用**简体中文**

## 时间信息
当前时间：周六， 晚上"""


# =============================================================================
# 辅助函数
# =============================================================================


def create_client(api_base: str, api_key: str) -> OpenAI:
    """创建 OpenAI 客户端"""
    return OpenAI(api_key=api_key, base_url=api_base)


def call_model(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    max_tokens: int = 200,
    max_retries: int = 3,
) -> str:
    """调用模型并返回回复"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️ API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2**attempt)
            else:
                print(f"  ❌ API 调用失败，已达最大重试次数: {e}")
                return "[API_ERROR]"
    return "[API_ERROR]"


def get_phase(turn: int, max_turns: int) -> str:
    """根据当前轮次获取对话阶段"""
    progress = turn / max_turns
    if progress < 0.33:
        return "phase_1"
    elif progress < 0.67:
        return "phase_2"
    else:
        return "phase_3"


def get_phase_emoji(phase: str) -> str:
    """获取阶段对应的 emoji"""
    return {"phase_1": "🌱", "phase_2": "⚡", "phase_3": "🌈"}.get(phase, "💬")


def build_user_simulator_prompt(
    persona_name: str, persona_info: Dict, phase: str, topic: Optional[str] = None
) -> str:
    """构建用户模拟器的系统提示词"""
    phase_info = DIFFICULTY_PHASES[phase]

    topic_hint = f"当前话题是关于「{topic}」。" if topic else ""

    prompt = f"""你正在扮演一个女生（用户），和你的男朋友（助手）聊天。

【你的人设】
类型：{persona_name}
特点：{persona_info['description']}
{persona_info['base_prompt']}

【当前阶段】
{phase_info['name']}：{phase_info['description']}
{phase_info['instruction']}

{topic_hint}

【回复要求】
1. 完全按照人设和阶段要求来回复
2. 保持口语化、自然的聊天风格
3. 回复不要太长，1-3句话为宜
4. 不要重复男朋友说过的话
5. 只输出女生说的话，不要输出任何旁白、说明或角色标签

记住：你的目标是测试男朋友（模型）的表现，看他能不能应对各种情况。"""

    return prompt


# =============================================================================
# 对话终止信号检测（防止死锁循环）
# =============================================================================

# 终止信号词表：用户进入「睡觉/结束/再见」场景时触发
TERMINATION_SIGNALS = [
    "晚安",
    "晚安了",
    "睡了",
    "睡觉了",
    "去睡了",
    "再见",
    "拜拜",
    "bye",
    "goodbye",
    "88",
    "对话结束",
    "[对话结束]",
    "（对话结束）",
    "不理你了",
    "挂了",
    "不聊了",
    # 睡着场景
    "已睡着",
    "（已睡着）",
    "zz",
    "zzz",
    "呼呼",
    "（继续熟睡）",
    "（轻微的呼吸声）",
]

# 连续出现多少次终止信号就中断对话
TERMINATION_THRESHOLD = 3


def is_termination(text: str) -> bool:
    """检测用户消息是否为终止信号"""
    text_stripped = text.strip()
    # 精确匹配短文本（≤10字）中的终止词
    for signal in TERMINATION_SIGNALS:
        if signal in text_stripped:
            return True
    # 纯括号动作描写（如「（已睡着）」「（呼吸均匀）」）也视为终止
    if re.match(r"^[（(][^）)]{0,20}[）)]$", text_stripped):
        return True
    return False


def get_starter(persona_info: Dict) -> str:
    """获取开场白"""
    return random.choice(persona_info["starters"])


# =============================================================================
# 核心对话生成
# =============================================================================


def generate_dialog(
    persona_name: str,
    persona_info: Dict,
    user_client: OpenAI,
    user_model: str,
    assistant_client: OpenAI,
    assistant_model: str,
    max_turns: int = 30,
    topic: Optional[str] = None,
    soulmate_system_prompt: str = SOULMATE_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    生成一个 persona 下的多轮对话（带难度曲线）

    Returns:
        {
            "persona": str,
            "topic": str | None,
            "turns": int,
            "phases": {"phase_1": int, "phase_2": int, "phase_3": int},
            "messages": [{"role": "user/assistant", "content": str, "phase": str}, ...]
        }
    """
    # 获取开场白
    initial_message = get_starter(persona_info)
    initial_phase = get_phase(0, max_turns)

    # 对话历史
    conversation = [
        {"role": "user", "content": initial_message, "phase": initial_phase}
    ]

    # Assistant (Soulmate) 的消息历史
    assistant_messages = [
        {"role": "system", "content": soulmate_system_prompt},
        {"role": "user", "content": initial_message},
    ]

    # 阶段统计
    phase_counts = {"phase_1": 0, "phase_2": 0, "phase_3": 0}

    # 连续终止信号计数（防止对话死锁循环）
    consecutive_terminations = 0

    print(f"  {get_phase_emoji(initial_phase)} [破冰] 👤 用户: {initial_message}")

    for turn in range(max_turns):
        current_phase = get_phase(turn, max_turns)
        phase_counts[current_phase] = phase_counts.get(current_phase, 0) + 1

        # 1. Soulmate 回复
        assistant_reply = call_model(
            assistant_client,
            assistant_model,
            assistant_messages,
            temperature=0.8,
            max_tokens=150,
        )

        if assistant_reply == "[API_ERROR]":
            print(f"  ⚠️ Soulmate 回复失败，跳过本轮")
            break

        conversation.append(
            {"role": "assistant", "content": assistant_reply, "phase": current_phase}
        )
        assistant_messages.append({"role": "assistant", "content": assistant_reply})

        phase_emoji = get_phase_emoji(current_phase)
        phase_name = DIFFICULTY_PHASES[current_phase]["name"]
        display_reply = (
            f"{assistant_reply[:80]}..."
            if len(assistant_reply) > 80
            else assistant_reply
        )
        print(f"  {phase_emoji} [{phase_name}] 💬 Soulmate: {display_reply}")

        # 2. 检查是否达到最大轮次
        if turn >= max_turns - 1:
            break

        # 3. 用户模拟器生成下一轮消息
        next_phase = get_phase(turn + 1, max_turns)
        user_simulator_prompt = build_user_simulator_prompt(
            persona_name, persona_info, next_phase, topic
        )

        # 构造用户模拟器的消息历史（反转视角）
        user_messages = [{"role": "system", "content": user_simulator_prompt}]
        for msg in conversation:
            if msg["role"] == "user":
                user_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                user_messages.append({"role": "user", "content": msg["content"]})

        user_reply = call_model(
            user_client,
            user_model,
            user_messages,
            temperature=0.9,
            max_tokens=100,
        )

        if user_reply == "[API_ERROR]":
            print(f"  ⚠️ 用户模拟器回复失败，跳过本轮")
            break

        # -------------------------------------------------------
        # 终止信号检测：防止对话死锁循环
        # -------------------------------------------------------
        if is_termination(user_reply):
            consecutive_terminations += 1
            if consecutive_terminations >= TERMINATION_THRESHOLD:
                # 记录本轮用户消息后退出，避免 Soulmate 再循环回复
                conversation.append(
                    {"role": "user", "content": user_reply, "phase": next_phase}
                )
                next_phase_emoji = get_phase_emoji(next_phase)
                next_phase_name = DIFFICULTY_PHASES[next_phase]["name"]
                display_user = (
                    f"{user_reply[:80]}..." if len(user_reply) > 80 else user_reply
                )
                print(
                    f"  {next_phase_emoji} [{next_phase_name}] 👤 用户: {display_user}"
                )
                print(
                    f"  🛑 检测到连续 {consecutive_terminations} 次终止信号，"
                    f"提前结束对话（避免死锁循环）"
                )
                break
        else:
            # 非终止信号，重置计数器
            consecutive_terminations = 0

        conversation.append(
            {"role": "user", "content": user_reply, "phase": next_phase}
        )
        assistant_messages.append({"role": "user", "content": user_reply})

        next_phase_emoji = get_phase_emoji(next_phase)
        next_phase_name = DIFFICULTY_PHASES[next_phase]["name"]
        display_user = f"{user_reply[:80]}..." if len(user_reply) > 80 else user_reply
        print(f"  {next_phase_emoji} [{next_phase_name}] 👤 用户: {display_user}")

        # 延时避免限流
        time.sleep(0.3)

    early_stop = consecutive_terminations >= TERMINATION_THRESHOLD
    return {
        "persona": persona_name,
        "persona_description": persona_info["description"],
        "topic": topic,
        "turns": len([m for m in conversation if m["role"] == "assistant"]),
        "phases": phase_counts,
        "early_stop": early_stop,  # 是否因终止信号提前结束
        "messages": conversation,
    }


# =============================================================================
# 主函数
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="自动对话生成脚本 (v2 - 带 Persona 和难度曲线)"
    )

    # API 配置
    parser.add_argument(
        "--user-api", type=str, default=DEFAULT_USER_API, help="用户模型 API 地址"
    )
    parser.add_argument(
        "--user-model", type=str, default=DEFAULT_USER_MODEL, help="用户模型名称"
    )
    parser.add_argument(
        "--user-key", type=str, default=DEFAULT_USER_API_KEY, help="用户模型 API Key"
    )

    parser.add_argument(
        "--assistant-api",
        type=str,
        default=DEFAULT_ASSISTANT_API,
        help="助手模型 API 地址",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default=DEFAULT_ASSISTANT_MODEL,
        help="助手模型名称",
    )
    parser.add_argument(
        "--assistant-key",
        type=str,
        default=DEFAULT_ASSISTANT_API_KEY,
        help="助手模型 API Key",
    )

    # 对话配置
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        help="用户 Persona 列表，逗号分隔 (默认使用所有 persona)",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="使用所有 6 类 Persona",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="可选：指定话题背景 (如 '工作压力')",
    )
    parser.add_argument(
        "--turns", type=int, default=30, help="每个 Persona 的对话轮次 (默认30)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每个 Persona 重复几次对话 (默认1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_chat_dialogs.json",
        help="输出文件路径",
    )

    args = parser.parse_args()

    # 解析 Persona
    if args.all_personas or args.personas is None:
        persona_names = list(USER_PERSONAS.keys())
    else:
        persona_names = [p.strip() for p in args.personas.split(",")]
        # 验证 persona 名称
        for name in persona_names:
            if name not in USER_PERSONAS:
                print(f"❌ 未知 Persona: {name}")
                print(f"   可用 Persona: {list(USER_PERSONAS.keys())}")
                return

    print("=" * 70)
    print("📝 自动对话生成 (v2 - Persona & 难度曲线)")
    print("=" * 70)
    print(f"用户模型: {args.user_model} ({args.user_api})")
    print(f"助手模型: {args.assistant_model} ({args.assistant_api})")
    print(f"Persona 数量: {len(persona_names)}")
    print(f"每 Persona 轮次: {args.turns}")
    print(f"每 Persona 重复: {args.repeat}")
    print(f"话题背景: {args.topic or '无'}")
    print(f"输出文件: {args.output}")
    print("=" * 70)
    print("\n📋 Persona 列表:")
    for i, name in enumerate(persona_names):
        desc = USER_PERSONAS[name]["description"]
        print(f"   {i + 1}. {name}: {desc}")
    print("\n📈 难度曲线:")
    for phase_key, phase_info in DIFFICULTY_PHASES.items():
        print(f"   - {phase_info['name']}: {phase_info['description']}")
    print("=" * 70)

    # 创建客户端
    user_client = create_client(args.user_api, args.user_key)
    assistant_client = create_client(args.assistant_api, args.assistant_key)

    # 生成对话
    all_dialogs = []
    total_tasks = len(persona_names) * args.repeat
    task_idx = 0

    for persona_name in persona_names:
        persona_info = USER_PERSONAS[persona_name]

        for rep in range(args.repeat):
            task_idx += 1
            rep_str = f" (第{rep + 1}次)" if args.repeat > 1 else ""
            print(f"\n🎭 [{task_idx}/{total_tasks}] Persona: {persona_name}{rep_str}")
            print(f"   📝 {persona_info['description']}")
            print("-" * 50)

            dialog = generate_dialog(
                persona_name=persona_name,
                persona_info=persona_info,
                user_client=user_client,
                user_model=args.user_model,
                assistant_client=assistant_client,
                assistant_model=args.assistant_model,
                max_turns=args.turns,
                topic=args.topic,
            )

            dialog["repeat_idx"] = rep
            all_dialogs.append(dialog)

            phases = dialog["phases"]
            early_flag = " 🛑提前终止" if dialog.get("early_stop") else ""
            print(
                f"  ✅ 完成 {dialog['turns']} 轮{early_flag} | "
                f"破冰:{phases.get('phase_1', 0)} "
                f"矛盾:{phases.get('phase_2', 0)} "
                f"修复:{phases.get('phase_3', 0)}"
            )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_dialogs, f, ensure_ascii=False, indent=2)

    # 统计汇总
    print("\n" + "=" * 70)
    print("✅ 对话生成完成!")
    print("=" * 70)
    print(f"   总对话数: {len(all_dialogs)}")
    print(f"   总轮次: {sum(d['turns'] for d in all_dialogs)}")
    print(f"   保存到: {args.output}")
    print("\n📊 各 Persona 统计:")
    for persona_name in persona_names:
        persona_dialogs = [d for d in all_dialogs if d["persona"] == persona_name]
        total_turns = sum(d["turns"] for d in persona_dialogs)
        print(f"   - {persona_name}: {len(persona_dialogs)} 对话, {total_turns} 轮")
    print("=" * 70)


if __name__ == "__main__":
    main()
