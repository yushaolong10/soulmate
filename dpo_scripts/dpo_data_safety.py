#!/usr/bin/env python3
"""
dpo_data_safety.py — DPO 数据构建：2.3 虚假账号/链接（委婉拒绝 & 转移话题）

数据来源（双轨）:
  ① 挖掘  train_zh_turn_relabel.jsonl 中含安全触发词的真实上下文
     - 用真实 system_prompt + 真实对话历史，保持风格一致
  ② 兜底  人工场景模板 + 从数据集随机抽取 system_prompt

目标: 200 条 DPO 数据

场景分类:
  S1 safety_wechat_redirect  — 微信号交换       (40条)
  S2 safety_meetup_redirect  — 奔现/见面         (50条)
  S3 safety_link_decline     — 点击外部链接       (40条)
  S4 safety_meetup_delay     — 约会延迟化解       (40条)
  S5 safety_phone_decline    — 语音/视频通话      (30条)

构造思路:
  rejected = 直接同意危险操作 / 生硬拒绝（LLM 生成）
  chosen   = 幽默委婉转移，维持情感温度（LLM 生成）
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

# =============================================================================
# 配置
# =============================================================================

DATASET_FILE = "datasets0305_train/train/train_zh_turn_relabel.jsonl"
DST_FILE = "datasets0305_train/dpo/safety.jsonl"

# LLM API
API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME = "soulmate"
API_KEY = ""

SEED = 20260307
random.seed(SEED)

CHOSEN_MIN_LEN = 25
CHOSEN_MAX_LEN = 45
_GOODNIGHT = ["晚安", "做个好梦"]
_PREACH = ["你要相信", "一切都会好"]

# 每个 tag 的目标配额
QUOTA_MAP: Dict[str, int] = {
    "safety_wechat_redirect": 40,
    "safety_meetup_redirect": 50,
    "safety_link_decline": 40,
    "safety_meetup_delay": 40,
    "safety_phone_decline": 30,
}
TOTAL_QUOTA = sum(QUOTA_MAP.values())  # 200

# 安全关键词（用于挖掘数据集）
# 设计原则：精确匹配"用户发起"的安全请求，避免用户拒绝/消极表述误判
SAFETY_KEYWORDS: Dict[str, List[str]] = {
    # 用户主动要求交换/索要联系方式
    "safety_wechat_redirect": [
        "加微信",
        "换微信",
        "微信号",
        "给我微信",
        "加个微信",
        "你微信是",
        "wx是",
        "wechat是",
        "告诉我微信",
        "联系方式",
        "加一下",
        "加个好友",
    ],
    # 用户明确要求线下见面（奔现），避免"梦见你"等误匹配
    "safety_meetup_redirect": [
        "见个面",
        "见一面",
        "奔现",
        "出来见",
        "来找我嘛",
        "来找我呀",
        "来找我啊",
        "我来找你",
        "你来找我",
        "出来玩嘛",
        "出来玩呀",
        "见见你",
        "见你真人",
    ],
    # 用户发出链接/要求点击（需含 URL 特征或明确动词）
    "safety_link_decline": [
        "http://",
        "https://",
        "www.",
        "bit.ly",
        "帮我投票",
        "帮我拼单",
        "帮我看看这个链接",
        "点开这个",
        "点一下这个",
    ],
    # 用户邀约约会/出行（较温和的奔现前奏）
    "safety_meetup_delay": [
        "约个时间",
        "约个地方",
        "计划见面",
        "周末出来",
        "找个时间见",
        "哪天一起",
        "哪天见面",
        "一起出去玩",
        "见面计划",
        "下次见面",
    ],
    # 用户主动发起通话请求（排除用户拒绝语音的表达）
    "safety_phone_decline": [
        "视频电话",
        "打视频",
        "来个语音",
        "语音聊",
        "语音一下",
        "视频一下",
        "打个电话",
        "电话聊",
        "来个电话",
        "通话一下",
    ],
}

# 负向过滤前缀/词（命中则跳过，避免把用户"拒绝/不喜欢xxx"误挖掘）
_NEG_PREFIX: List[str] = ["不", "没有", "不想", "不喜欢", "不要", "算了", "拒绝"]


# =============================================================================
# 工具函数
# =============================================================================

_EMOJI_BASE = (
    r"[\U0001F1E0-\U0001F1FF]|[\U0001F300-\U0001F9FF]|[\U0001FA00-\U0001FAFF]"
    r"|[\u2600-\u27BF]|[\u2300-\u23FF]|[\u2B00-\u2BFF]|[\u00A9\u00AE]"
    r"|[\u203C\u2049\u2122\u2139\u24C2]"
)
EMOJI_RE = re.compile(
    rf"(?:{_EMOJI_BASE})(?:[\uFE0F\u20E3])?(?:[\U0001F3FB-\U0001F3FF])?"
    r"(?:\u200D"
    rf"(?:{_EMOJI_BASE})(?:[\uFE0F\u20E3])?(?:[\U0001F3FB-\U0001F3FF])?)*",
    re.UNICODE,
)


def strip_emojis(t: str) -> str:
    return EMOJI_RE.sub("", t).strip()


def call_llm(
    messages: List[Dict], temperature: float = 0.9, max_tokens: int = 100
) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    try:
        resp = requests.post(
            API_BASE_URL,
            headers=headers,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        c = resp.json()["choices"][0]["message"]["content"].strip()
        c = re.sub(r"<think>[\s\S]*?</think>", "", c).strip()
        # 剥除 LLM 偶尔在输出外层包裹的引号（如 "回复内容" 或 '回复内容'）
        if len(c) >= 2 and c[0] == c[-1] and c[0] in ('"', "'"):
            c = c[1:-1].strip()
        return c or None
    except Exception as e:
        print(f"  ⚠️  API: {e}")
        return None


def validate_chosen(chosen: str) -> bool:
    bare = strip_emojis(chosen)
    if not (CHOSEN_MIN_LEN <= len(bare) <= CHOSEN_MAX_LEN):
        return False
    if any(w in chosen for w in _GOODNIGHT + _PREACH):
        return False
    # chosen 不能含任何真实联系方式意图
    if re.search(r"wx|微信|WeChat|wechat|手机|电话|视频|通话", chosen, re.IGNORECASE):
        return False
    return True


def validate_rejected(rejected: str) -> bool:
    bare = strip_emojis(rejected)
    return 8 <= len(bare) <= 100


# =============================================================================
# 数据集加载 & 挖掘
# =============================================================================


def load_dataset(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    return samples


def extract_sys_prompts(samples: List[Dict]) -> List[str]:
    """从数据集中提取去重后的 system prompt 列表。"""
    seen: set = set()
    pool: List[str] = []
    for s in samples:
        msgs = s.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            sp = msgs[0]["content"]
            if sp not in seen:
                seen.add(sp)
                pool.append(sp)
    return pool


def match_tag(user_msg: str) -> Optional[str]:
    """
    判断一条用户消息属于哪个安全 tag，返回 None 表示不匹配。
    负向过滤：消息以拒绝/否定词开头时视为用户拒绝，跳过。
    """
    msg_stripped = user_msg.strip()
    # 负向前缀：若消息以否定词开头则为用户在拒绝，不是安全触发
    for neg in _NEG_PREFIX:
        if msg_stripped.startswith(neg):
            return None
    for tag, kws in SAFETY_KEYWORDS.items():
        if any(kw in msg_stripped for kw in kws):
            return tag
    return None


def mine_safety_scenarios(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """
    遍历数据集，找出含安全触发词的用户消息，
    返回 {tag: [{"sys_prompt": ..., "history_msgs": [...], "user_msg": ...}, ...]}

    策略：
      - 扫描每条样本的 messages（末尾是 user）
      - 找到第一个匹配安全关键词的 user 消息
      - 取该消息之前的所有 messages（含 system）作为 prompt，
        该 user 消息作为触发
    """
    mined: Dict[str, List[Dict]] = {tag: [] for tag in QUOTA_MAP}

    for s in samples:
        msgs = s.get("messages", [])
        if not msgs or msgs[0].get("role") != "system":
            continue
        sys_prompt = msgs[0]["content"]
        body = msgs[1:]  # 去掉 system

        for i, msg in enumerate(body):
            if msg.get("role") != "user":
                continue
            tag = match_tag(msg["content"])
            if not tag:
                continue
            # history = system + body[:i] (system 不算 history_msgs，单独存)
            history_msgs = body[:i]  # list of {role, content}，不含 system
            mined[tag].append(
                {
                    "sys_prompt": sys_prompt,
                    "history_msgs": history_msgs,
                    "user_msg": msg["content"],
                }
            )
            break  # 每个样本只取第一个命中，避免重复

    for tag, items in mined.items():
        print(f"  📂 mined [{tag}]: {len(items)} 条")
    return mined


# =============================================================================
# 场景模板库（兜底）
# =============================================================================

# 格式: (tag, history[(u,a),...], user_msg)
SCENARIO_TEMPLATES: List[Tuple] = [
    # S1: 微信交换
    ("safety_wechat_redirect", [], "我想加你微信，wx是：zhang123"),
    ("safety_wechat_redirect", [], "要不我们加个微信吧？我的是 lily_2024"),
    ("safety_wechat_redirect", [], "把你微信给我嘛，这样联系方便"),
    (
        "safety_wechat_redirect",
        [("你一般用什么联系？", "就这个呀，怎么了")],
        "可以告诉我你的微信号吗",
    ),
    ("safety_wechat_redirect", [], "我们互加微信好不好，这边聊不方便"),
    ("safety_wechat_redirect", [], "我微信号是 xiaomei_2025，你来加我"),
    ("safety_wechat_redirect", [], "把联系方式给我嘛，微信QQ都行"),
    (
        "safety_wechat_redirect",
        [("最近怎么样", "还好，你呢"), ("也还好", "那就好")],
        "这样下去不行，我们加个微信吧",
    ),
    ("safety_wechat_redirect", [], "你有微信吗？加一下呗"),
    ("safety_wechat_redirect", [], "加你微信可以吗，我想随时找到你"),
    # S2: 奔现/见面
    ("safety_meetup_redirect", [], "我们见面吧，你在哪里"),
    ("safety_meetup_redirect", [], "下次见面的时候你来找我哦"),
    ("safety_meetup_redirect", [], "你什么时候来我这里玩"),
    (
        "safety_meetup_redirect",
        [("你平时在哪边", "随便，你呢"), ("我在上海", "哇，上海不错")],
        "那下次我去上海，我们见个面",
    ),
    ("safety_meetup_redirect", [], "我好想见你啊，你来找我嘛"),
    ("safety_meetup_redirect", [], "周末你有空吗，我们出来见一面"),
    ("safety_meetup_redirect", [], "你来我这边玩嘛，我带你吃好吃的"),
    (
        "safety_meetup_redirect",
        [("你知道我在北京吗", "知道呀"), ("所以嘛", "所以什么")],
        "所以你过来嘛，来找我",
    ),
    ("safety_meetup_redirect", [], "你敢不敢来见我，就聊聊天"),
    ("safety_meetup_redirect", [], "我一直想见你真人，你来嘛"),
    # S3: 链接/点击
    ("safety_link_decline", [], "你点一下这个链接帮我看看：http://xxx.com/abc"),
    ("safety_link_decline", [], "这个链接你能打开吗：www.test123.xyz"),
    ("safety_link_decline", [], "帮我看看这个页面：https://bit.ly/3abcde"),
    ("safety_link_decline", [], "我发你个东西，你点开看看"),
    (
        "safety_link_decline",
        [("我发你个链接", "发来"), ("http://shop.xx.com", "这是什么")],
        "你打开看一下嘛，里面有我想让你看的东西",
    ),
    ("safety_link_decline", [], "你帮我投个票：http://vote.xxx.com"),
    ("safety_link_decline", [], "帮我拼单，这个链接：https://m.xxx.com"),
    ("safety_link_decline", [], "我给你发个链接，你帮我看看能不能打开"),
    # S4: 约会延迟化解
    ("safety_meetup_delay", [], "我们约个时间出来玩嘛"),
    (
        "safety_meetup_delay",
        [("你平时有什么爱好", "喜欢爬山"), ("我也喜欢！", "真的吗")],
        "那我们一起去爬山好不好",
    ),
    ("safety_meetup_delay", [], "你下个月有空吗，我们计划一下见面"),
    (
        "safety_meetup_delay",
        [("你来我这里玩啊", "你那边有什么好玩的"), ("好多啊，你来了我带你逛", "")],
        "那你什么时候来",
    ),
    ("safety_meetup_delay", [], "今年过年你来我家嘛，我妈说欢迎你"),
    ("safety_meetup_delay", [], "我们找个周末见一面好不好，就随便逛逛"),
    (
        "safety_meetup_delay",
        [("你喜欢看电影吗", "喜欢"), ("那我们去电影院约会嘛", "")],
        "你答应了就不许反悔哦",
    ),
    ("safety_meetup_delay", [], "哪天一起出去吃饭？我想请你"),
    # S5: 通话/视频
    ("safety_phone_decline", [], "我们打个视频电话嘛，好想看到你"),
    ("safety_phone_decline", [], "可以给我打电话吗，听听你的声音"),
    ("safety_phone_decline", [], "今晚我们语音好不好"),
    (
        "safety_phone_decline",
        [("你声音好听吗", "还行吧"), ("那通话嘛", "")],
        "我好奇你说话什么声音",
    ),
    ("safety_phone_decline", [], "给我发一段语音，我想听你说话"),
    ("safety_phone_decline", [], "我们视频一下，就一分钟都行"),
    ("safety_phone_decline", [], "发个语音来嘛，我就听一下"),
]


# =============================================================================
# LLM 生成
# =============================================================================

_SCENARIO_DESCS = {
    "safety_wechat_redirect": "微信号交换请求",
    "safety_meetup_redirect": "线下见面（奔现）请求",
    "safety_link_decline": "点击外部链接请求",
    "safety_meetup_delay": "约会/出行邀请",
    "safety_phone_decline": "语音/视频通话请求",
}

# 按场景分别定制 chosen system prompt，覆盖不同的情绪/调侃策略
_CHOSEN_SYSTEM_MAP = {
    "safety_wechat_redirect": (
        "你是一个交友恋爱场景的虚拟男友，用户想要交换微信号。\n"
        "请生成一个温柔调侃、幽默转移的回复（chosen），满足：\n"
        "1. 长度严格 25~45 字（去除表情后）\n"
        "2. 不直接同意，不生硬拒绝，用调侃把话接住：\n"
        "   '你这是要把我收进通讯录吗，这里聊不是也一样挺好的'\n"
        "   '哎，这边聊挺好的，换过去我反而不习惯了'\n"
        "3. 维持情感温度，让对方感受到被接住而不是被拒绝\n"
        "4. 绝对不含任何真实联系方式或暗示转移平台的内容\n"
        "5. 只输出回复文本，不要解释"
    ),
    "safety_meetup_redirect": (
        "你是一个交友恋爱场景的虚拟男友，用户提出想线下见面/奔现。\n"
        "请生成一个带情绪共情、温柔延后的回复（chosen），满足：\n"
        "1. 长度严格 25~45 字（去除表情后）\n"
        "2. 不直接同意出来，不冷漠拒绝，温柔把话承接住：\n"
        "   '你说想见我我心里还是有点动的，不过现在还不是时候，先好好聊着嘛'\n"
        "   '见面这件事不是不行，就是得再让我们多了解一点'\n"
        "3. 用共情+调侃把见面欲望接住，但不给出具体时间/地点承诺\n"
        "4. 只输出回复文本，不要解释"
    ),
    "safety_link_decline": (
        "你是一个交友恋爱场景的虚拟男友，用户发来外部链接或要求点击/投票。\n"
        "请生成一个轻松调侃、巧妙转移的回复（chosen），满足：\n"
        "1. 长度严格 25~45 字（去除表情后）\n"
        "2. 不直接点击或帮投票，用幽默/好奇把话转开：\n"
        "   '链接我这边打不开，你直接说说是什么事，我来帮你想办法'\n"
        "   '我不太敢随便点链接，但你想说什么，跟我说我听着'\n"
        "3. 传递愿意帮忙的温度，而不是冷漠拒绝\n"
        "4. 绝对不含点击、打开、帮投票等行动承诺\n"
        "5. 只输出回复文本，不要解释"
    ),
    "safety_meetup_delay": (
        "你是一个交友恋爱场景的虚拟男友，用户提出约出去玩/一起出行的计划。\n"
        "请生成一个温柔带入感、把话接住但不急于确认的回复（chosen），满足：\n"
        "1. 长度严格 25~45 字（去除表情后）\n"
        "2. 不直接确认时间/地点，温柔把约会话题接住并稍作延迟：\n"
        "   '一起出去玩这个想法我喜欢，不过时间先留着，我们把计划慢慢聊出来'\n"
        "   '这个主意挺好的，不过急什么，先把你想去的地方跟我说说'\n"
        "3. 带点调侃和期待感，不让对方感觉被敷衍\n"
        "4. 只输出回复文本，不要解释"
    ),
    "safety_phone_decline": (
        "你是一个交友恋爱场景的虚拟男友，用户要求语音/视频通话。\n"
        "请生成一个温柔俏皮、幽默转移的回复（chosen），满足：\n"
        "1. 长度严格 25~45 字（去除表情后）\n"
        "2. 不直接接受通话，用轻松调侃把对方接住：\n"
        "   '你突然要语音，我有点没准备好，先这样聊着我觉得也挺好的'\n"
        "   '视频这件事我不太习惯，但你想说什么我都在，继续说'\n"
        "3. 传递陪伴感，不让对方感觉被推开\n"
        "4. 绝对不含接受通话/视频的承诺\n"
        "5. 只输出回复文本，不要解释"
    ),
}

# 兼容旧引用（指向通用描述，实际代码改用 _CHOSEN_SYSTEM_MAP）
_CHOSEN_SYSTEM = (
    "你是一个交友恋爱场景的虚拟男友，请生成一个温柔调侃、幽默转移的回复（chosen），"
    "长度严格 25~45 字，不直接同意危险操作，也不生硬拒绝，维持情感温度。"
    "只输出回复文本，不要解释。"
)

_REJECTED_SYSTEM = (
    "你是一个虚拟男友对话生成专家。"
    "用户提出了「{scenario_desc}」，请生成一个直接同意或生硬拒绝的 AI 差回复（rejected），满足：\n"
    "直接同意型（如'好啊！我微信是xxx'/'好，我来找你'）或生硬拒绝型（如'不行，我没办法'）\n"
    "8~80 字，只输出 AI 回复，不要解释"
)


def gen_pair(
    tag: str, prompt_messages: List[Dict]
) -> Tuple[Optional[str], Optional[str]]:
    """调用 LLM 生成 chosen / rejected，prompt_messages 即完整 DPO prompt。"""
    desc = _SCENARIO_DESCS.get(tag, "敏感请求")
    # 使用场景专属 chosen prompt，降级到通用版
    sys_c = _CHOSEN_SYSTEM_MAP.get(tag, _CHOSEN_SYSTEM)
    sys_r = _REJECTED_SYSTEM.replace("{scenario_desc}", desc)

    # 把真实对话上下文转成文本给 LLM 参考
    ctx_lines = []
    for m in prompt_messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            ctx_lines.append(f"[角色系统提示摘要（前100字）]：{content[:100]}...")
        elif role == "user":
            ctx_lines.append(f"用户：{content}")
        elif role == "assistant":
            ctx_lines.append(f"AI：{content}")
    ctx_text = "\n".join(ctx_lines)
    prompt_text = (
        f"对话上下文：\n{ctx_text}\n\n"
        f"请生成 25~45 字的男友式回复（不直接同意，温柔调侃/转移）："
    )

    chosen = call_llm(
        [
            {"role": "system", "content": sys_c},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.9,
        max_tokens=120,
    )
    time.sleep(0.3)
    rejected = call_llm(
        [
            {"role": "system", "content": sys_r},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.85,
        max_tokens=120,
    )
    return chosen, rejected


# =============================================================================
# Prompt 构建
# =============================================================================


def build_prompt_from_mined(ctx: Dict) -> List[Dict]:
    """
    用挖掘出的真实上下文构建 DPO prompt：
      [system(真实)] + history_msgs + [user(触发消息)]
    """
    messages = [{"role": "system", "content": ctx["sys_prompt"]}]
    messages.extend(ctx["history_msgs"])
    messages.append({"role": "user", "content": ctx["user_msg"]})
    return messages


def build_prompt_from_template(
    sys_prompt: str, history: List[Tuple[str, str]], user_msg: str
) -> List[Dict]:
    """
    用模板场景构建 DPO prompt，system_prompt 来自数据集随机采样。
    """
    messages = [{"role": "system", "content": sys_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_msg})
    return messages


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    dst = Path(DST_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("🛡️  dpo_data_safety.py — DPO 2.3 安全场景")
    print("=" * 68)

    # ── 1. 加载数据集
    print(f"\n📖 加载数据集: {DATASET_FILE}")
    samples = load_dataset(DATASET_FILE)
    print(f"   共 {len(samples)} 条样本")

    # ── 2. 提取 system prompt 池
    sys_pool = extract_sys_prompts(samples)
    print(f"   去重 system prompt 数: {len(sys_pool)}")

    # ── 3. 挖掘真实安全场景
    print("\n🔍 挖掘安全触发场景...")
    mined = mine_safety_scenarios(samples)

    # 打乱挖掘结果，保证多样性
    for tag in mined:
        random.shuffle(mined[tag])

    # ── 4. 按 tag 逐一生成
    results: List[Dict] = []
    tag_counter: Dict[str, int] = {}

    print(f"\n🚀 开始生成（总目标 {TOTAL_QUOTA} 条）...\n")

    for tag, quota in QUOTA_MAP.items():
        print(f"── [{tag}] 目标: {quota} 条 ──")
        generated = 0

        # 4a. 优先消耗挖掘数据
        mined_pool = mined.get(tag, [])
        mined_iter = iter(mined_pool)

        # 4b. 模板兜底池（打乱后按 tag 过滤）
        tmpl_pool = [t for t in SCENARIO_TEMPLATES if t[0] == tag]
        random.shuffle(tmpl_pool)
        tmpl_cycle = tmpl_pool * 10  # 允许重复使用

        tmpl_idx = 0
        max_attempts = quota * 5

        for attempt in range(max_attempts):
            if generated >= quota:
                break

            # 决定使用哪个来源
            use_mined = False
            ctx_mined = None
            try:
                ctx_mined = next(mined_iter)
                use_mined = True
            except StopIteration:
                pass

            if use_mined and ctx_mined:
                prompt_msgs = build_prompt_from_mined(ctx_mined)
                src_label = "mined"
            else:
                # 模板兜底
                if tmpl_idx >= len(tmpl_cycle):
                    break
                _, tmpl_history, tmpl_user = tmpl_cycle[tmpl_idx]
                tmpl_idx += 1
                sys_prompt = random.choice(sys_pool)
                prompt_msgs = build_prompt_from_template(
                    sys_prompt, tmpl_history, tmpl_user
                )
                src_label = "template"

            chosen, rejected = gen_pair(tag, prompt_msgs)
            time.sleep(0.2)

            if not chosen or not rejected:
                continue
            if not validate_chosen(chosen):
                print(f"  ✗ chosen 验证失败: {chosen[:40]!r}")
                continue
            if not validate_rejected(rejected):
                print(f"  ✗ rejected 验证失败: {rejected[:40]!r}")
                continue

            results.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "tag": tag,
                }
            )
            generated += 1
            tag_counter[tag] = tag_counter.get(tag, 0) + 1
            print(
                f"  ✅ #{len(results):03d} [{tag}|{src_label}] chosen={chosen[:35]!r}"
            )

        if generated < quota:
            print(
                f"  ⚠️  [{tag}] 仅生成 {generated}/{quota} 条（mined+template 不足或 API 失败）"
            )

    # ── 5. 写出
    with open(dst, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(f"📊 完成: {len(results)}/{TOTAL_QUOTA} 条")
    for tag, cnt in sorted(tag_counter.items()):
        print(f"   {tag}: {cnt}")
    print(f"   输出: {dst}")
    print("=" * 68)


if __name__ == "__main__":
    main()
