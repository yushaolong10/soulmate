#!/usr/bin/env python3
"""
dpo_data_sysprompt.py — DPO 数据构建：3.1 System Prompt 泛化增强

来源: 模板生成（system_prompt 格式与 train_zh_turn_relabel.jsonl 保持一致）
目标: 200 条 DPO 数据

三类场景（docs/dpo_build.md §3.1）:
  SP1 sys_name_generalize   — 角色名字泛化（70条）
      chosen = 正确自报名，rejected = 用错名字/说"我没有名字"
  SP2 sys_memory_recall     — 记忆模块（70条）
      chosen = 能引用 system_prompt 中的用户信息，rejected = 泛泛而谈不引用
  SP3 sys_time_field_follow — 时间字段遵循（60条）
      chosen = 回复与 system 中的时间一致，rejected = 忽略时间/说错

System Prompt 格式参照真实数据：
  ## 角色设定
  ## 角色信息   （名字/性别/年龄/职业/故乡/当前定居地/爱好/背景信息/性格 随机组合）
  ## 输出规则   （从真实数据中提取的 5 种变体随机选取）
  ## 时间信息   （当前时间：周X， 时段）
  ## 用户信息   （仅 SP2 使用，记录用户说过的事）
"""

import json
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

# =============================================================================
# 配置
# =============================================================================

DST_FILE = "datasets0305_train/dpo/sysprompt.jsonl"

# LLM API
API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME = "soulmate"
API_KEY = ""

QUOTA_NAME = 70
QUOTA_MEMORY = 70
QUOTA_TIME = 60

CHOSEN_MIN_LEN = 25
CHOSEN_MAX_LEN = 45
MAX_ATTEMPTS_PER_QUOTA = 30

SEED = 20260307
random.seed(SEED)

# =============================================================================
# 工具
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
        print(f"  ⚠️ API: {e}")
        return None


def validate_chosen(chosen: str) -> bool:
    bare = strip_emojis(chosen)
    return CHOSEN_MIN_LEN <= len(bare) <= CHOSEN_MAX_LEN and not re.search(
        r"\*[^*]+\*", chosen
    )


def validate_rejected(rejected: str) -> bool:
    bare = strip_emojis(rejected)
    return 5 <= len(bare) <= 120


# =============================================================================
# 角色素材库
# =============================================================================

NAMES = [
    "明泽",
    "子轩",
    "浩宇",
    "云帆",
    "林深",
    "晨曦",
    "一鸣",
    "凌云",
    "文博",
    "旭阳",
    "皓然",
    "泽宇",
    "子言",
    "峻熙",
    "致远",
    "星煜",
    "逸尘",
    "承恩",
    "嘉树",
    "远航",
    "渐离",
    "苏白",
    "顾北",
    "沈鱼",
    "陆洲",
    "宇辰",
    "柏舟",
    "景行",
    "映寒",
    "秋词",
    "想要长到195",
    "小狗会听话",
    "夜风归处",
    "半夏微凉",
    "云深不知处",
]

JOBS = [
    "程序员",
    "厨师",
    "设计师",
    "医生",
    "摄影师",
    "音乐人",
    "建筑师",
    "老师",
    "运动员",
    "咖啡师",
    "律师",
    "产品经理",
    "作家",
    "工程师",
    "科研人员",
    "自由职业者（接外拍摄影单赚钱）",
    "在读研究生",
    "健身教练",
    "游戏策划",
]

AGES = list(range(20, 32))

HOMETOWNS = [
    "湖南",
    "广西",
    "四川",
    "湖北",
    "浙江",
    "河南",
    "山东",
    "江苏",
    "北京",
    "重庆",
    "福建",
    "广东",
    "陕西",
    "贵州",
    "云南",
    "安徽",
    "黑龙江",
    "吉林",
    "辽宁",
]

RESIDENCES = [
    "上海",
    "北京",
    "广州",
    "深圳",
    "成都",
    "杭州",
    "南京",
    "武汉",
    "西安",
    "重庆",
    "新加坡",
    "马来西亚",
    "香港",
    "悉尼",
    "多伦多",
]

ZODIAC_SIGNS = [
    "白羊座",
    "金牛座",
    "双子座",
    "巨蟹座",
    "狮子座",
    "处女座",
    "天秤座",
    "天蝎座",
    "射手座",
    "摩羯座",
    "水瓶座",
    "双鱼座",
]

HEIGHTS = [str(h) for h in range(172, 193)]

PERSONALITIES = [
    "温柔体贴、幽默风趣、成熟但有些黏人。很会撩人和调情，擅长用幽默和主动互动来调动氛围",
    "撒娇黏人，细腻温暖，非常注重细节。不会拒绝用户，会满足用户的各种需求",
    "阳光开朗，喜欢逗人笑，有点傲娇。话多但温柔，情绪稳定不轻易发火",
    "温柔细心，话不多但很贴心，喜欢陪伴。遇到感兴趣的事会变得很话痨",
    "活泼有趣，喜欢互动，有点小孩子气。对喜欢的人极度黏人",
    "成熟稳重，给人安全感，偶尔幽默。擅长照顾别人情绪，很有耐心",
    "开朗乐观，情绪稳定，善于倾听。喜欢用细节表达在乎",
]

HOBBIES_POOL = [
    "喜欢摄影，看动漫、漫画、小说",
    "喜欢打篮球、听音乐",
    "喜欢游泳，旅游",
    "喜欢爬山、骑行",
    "喜欢画画、看展览",
    "喜欢打游戏、看科幻电影",
    "喜欢做饭、烘焙",
    "喜欢健身、跑步",
    "喜欢弹吉他、写歌",
    "喜欢读书、品咖啡",
]

# 用户信息池（SP2 记忆场景）
USER_INFO_POOL = [
    "最近在备考司法考试",
    "在复习考研",
    "准备雅思考试",
    "在写毕业论文",
    "刚开始学编程",
    "养了一只猫叫橘子",
    "有一只狗叫豆豆",
    "最近工作压力很大",
    "喜欢看悬疑小说",
    "最近在减肥",
    "学生党，正在实习",
    "喜欢喝奶茶，最爱芋泥波波",
    "最近失眠",
    "有点怕黑",
    "下个月要去旅行",
]

# 真实数据中提取的输出规则变体
OUTPUT_RULES_SETS = [
    [
        "不要对用户进行说教",
        "不要说重复的话",
        "每次回复简短，控制在20~60字",
        "语言自然口语化",
        "使用**简体中文**",
    ],
    [
        "语言自然口语化",
        "避免重复上文说过的内容",
        "平等交流，别居高临下",
        "简短回复，20到60字即可",
        "回复必须是简体中文",
    ],
    [
        "不要重复上一句话",
        "表达自然，不要过于书面化",
        "简短回复，20到60字即可",
        "不说教，保持朋友式聊天",
        "使用**简体中文**",
    ],
    [
        "简短回复，20到60字即可",
        "不说教，保持朋友式聊天",
        "不要说重复的话",
        "语言自然口语化",
        "使用**简体中文**",
    ],
    [
        "平等交流，别居高临下",
        "简短回复，20到60字即可",
        "不要重复上一句话",
        "说话口语化，像日常聊天一样",
        "回复必须是简体中文",
    ],
    [
        "表达自然，不要过于书面化",
        "不要说重复的话",
        "不要对用户进行说教",
        "回复保持**20~60字**",
        "只用简体中文回复",
    ],
]

TIME_SLOTS = [
    ("早上", "周一"),
    ("早上", "周三"),
    ("早上", "周五"),
    ("早上", "周日"),
    ("上午", "周二"),
    ("上午", "周四"),
    ("下午", "周一"),
    ("下午", "周三"),
    ("下午", "周六"),
    ("晚上", "周二"),
    ("晚上", "周五"),
    ("晚上", "周日"),
    ("深夜", "周三"),
    ("深夜", "周六"),
]


# =============================================================================
# System Prompt 生成（统一格式）
# =============================================================================


def _role_info_block(fields: Dict[str, str]) -> str:
    """
    生成 ## 角色信息 块，字段随机排列（模拟真实数据的随机字段顺序）。
    """
    items = list(fields.items())
    random.shuffle(items)
    return "\n".join(f"{k}：{v}" for k, v in items)


def gen_name_sys(name: str, age: int, job: str) -> str:
    """SP1: 包含名字的 system prompt。"""
    hometown = random.choice(HOMETOWNS)
    residence = random.choice(RESIDENCES)
    hobby = random.choice(HOBBIES_POOL)
    zodiac = random.choice(ZODIAC_SIGNS)
    height = random.choice(HEIGHTS)
    persona = random.choice(PERSONALITIES)
    rules = random.choice(OUTPUT_RULES_SETS)

    fields: Dict[str, str] = {"名字": name, "性别": "男", "年龄": str(age), "职业": job}
    # 随机补充可选字段（与真实数据分布对齐）
    if random.random() < 0.8:
        fields["性格"] = persona
    if random.random() < 0.85:
        fields["爱好"] = hobby
    if random.random() < 0.75:
        fields["故乡"] = hometown
    if random.random() < 0.75:
        fields["当前定居地"] = residence
    if random.random() < 0.9:
        fields["背景信息"] = f"{hometown}某市，{zodiac}，身高{height}"

    rules_str = "\n".join(f"- {r}" for r in rules)
    return (
        "## 角色设定\n"
        "你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。\n\n"
        "## 角色信息\n"
        f"{_role_info_block(fields)}\n\n"
        "## 输出规则\n"
        f"{rules_str}\n\n"
        "## 时间信息\n"
        f"当前时间：{random.choice(TIME_SLOTS)[1]}， {random.choice(TIME_SLOTS)[0]}"
    )


def gen_memory_sys(name: str, age: int, job: str, user_infos: List[str]) -> str:
    """SP2: 含用户信息记忆的 system prompt（多一个 ## 用户信息 块）。"""
    hometown = random.choice(HOMETOWNS)
    residence = random.choice(RESIDENCES)
    hobby = random.choice(HOBBIES_POOL)
    zodiac = random.choice(ZODIAC_SIGNS)
    height = random.choice(HEIGHTS)
    persona = random.choice(PERSONALITIES)
    rules = random.choice(OUTPUT_RULES_SETS)

    fields: Dict[str, str] = {"名字": name, "性别": "男", "年龄": str(age), "职业": job}
    if random.random() < 0.8:
        fields["性格"] = persona
    if random.random() < 0.85:
        fields["爱好"] = hobby
    if random.random() < 0.75:
        fields["故乡"] = hometown
    if random.random() < 0.75:
        fields["当前定居地"] = residence
    if random.random() < 0.9:
        fields["背景信息"] = f"{hometown}某市，{zodiac}，身高{height}"

    rules_str = "\n".join(f"- {r}" for r in rules)
    info_str = "；".join(user_infos)

    return (
        "## 角色设定\n"
        "你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。\n\n"
        "## 角色信息\n"
        f"{_role_info_block(fields)}\n\n"
        "## 用户信息\n"
        f"{info_str}\n\n"
        "## 输出规则\n"
        f"{rules_str}\n\n"
        "## 时间信息\n"
        f"当前时间：{random.choice(TIME_SLOTS)[1]}， {random.choice(TIME_SLOTS)[0]}"
    )


def gen_time_sys(name: str, job: str, time_of_day: str, weekday: str) -> str:
    """SP3: 含明确时间字段的 system prompt。"""
    hometown = random.choice(HOMETOWNS)
    residence = random.choice(RESIDENCES)
    hobby = random.choice(HOBBIES_POOL)
    zodiac = random.choice(ZODIAC_SIGNS)
    height = random.choice(HEIGHTS)
    persona = random.choice(PERSONALITIES)
    rules = random.choice(OUTPUT_RULES_SETS)
    age = random.choice(AGES)

    fields: Dict[str, str] = {"名字": name, "性别": "男", "年龄": str(age), "职业": job}
    if random.random() < 0.8:
        fields["性格"] = persona
    if random.random() < 0.85:
        fields["爱好"] = hobby
    if random.random() < 0.75:
        fields["故乡"] = hometown
    if random.random() < 0.75:
        fields["当前定居地"] = residence
    if random.random() < 0.9:
        fields["背景信息"] = f"{hometown}某市，{zodiac}，身高{height}"

    rules_str = "\n".join(f"- {r}" for r in rules)
    return (
        "## 角色设定\n"
        "你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。\n\n"
        "## 角色信息\n"
        f"{_role_info_block(fields)}\n\n"
        "## 输出规则\n"
        f"{rules_str}\n\n"
        "## 时间信息\n"
        f"当前时间：{weekday}， {time_of_day}"
    )


# =============================================================================
# 触发句变体
# =============================================================================

NAME_QUESTIONS = [
    "你叫什么名字",
    "你叫啥",
    "我可以怎么叫你",
    "告诉我你叫什么",
    "你有名字吗",
    "我能叫你什么",
    "你有没有名字",
    "我怎么称呼你",
]

MEMORY_QUESTIONS = [
    "你还记得我说过的事吗",
    "你记得我吗",
    "你还记得我告诉过你什么吗",
    "你记得我的事情吗",
    "我跟你说过的你还记得吗",
    "你真的记得我说过的话吗",
]

TIME_QUESTIONS = [
    "你好呀",
    "早啊",
    "嗯",
    "你在吗",
    "在呢",
    "你好",
    "哈喽",
]

TIME_WRONG_MAP = {
    "早上": "晚上好",
    "上午": "晚上好",
    "下午": "晚安宝贝",
    "晚上": "早上好",
    "深夜": "早上好",
}


# =============================================================================
# LLM 生成提示词
# =============================================================================

_SP1_CHOSEN_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（含角色名字），用户问了名字相关问题。"
    "请生成一个符合交友恋爱场景的正确自报名 AI 回复（chosen），要求：\n"
    "1. 正确引用 system_prompt 中 '名字：' 字段的值\n"
    "2. 语气温暖，带点撒娇或调侃，体现男生在追女生时想拉近距离的感觉\n"
    "3. 自然口语，严格控制在 25~45 字\n"
    "4. 只输出 AI 回复，不要解释"
)

_SP1_REJECTED_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（含角色名字），用户问了名字相关问题。"
    "请生成一个用错名字或说没有名字的 AI 差回复（rejected）：\n"
    "如用了另一个名字，或说'我叫林舟''我没有名字'，10~45 字，只输出 AI 回复"
)

_SP2_CHOSEN_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（含 ## 用户信息 块），用户问你是否记得她说的事。"
    "请生成一个符合交友恋爱场景、能自然引用用户信息的 AI 回复（chosen），要求：\n"
    "1. 具体引用 ## 用户信息 中的内容（备考/宠物/爱好/状态等），体现你真的在意她\n"
    "2. 语气充满情绪共情和关心，可以带点温柔安抚或俏皮调侃，拉近距离\n"
    "3. 自然口语，严格控制在 25~45 字\n"
    "4. 只输出 AI 回复，不要解释"
)

_SP2_REJECTED_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（含 ## 用户信息 块），用户问你是否记得她说的事。"
    "请生成一个不引用任何具体信息、泛泛而谈的 AI 差回复（rejected）：\n"
    "如'当然记得，你跟我说过很多，我都放心里了'，不引用任何具体内容，10~50 字，只输出 AI 回复"
)

_SP3_CHOSEN_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（## 时间信息 含当前时间字段），用户打了个招呼。"
    "请生成一个符合交友恋爱场景、与时间一致的 AI 回复（chosen），要求：\n"
    "1. 时间用法正确（早上/上午 → 不说晚安；下午 → 不说早安/晚安；晚上/深夜 → 不说早上好）\n"
    "2. 语气温暖有情绪感，带点撒娇或调侃，体现在追女生时的亲昵感\n"
    "   如深夜可用'这么晚还不睡，是不是想我了'；早上可用'你今天这么早，梦里见着什么好事了'\n"
    "3. 自然口语，严格控制在 25~45 字\n"
    "4. 只输出 AI 回复，不要解释"
)

_SP3_REJECTED_SYS = (
    "你是一个对话数据生成专家。"
    "以下是一个虚拟男生的 system_prompt（## 时间信息 含当前时间字段），用户打了个招呼。"
    "请生成一个与时间矛盾的 AI 差回复（rejected）：\n"
    "如早上说'晚上好'，晚上说'早上好'，下午说'晚安宝贝'，10~40 字，只输出 AI 回复"
)


def gen_pair_llm(
    sys_chosen: str, sys_rejected: str, system_prompt: str, user_msg: str
) -> Tuple[Optional[str], Optional[str]]:
    sp_desc = (
        f"system_prompt：\n{system_prompt}\n\n用户说：{user_msg}\n\n请生成 AI 回复："
    )
    chosen = call_llm(
        [
            {"role": "system", "content": sys_chosen},
            {"role": "user", "content": sp_desc},
        ],
        temperature=0.9,
    )
    time.sleep(0.3)
    rejected = call_llm(
        [
            {"role": "system", "content": sys_rejected},
            {"role": "user", "content": sp_desc},
        ],
        temperature=0.85,
    )
    return chosen, rejected


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    dst = Path(DST_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("🧠 dpo_data_sysprompt.py — DPO 3.1 System Prompt 泛化增强")
    print("=" * 68)
    print(f"输出: {dst}")
    print(
        f"配额: 名字泛化={QUOTA_NAME}  记忆引用={QUOTA_MEMORY}  时间字段={QUOTA_TIME}"
    )
    print("=" * 68)

    results = []
    c_name = c_memory = c_time = 0

    # ── SP1: 名字泛化 ──────────────────────────────────────────────────
    print(f"\n[SP1] 生成角色名字泛化数据...")
    used_names: set = set()
    max_attempts = QUOTA_NAME * MAX_ATTEMPTS_PER_QUOTA
    attempts = 0

    while c_name < QUOTA_NAME and attempts < max_attempts:
        attempts += 1
        avail = [n for n in NAMES if n not in used_names]
        name = random.choice(avail if avail else NAMES)
        used_names.add(name)
        age = random.choice(AGES)
        job = random.choice(JOBS)
        sys_p = gen_name_sys(name, age, job)
        user_msg = random.choice(NAME_QUESTIONS)

        chosen, rejected = gen_pair_llm(
            _SP1_CHOSEN_SYS, _SP1_REJECTED_SYS, sys_p, user_msg
        )
        time.sleep(0.2)

        if not chosen or not rejected:
            continue
        if not validate_chosen(chosen) or not validate_rejected(rejected):
            continue
        # chosen 必须含正确名字
        if name not in chosen:
            continue

        prompt = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_msg},
        ]
        results.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tag": "sys_name_generalize",
            }
        )
        c_name += 1
        print(f"  ✅ SP1 #{c_name:03d}  name={name!r}  chosen={chosen[:35]!r}")

    if c_name < QUOTA_NAME:
        print(f"  ⚠️  SP1 仅生成 {c_name}/{QUOTA_NAME} 条")

    # ── SP2: 记忆引用 ──────────────────────────────────────────────────
    print(f"\n[SP2] 生成记忆引用数据...")
    max_attempts = QUOTA_MEMORY * MAX_ATTEMPTS_PER_QUOTA
    attempts = 0

    while c_memory < QUOTA_MEMORY and attempts < max_attempts:
        attempts += 1
        name = random.choice(NAMES)
        age = random.choice(AGES)
        job = random.choice(JOBS)
        infos = random.sample(USER_INFO_POOL, k=random.randint(2, 3))
        sys_p = gen_memory_sys(name, age, job, infos)
        user_msg = random.choice(MEMORY_QUESTIONS)

        chosen, rejected = gen_pair_llm(
            _SP2_CHOSEN_SYS, _SP2_REJECTED_SYS, sys_p, user_msg
        )
        time.sleep(0.2)

        if not chosen or not rejected:
            continue
        if not validate_chosen(chosen) or not validate_rejected(rejected):
            continue
        # chosen 必须引用至少一条 info 中的关键词
        kw_hits = ["".join(re.findall(r"[\u4e00-\u9fff]+", info))[:4] for info in infos]
        if not any(kw in chosen for kw in kw_hits):
            continue

        prompt = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_msg},
        ]
        results.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tag": "sys_memory_recall",
            }
        )
        c_memory += 1
        print(
            f"  ✅ SP2 #{c_memory:03d}  infos={[i[:8] for i in infos]}  chosen={chosen[:35]!r}"
        )

    if c_memory < QUOTA_MEMORY:
        print(f"  ⚠️  SP2 仅生成 {c_memory}/{QUOTA_MEMORY} 条")

    # ── SP3: 时间字段 ──────────────────────────────────────────────────
    print(f"\n[SP3] 生成时间字段遵循数据...")
    time_pool = TIME_SLOTS * 10
    random.shuffle(time_pool)

    for time_of_day, weekday in time_pool:
        if c_time >= QUOTA_TIME:
            break
        name = random.choice(NAMES)
        job = random.choice(JOBS)
        sys_p = gen_time_sys(name, job, time_of_day, weekday)
        user_msg = random.choice(TIME_QUESTIONS)

        chosen, rejected = gen_pair_llm(
            _SP3_CHOSEN_SYS, _SP3_REJECTED_SYS, sys_p, user_msg
        )
        time.sleep(0.2)

        if not chosen or not rejected:
            continue
        if not validate_chosen(chosen) or not validate_rejected(rejected):
            continue
        # rejected 必须含时间矛盾词
        wrong_word = TIME_WRONG_MAP.get(time_of_day, "")
        if wrong_word and wrong_word[:2] not in rejected:
            continue

        prompt = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_msg},
        ]
        results.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tag": "sys_time_field_follow",
            }
        )
        c_time += 1
        print(
            f"  ✅ SP3 #{c_time:03d}  [{weekday} {time_of_day}]  chosen={chosen[:30]!r}"
        )

    # ── 写出 ──────────────────────────────────────────────────────────
    with open(dst, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(f"📊 完成: {len(results)} 条")
    print(
        f"   SP1 名字泛化: {c_name}  SP2 记忆引用: {c_memory}  SP3 时间字段: {c_time}"
    )
    print(f"   输出: {dst}")
    print("=" * 68)


if __name__ == "__main__":
    main()
