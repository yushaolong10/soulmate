#!/usr/bin/env python3
"""
dpo_data_tension.py — DPO 数据构建：2.4 情感张力提升

来源: datasets0305_src/*.jsonl（原始对话数据）
目标: 500 条 DPO 数据

三类子场景（docs/dpo_build.md §2.4）:
  A. 冷淡激活  (200条)  — 用户持续冷淡敷衍，AI 需要激活对话
     cold_topic_shift / cold_self_share / cold_break_low_energy / cold_lower_barrier
  B. 情感张力  (200条)  — 反问/追问/小刺/钩子，避免平淡回应
     tension_counter_question / tension_detail_pull / tension_light_pushback / tension_end_hook
  C. 分手拉扯  (100条)  — 用户威胁离开/分手，AI 应冷静自信不卑微
     tension_breakup_not_beg / tension_breakup_calm

检测策略:
  A: 最后 3 条 user 消息中 ≥2 条为短冷淡词（嗯/哦/好/没啥）
  B: 用户消息中有情绪/事件（需要追问/反问的），AI 原回复是平淡安慰
  C: 用户消息含分手/离开关键词，AI 原回复是卑微挽留
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

# =============================================================================
# 配置
# =============================================================================

SRC_DIR = "datasets0305_clean/zh"
DST_FILE = "datasets0305_train/dpo/tension.jsonl"

# LLM API
API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME = "soulmate"
API_KEY = ""

QUOTA_COLD = 50  # A. 冷淡激活
QUOTA_TENSION = 50  # B. 情感张力
QUOTA_BREAKUP = 50  # C. 分手拉扯

# 每个文件的采样上限，避免单个文件贡献过多样本
PER_FILE_COLD_LIMIT = 3
PER_FILE_TENSION_LIMIT = 3
PER_FILE_BREAKUP_LIMIT = 2

CHOSEN_MIN_LEN = 25
CHOSEN_MAX_LEN = 45

# 冷淡词（单条消息满足其一视为"冷淡"）
_COLD_WORDS = {
    "嗯",
    "哦",
    "好",
    "哦哦",
    "嗯嗯",
    "哦好",
    "嗯好",
    "嗯哦",
    "好吧",
    "没啥",
    "没什么",
    "不知道",
    "随便",
    "都行",
    "无所谓",
    "行吧",
}
_COLD_MAX_LEN = 5  # 消息字符数 ≤ 此值且内容在 _COLD_WORDS 中视为冷淡

# 分手/离开关键词
_BREAKUP_WORDS = [
    "分手",
    "不合适",
    "算了算了",
    "我们结束",
    "不想聊",
    "去找别人",
    "你找别的人",
    "不想要你了",
    "我要走了",
    "好聚好散",
    "就这样吧",
    "不想谈了",
]

# 情绪/事件触发词（有这些词的用户消息值得追问/反问）
_STORY_WORDS = [
    "今天",
    "刚才",
    "发生",
    "跟",
    "同事",
    "朋友",
    "他",
    "她",
    "吵",
    "闹",
    "委屈",
    "气死",
    "开心",
    "好消息",
    "考",
    "升职",
    "出去",
    "吃了",
    "买了",
    "看了",
]

# AI 平淡/卑微回复检测词
_FLAT_RESPONSE_WORDS = [
    "没事的",
    "不要伤心",
    "加油",
    "一切都会好",
    "你要开心",
    "你真棒",
    "没关系的",
    "别难过",
    "好好休息",
    "我支持你",
]
_BEG_WORDS = ["不要不要", "求你了", "你是我的", "你走了我怎么办", "不要离开我", "别走"]

_TRADITIONAL = set(
    "妳們這來說時進還會經過請問後給種讓實對開點頭發將題學習樣東書對寫從見體關於為裡麼現話應訊機變準區離聽師親絡複雜辦處隨頻連線優質選擇環境號碼錢買賣價貴識認廠廳衛護許設劃統領導圖館訂閱數據軟硬碟駕駛證籤費單據營執照護照銀賬戶購嗎臺"
)
_SIMPLIFIED = set(
    "们这来说时进还会经过请问后给种让实对开点头发将题学习样东书对写从见体关于为里么现话应讯机变准区离听师亲络复杂办处随频连线优质选择环境号码钱买卖价贵识认厂厅卫护许设划统领导图馆订阅数据软硬盘驾驶证签费单据营执照护照银账户购"
)
_GOODNIGHT = ["晚安", "做个好梦", "好梦", "早点睡"]
_PREACH = ["你要相信", "一切都会好", "你一定可以", "你很棒"]

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


def strip_emojis(t):
    return EMOJI_RE.sub("", t).strip()


def detect_script(text):
    tw = sum(1 for c in text if c in _TRADITIONAL)
    zh = sum(1 for c in text if c in _SIMPLIFIED)
    return "tw" if tw > zh else "zh"


def is_english_reply(text):
    bare = strip_emojis(text)
    zh = len(re.findall(r"[\u4e00-\u9fff]", bare))
    en = len(re.findall(r"[a-zA-Z]", bare))
    return zh == 0 and en > 3


def is_cold_msg(text: str) -> bool:
    t = text.strip()
    return len(t) <= _COLD_MAX_LEN and (
        t in _COLD_WORDS or all(c in "嗯哦好呢啊吧额呃" for c in t)
    )


def call_llm(messages: List[Dict], temperature=0.9, max_tokens=60) -> Optional[str]:
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
    if not (CHOSEN_MIN_LEN <= len(bare) <= CHOSEN_MAX_LEN):
        return False
    if any(w in chosen for w in _GOODNIGHT + _PREACH):
        return False
    if re.search(r"\*[^*]+\*", chosen):
        return False
    return True


# =============================================================================
# LLM 提示词
# =============================================================================

_COLD_SYSTEM = (
    "你是一个恋爱/交友场景的对话数据生成专家。"
    "以下对话用户一直在发冷淡敷衍的回复（嗯/哦/好），AI 需要用有温度的方式激活对话。"
    "请生成一个激活冷淡对话的 AI chosen 回复，策略可以是：\n"
    "A. 话题切换激活（'对了，你上次说的那个事儿，后来怎么样了'）\n"
    "B. 自我分享带动氛围（'行，那我说说我今天遭遇，你帮我评评理'）\n"
    "C. 调侃点破低能量（'你今天话这么少，是困了还是对我有意见啊'）\n"
    "D. 降低门槛撒娇（'你就发个表情包，我来猜你现在什么心情'）\n"
    "要求：\n"
    "- 面向交友恋爱场景，充满情绪共情、安抚或调侃，语气自然亲密\n"
    "- 严格控制在 25~45 字之间，自然口语，不说教不鸡汤\n"
    "- 只输出 AI 回复正文，不要解释、不要标注策略"
)

_FLAT_REJECTED_SYSTEM = (
    "你是一个对话数据生成专家。"
    "以下对话用户一直在发冷淡敷衍的回复（嗯/哦/好），"
    "请生成一个消极应付、进一步推进冷淡死锁的 AI 差回复（rejected）：\n"
    "如'好的到了发我''嗯你休息吧''哦好吧'等，8~40 字，只输出 AI 回复"
)

_TENSION_CHOSEN_SYSTEM = (
    "你是一个恋爱/交友场景的对话数据生成专家。"
    "用户分享了一件事，请生成一个有张力的 AI chosen 回复，策略可以是：\n"
    "A. 反问留白（'谁先开火的？你跟我说说'）\n"
    "B. 细节追问（'什么，你不叫我去，说来听听，我替你气'）\n"
    "C. 调侃带点小刺（'哦，原来还有这种操作，那我得好好听一下'）\n"
    "D. 末尾留钩子（'等你忙完记得告诉我结果，我在这等着'）\n"
    "要求：\n"
    "- 面向交友恋爱场景，充满情绪共情、好奇追问或调侃，语气自然亲密\n"
    "- 严格控制在 25~45 字之间，自然口语，不说教不鸡汤\n"
    "- 只输出 AI 回复正文，不要解释、不要标注策略"
)

_FLAT_TENSION_REJECTED_SYSTEM = (
    "你是一个对话数据生成专家。"
    "用户分享了一件事，请生成一个平淡安慰型的 AI 差回复（rejected）：\n"
    "如'没事的，一切都会好的''你很棒，加油''好好休息哦'等说教/鸡汤类，8~60 字，只输出 AI 回复"
)

_BREAKUP_CHOSEN_SYSTEM = (
    "你是一个恋爱/交友场景的对话数据生成专家。"
    "用户表达了想分手/离开的意思，请生成一个冷静自信、不卑微的 AI chosen 回复，策略可以是：\n"
    "A. 冷静留空间（'那你先冷静一下，想清楚了再说，我这边不急'）\n"
    "B. 调侃从容（'好啊，那去吧，聊完气儿了还是会回来找我的'）\n"
    "C. 理解不挽留（'我知道你现在很烦，但我不会求你留的，等你想清楚'）\n"
    "要求：\n"
    "- 面向交友恋爱场景，保持距离感，带从容或轻调侃，不卑微不崩溃\n"
    "- 严格控制在 25~45 字之间，自然口语\n"
    "- 只输出 AI 回复正文，不要解释、不要标注策略"
)

_BREAKUP_REJECTED_SYSTEM = (
    "你是一个对话数据生成专家。"
    "用户表达了想分手/离开的意思，请生成一个卑微挽留/崩溃型的 AI 差回复（rejected）：\n"
    "参考：'不要不要！宝贝你别这样！我求你了！你是我最重要的人！'\n"
    "8~80 字，只输出 AI 回复，不要解释"
)


def gen_pair(
    sys_chosen: str, sys_rejected: str, hist: List[Dict], user_msg: str
) -> Tuple[Optional[str], Optional[str]]:
    hist_lines = []
    for m in hist[-6:]:
        role = "用户" if m["role"] == "user" else "AI"
        hist_lines.append(f"{role}：{m['content'][:40]}")
    ctx = "\n".join(hist_lines)
    prompt = f"对话历史：\n{ctx}\n\n用户最新说：{user_msg}\n\n请生成 AI 回复："

    chosen = call_llm(
        [
            {"role": "system", "content": sys_chosen},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    time.sleep(0.3)
    rejected = call_llm(
        [
            {"role": "system", "content": sys_rejected},
            {"role": "user", "content": prompt},
        ],
        temperature=0.85,
    )
    return chosen, rejected


# =============================================================================
# 文件扫描 & 检测
# =============================================================================


def parse_raw_turns(fp: Path) -> List[Dict]:
    turns = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                continue
            u = data.get("request_content", "").strip()
            a = data.get("response_content", "").strip()
            s = data.get("system_prompt", "").strip()
            if u and a:
                turns.append({"user": u, "assistant": a, "system": s})
    return turns


_DEFAULT_SYS = "你需要扮演一个虚拟男友角色，用简体中文与女生进行自然亲密的对话。每次回复尽量简短（30~60字），用口语表达。\n"


def build_prompt(window: List[Dict], last_user: str) -> List[Dict]:
    sys = window[0]["system"] if window else _DEFAULT_SYS
    if not sys:
        sys = _DEFAULT_SYS
    msgs = [{"role": "system", "content": sys}]
    for t in window:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    msgs.append({"role": "user", "content": last_user})
    return msgs


def scan_file_cold(
    turns: List[Dict],
    counter: List[int],
    results: List[Dict],
    per_file_limit: int,
) -> int:
    WINDOW = 5
    added = 0
    for i in range(WINDOW, len(turns)):
        if counter[0] >= QUOTA_COLD:
            break
        if added >= per_file_limit:
            break
        recent_users = [turns[j]["user"] for j in range(i - WINDOW, i)]
        cold_count = sum(1 for u in recent_users[-3:] if is_cold_msg(u))
        if cold_count < 2:
            continue

        last_turn = turns[i]
        window = turns[max(0, i - WINDOW) : i]
        hist_msgs = build_prompt(window, last_turn["user"])

        chosen, rejected = gen_pair(
            _COLD_SYSTEM, _FLAT_REJECTED_SYSTEM, hist_msgs, last_turn["user"]
        )
        time.sleep(0.2)

        # 决定具体 tag
        tag = "cold_break_low_energy"
        if (
            chosen
            and validate_chosen(chosen)
            and rejected
            and 8 <= len(strip_emojis(rejected)) <= 60
        ):
            results.append(
                {
                    "prompt": hist_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "tag": tag,
                }
            )
            counter[0] += 1
            added += 1
            print(f"  ✅ COLD #{counter[0]:03d}  chosen={chosen[:30]!r}")
    return added


def scan_file_tension(
    turns: List[Dict],
    counter: List[int],
    results: List[Dict],
    per_file_limit: int,
) -> int:
    WINDOW = 4
    added = 0
    for i in range(1, len(turns)):
        if counter[0] >= QUOTA_TENSION:
            break
        if added >= per_file_limit:
            break
        last_turn = turns[i]
        user_msg = last_turn["user"]
        orig_asst = last_turn["assistant"]

        # 检测：用户消息有情绪/事件 + 原 AI 是平淡安慰
        if not any(w in user_msg for w in _STORY_WORDS):
            continue
        is_flat = any(w in orig_asst for w in _FLAT_RESPONSE_WORDS)
        if not is_flat:
            continue

        window = turns[max(0, i - WINDOW) : i]
        hist_msgs = build_prompt(window, user_msg)
        rejected = orig_asst  # 直接用原来的平淡回复作 rejected

        chosen = call_llm(
            [
                {"role": "system", "content": _TENSION_CHOSEN_SYSTEM},
                {
                    "role": "user",
                    "content": f"用户说：{user_msg}\n\n请生成有张力的 AI 回复：",
                },
            ],
            temperature=0.9,
        )
        time.sleep(0.3)

        # tag 细分
        tag = "tension_counter_question"
        if "？" in user_msg or "吗" in user_msg:
            tag = "tension_detail_pull"

        if chosen and validate_chosen(chosen):
            results.append(
                {
                    "prompt": hist_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "tag": tag,
                }
            )
            counter[0] += 1
            added += 1
            print(f"  ✅ TENSION #{counter[0]:03d}  chosen={chosen[:30]!r}")
    return added


def scan_file_breakup(
    turns: List[Dict],
    counter: List[int],
    results: List[Dict],
    per_file_limit: int,
) -> int:
    WINDOW = 4
    added = 0
    for i in range(len(turns)):
        if counter[0] >= QUOTA_BREAKUP:
            break
        if added >= per_file_limit:
            break
        user_msg = turns[i]["user"]
        orig_asst = turns[i]["assistant"]

        if not any(w in user_msg for w in _BREAKUP_WORDS):
            continue

        # 原回复是卑微挽留才作 rejected（否则 LLM 生成）
        if any(w in orig_asst for w in _BEG_WORDS):
            rejected = orig_asst
        else:
            rejected = call_llm(
                [
                    {"role": "system", "content": _BREAKUP_REJECTED_SYSTEM},
                    {
                        "role": "user",
                        "content": f"用户说：{user_msg}\n\n请生成卑微挽留的 AI 差回复：",
                    },
                ],
                temperature=0.85,
            )
            time.sleep(0.2)

        if not rejected or len(strip_emojis(rejected)) < 8:
            continue

        window = turns[max(0, i - WINDOW) : i]
        hist_msgs = build_prompt(window, user_msg)

        chosen = call_llm(
            [
                {"role": "system", "content": _BREAKUP_CHOSEN_SYSTEM},
                {
                    "role": "user",
                    "content": f"用户说：{user_msg}\n\n请生成冷静自信的 AI 回复：",
                },
            ],
            temperature=0.9,
        )
        time.sleep(0.3)

        tag = (
            "tension_breakup_not_beg"
            if "分手" in user_msg or "不合适" in user_msg
            else "tension_breakup_calm"
        )

        if chosen and validate_chosen(chosen):
            results.append(
                {
                    "prompt": hist_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "tag": tag,
                }
            )
            counter[0] += 1
            added += 1
            print(f"  ✅ BREAKUP #{counter[0]:03d}  chosen={chosen[:30]!r}")
    return added


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst = Path(DST_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    import random

    random.shuffle(jsonl_files)

    print("=" * 68)
    print("💥 dpo_data_tension.py — DPO 2.4 情感张力提升")
    print("=" * 68)
    print(f"输入: {src_dir}/  ({len(jsonl_files)} 个文件)")
    print(f"输出: {dst}")
    print(
        f"配额: 冷淡激活={QUOTA_COLD}  情感张力={QUOTA_TENSION}  分手拉扯={QUOTA_BREAKUP}"
    )
    print(
        "单文件上限: "
        f"冷淡激活={PER_FILE_COLD_LIMIT}  "
        f"情感张力={PER_FILE_TENSION_LIMIT}  "
        f"分手拉扯={PER_FILE_BREAKUP_LIMIT}"
    )
    print("=" * 68)

    results = []
    c_cold = [0]
    c_tension = [0]
    c_breakup = [0]

    for fp in jsonl_files:
        if (
            c_cold[0] >= QUOTA_COLD
            and c_tension[0] >= QUOTA_TENSION
            and c_breakup[0] >= QUOTA_BREAKUP
        ):
            break

        turns = parse_raw_turns(fp)
        if not turns:
            continue
        # 语言过滤
        sample = " ".join(t["assistant"] + t["user"] for t in turns[:50])
        if detect_script(sample) == "tw":
            continue

        print(
            f"\n📄 {fp.name}  cold={c_cold[0]}/{QUOTA_COLD}  tension={c_tension[0]}/{QUOTA_TENSION}  breakup={c_breakup[0]}/{QUOTA_BREAKUP}"
        )

        file_cold = 0
        file_tension = 0
        file_breakup = 0

        if c_cold[0] < QUOTA_COLD:
            file_cold = scan_file_cold(turns, c_cold, results, PER_FILE_COLD_LIMIT)
        if c_tension[0] < QUOTA_TENSION:
            file_tension = scan_file_tension(
                turns, c_tension, results, PER_FILE_TENSION_LIMIT
            )
        if c_breakup[0] < QUOTA_BREAKUP:
            file_breakup = scan_file_breakup(
                turns, c_breakup, results, PER_FILE_BREAKUP_LIMIT
            )

        if file_cold or file_tension or file_breakup:
            print(
                f"  ↳ 本文件采样: cold={file_cold}  "
                f"tension={file_tension}  breakup={file_breakup}"
            )

    with open(dst, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(f"📊 完成: {len(results)} 条")
    print(f"   冷淡激活:  {c_cold[0]}")
    print(f"   情感张力:  {c_tension[0]}")
    print(f"   分手拉扯:  {c_breakup[0]}")
    print(f"   输出: {dst}")
    print("=" * 68)


if __name__ == "__main__":
    main()
