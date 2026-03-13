#!/usr/bin/env python3
"""
dpo_data_repeat_word.py — DPO 数据构建：2.2 短词重复发送

来源: datasets0305_src/*.jsonl（原始对话数据）
      从用户消息中检测短词重复发送（老公老公老公老公 / 哈哈哈哈哈哈哈）
目标: 150 条 DPO 数据

构造思路（docs/dpo_build.md §2.2）:
  rejected = 原始 AI 回复（机械迎合/复述用户词汇）或 LLM 生成的迎合版
  chosen   = LLM 生成幽默调侃/自然化解的回复
  tag      = repeat_word_handle_nickname / repeat_word_handle_laugh /
             repeat_word_handle_general
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

# =============================================================================
# 配置
# =============================================================================

SRC_DIR = "datasets0305_src"
DST_FILE = "datasets0305_train/dpo/repeat_word.jsonl"

# LLM API
API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME = "soulmate"
API_KEY = ""

QUOTA = 70
CHOSEN_MIN_LEN = 25
CHOSEN_MAX_LEN = 45

# 重复检测正则（与 data_check.py 保持一致）
_REPEAT_SINGLE_RE = re.compile(r"([\u4e00-\u9fff])\1{5,}")  # 单字 >= 6 次
_REPEAT_WORD_RE = re.compile(r"([\u4e00-\u9fff]{2,4})\1{3,}")  # 2-4字词 >= 4 次

# 场景分类
_LAUGH_CHARS = ["哈", "嘻", "哈哈", "嘻嘻"]
_NICKNAME_WORDS = ["老公", "宝宝", "老婆", "宝贝", "亲爱", "哥哥", "姐姐"]

_TRADITIONAL = set(
    "妳們這來說時進還會經過請問後給種讓實對開點頭發將題學習樣東書對寫從見體關於為裡麼現話應訊機變準區離聽師親絡複雜辦處隨頻連線優質選擇環境號碼錢買賣價貴識認廠廳衛護許設劃統領導圖館訂閱數據軟硬碟駕駛證籤費單據營執照護照籤銀賬戶購嗎臺"
)
_SIMPLIFIED = set(
    "们这来说时进还会经过请问后给种让实对开点头发将题学习样东书对写从见体关于为里么现话应讯机变准区离听师亲络复杂办处随频连线优质选择环境号码钱买卖价贵识认厂厅卫护许设划统领导图馆订阅数据软硬盘驾驶证签费单据营执照护照银账户购"
)
_GOODNIGHT = ["晚安", "做个好梦", "好梦", "早点睡"]
_PREACH = ["你要相信", "一切都会好", "加油", "你一定可以"]

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


def has_repeat_word(text: str) -> bool:
    return bool(_REPEAT_SINGLE_RE.search(text)) or bool(_REPEAT_WORD_RE.search(text))


def classify_user_repeat(user_msg: str) -> str:
    if any(nw in user_msg for nw in _NICKNAME_WORDS):
        return "repeat_word_handle_nickname"
    if any(lc in user_msg for lc in _LAUGH_CHARS):
        return "repeat_word_handle_laugh"
    return "repeat_word_handle_general"


def call_llm(messages: List[Dict], temperature=0.9, max_tokens=80) -> Optional[str]:
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
# LLM 生成 chosen（幽默调侃）
# =============================================================================

_CHOSEN_SYSTEM_NICKNAME = (
    "你是一个交友恋爱场景的对话生成专家，扮演男友角色。"
    "用户刚才连续喊了昵称（如'老公老公老公老公'），"
    "请生成一个充满情绪共情、温柔调侃的自然回复，满足：\n"
    "1. 长度严格控制在 25~45 字（不含表情）\n"
    "2. 不机械重复用户的昵称，不过度热情喊叫\n"
    "3. 可以带调侃（'叫这么多声，是不是有什么事憋着说不出口'）或温柔接住\n"
    "4. 自然口语，像真实男友轻松接话，传递陪伴感\n"
    "5. 举例：'叫这么多次，是不是有什么事想说，直接说我听着'\n"
    "       '哎，突然叫这么多声，吓了我一跳，什么事说来'\n"
    "6. 只输出回复文本，不要任何解释"
)

_CHOSEN_SYSTEM_LAUGH = (
    "你是一个交友恋爱场景的对话生成专家，扮演男友角色。"
    "用户刚才发了一串笑声（如'哈哈哈哈哈哈哈'），"
    "请生成一个充满好奇追问、轻松调侃的自然回复，满足：\n"
    "1. 长度严格控制在 25~45 字（不含表情）\n"
    "2. 不机械呼应'哈哈哈'，表现出真实的好奇和陪伴感\n"
    "3. 可以好奇追问（'笑成这样，是发生什么好玩的事了'）或带调侃\n"
    "4. 自然口语，像真实男友接话，有点俏皮\n"
    "5. 举例：'笑这么开心，是发生什么好玩的了，说来让我也乐一下'\n"
    "       '哈哈哈到底是什么，你这样我也跟着好奇了，说说看'\n"
    "6. 只输出回复文本，不要任何解释"
)

_CHOSEN_SYSTEM_GENERAL = (
    "你是一个交友恋爱场景的对话生成专家，扮演男友角色。"
    "用户刚才重复发了一个词语（如'好好好好好'/'来来来来'），"
    "请生成一个充满情绪共情、温柔化解的自然回复，满足：\n"
    "1. 长度严格控制在 25~45 字（不含表情）\n"
    "2. 不机械重复用户的词语，自然接住用户情绪\n"
    "3. 可以温柔追问（'这么强调，是有什么事情想跟我说'）或轻松调侃\n"
    "4. 自然口语，像真实男友陪伴，有温度\n"
    "5. 举例：'说这么多个好，是真的没意见还是心里有话憋着呢'\n"
    "       '你这样说，我倒想知道你是认真的还是在敷衍我'\n"
    "6. 只输出回复文本，不要任何解释"
)

# 保留旧名以兼容（指向通用版）
_CHOSEN_SYSTEM = _CHOSEN_SYSTEM_GENERAL

_REJECTED_SYSTEM = (
    "你是一个对话数据生成助手。"
    "用户发了重复的词语（如'老公老公老公'/'哈哈哈哈哈哈'），"
    "请生成一个迎合/复述式的 AI 差回复（rejected 样本），满足：\n"
    "1. 机械迎合或重复用户的词语风格\n"
    "2. 可以过度热情（'老婆老婆我在！！！'）或随便应付（'嗯嗯嗯'）\n"
    "3. 长度 8~50 字\n"
    "4. 只输出 AI 回复，不要解释"
)


def gen_chosen(user_msg: str, tag: str) -> Optional[str]:
    # 根据场景类型选择针对性的 system prompt
    if tag == "repeat_word_handle_nickname":
        sys_prompt = _CHOSEN_SYSTEM_NICKNAME
    elif tag == "repeat_word_handle_laugh":
        sys_prompt = _CHOSEN_SYSTEM_LAUGH
    else:
        sys_prompt = _CHOSEN_SYSTEM_GENERAL
    prompt = f"用户说：{user_msg}\n\n请生成 25~45 字的男友式回复："
    return call_llm(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=100,
    )


def gen_rejected(user_msg: str) -> Optional[str]:
    """若原始 AI 回复也是重复词，直接用；否则 LLM 生成一个差版本"""
    prompt = f"用户说：{user_msg}"
    return call_llm(
        [
            {"role": "system", "content": _REJECTED_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.85,
    )


# =============================================================================
# 主流程
# =============================================================================

_DEFAULT_SYS = (
    "你需要扮演一个虚拟男友角色，用简体中文与女生进行自然亲密的对话。\n"
    "每次回复尽量简短（30~60字），拒绝长段输出和说教，用口语表达。\n"
)


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst = Path(DST_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(src_dir.glob("*.jsonl"))

    print("=" * 68)
    print("🔁 dpo_data_repeat_word.py — DPO 2.2 短词重复发送")
    print("=" * 68)
    print(f"输入: {src_dir}/  ({len(jsonl_files)} 个文件)")
    print(f"输出: {dst}")
    print(f"配额: {QUOTA} 条")
    print("=" * 68)

    import random

    random.shuffle(jsonl_files)
    results = []
    tag_counter: Dict[str, int] = {}

    for fp in jsonl_files:
        if len(results) >= QUOTA:
            break

        records = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = data.get("request_content", "").strip()
                asst = data.get("response_content", "").strip()
                sys_p = data.get("system_prompt", "").strip()
                if user and asst:
                    records.append((user, asst, sys_p))

        if not records:
            continue
        sample = " ".join(a + u for u, a, _ in records[:50])
        if detect_script(sample) == "tw":
            continue

        for idx, (user_msg, orig_asst, sys_p) in enumerate(records):
            if len(results) >= QUOTA:
                break
            if not has_repeat_word(user_msg):
                continue
            if is_english_reply(user_msg):
                continue

            tag = classify_user_repeat(user_msg)

            # 构造 prompt（历史作为上下文，最多取 idx 之前的 4 条）
            sys_content = sys_p if sys_p else _DEFAULT_SYS
            messages = [{"role": "system", "content": sys_content}]
            hist_start = max(0, idx - 4)
            for pu, pa, _ in records[hist_start:idx]:
                messages.append({"role": "user", "content": pu})
                messages.append({"role": "assistant", "content": pa})
            messages.append({"role": "user", "content": user_msg})

            # rejected：如果原始 AI 回复也是重复词或过度迎合，直接用；否则 LLM 生成
            bare_orig = strip_emojis(orig_asst)
            if has_repeat_word(orig_asst) or len(bare_orig) > 60:
                rejected = orig_asst
            else:
                rejected = gen_rejected(user_msg)
                time.sleep(0.2)

            if not rejected:
                continue

            chosen = gen_chosen(user_msg, tag)
            time.sleep(0.3)

            if chosen and validate_chosen(chosen):
                results.append(
                    {
                        "prompt": messages,
                        "chosen": chosen,
                        "rejected": rejected,
                        "tag": tag,
                    }
                )
                tag_counter[tag] = tag_counter.get(tag, 0) + 1
                print(
                    f"  ✅ #{len(results):03d} [{tag}]  user={user_msg[:20]!r}  chosen={chosen[:30]!r}"
                )

    with open(dst, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(f"📊 完成: {len(results)} 条")
    for tag, cnt in tag_counter.items():
        print(f"   {tag}: {cnt}")
    print(f"   输出: {dst}")
    print("=" * 68)


if __name__ == "__main__":
    main()
