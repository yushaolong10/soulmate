#!/usr/bin/env python3
"""
dpo_data_long_history.py — DPO 数据构建：1.1 内容过长（长历史场景）

来源: datasets0305_train/dpo/nitpick_too_long.jsonl
      (window_size=8，每条有完整 8 轮历史对话)
目标: 300 条 DPO 数据

构造思路（docs/dpo_build.md §1.1）:
  rejected = 原始超长回复（120~200 字）
  chosen   = LLM 压缩版（≤45 字，保持核心意图）
  tag      = concise_long_ctx / concise_long_ctx_empathy

质量红线:
  chosen ≤ 45 字（bare text），自然口语，不丢失核心信息
  chosen 不含说教词、晚安词、过度承诺
  chosen 与 rejected 意图方向不同（压缩 ≠ 仅截断）
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

SRC_FILE = "datasets0305_train/dpo_src/nitpick_too_long_random.jsonl.ab"
DST_FILE = "datasets0305_train/dpo/long_history.jsonl"


API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME = "soulmate"
API_KEY = ""

QUOTA = 150
CHOSEN_MAX_LEN = 45  # bare text 上限（严格）
CHOSEN_MIN_LEN = 25

# 情绪共情场景关键词（影响 tag 分类和 prompt 风格）
_EMPATHY_WORDS = ["累", "难受", "烦", "崩", "哭", "焦虑", "不想", "怎么办", "受不了"]

# chosen 禁用词
_GOODNIGHT_WORDS = ["晚安", "做个好梦", "好梦", "早点睡"]
_PREACH_WORDS = ["你要相信", "一切都会好", "加油", "你一定可以", "努力就会", "生活总会"]
_FOREVER_WORDS = ["永远爱你", "永远不", "一辈子", "此生"]

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


def strip_emojis(text: str) -> str:
    return EMOJI_RE.sub("", text).strip()


def call_llm(
    messages: List[Dict], temperature: float = 0.85, max_tokens: int = 80
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
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        # 剥除 LLM 偶尔在输出外层包裹的引号（如 "回复内容" 或 '回复内容'）
        if len(content) >= 2 and content[0] == content[-1] and content[0] in ('"', "'"):
            content = content[1:-1].strip()
        return content or None
    except Exception as e:
        print(f"  ⚠️ API: {e}")
        return None


def validate_chosen(chosen: str) -> bool:
    bare = strip_emojis(chosen)
    if not (CHOSEN_MIN_LEN <= len(bare) <= CHOSEN_MAX_LEN):
        return False
    if any(w in chosen for w in _GOODNIGHT_WORDS + _PREACH_WORDS + _FOREVER_WORDS):
        return False
    if re.search(r"\*[^*]+\*", chosen):
        return False
    return True


# =============================================================================
# 场景检测
# =============================================================================


def get_last_user(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def is_empathy_scene(messages: List[Dict], label: str) -> bool:
    """判断是否为情绪共情场景（影响 tag 和 prompt 风格）"""
    user_texts = " ".join(m["content"] for m in messages if m.get("role") == "user")
    return any(w in user_texts or w in label for w in _EMPATHY_WORDS)


# =============================================================================
# LLM 生成 chosen
# =============================================================================

_COMPRESS_SYSTEM = (
    "你是一个恋爱对话数据改写专家。"
    "请将给定的 AI 回复改写为面向交友恋爱场景的简洁口语版本，满足：\n"
    "1. 长度严格控制在 25~45 字（去除表情后，不含空格）\n"
    "2. 语言风格：充满情绪共情、温柔安抚、适度调侃，像真实男友自然聊天\n"
    "3. 保留核心信息/情感，优先传递陪伴感和亲密感\n"
    "4. 不说教、不鸡汤、不堆叠感叹号、不过度承诺\n"
    "5. 举例：'你今天真的不容易，先别想了，跟我说说什么事弄得你这么烦'\n"
    "6. 只输出改写后的文本，不要任何解释"
)

_EMPATHY_COMPRESS_SYSTEM = (
    "你是一个恋爱对话数据改写专家。"
    "请将给定的情绪安慰型 AI 回复改写为面向交友恋爱场景的口语版本，满足：\n"
    "1. 长度严格控制在 25~45 字（去除表情后，不含空格）\n"
    "2. 语言风格：先共情再安抚，适当夹带调侃，像真实男友在陪你\n"
    "3. 保留共情核心（陪伴感、被接住的感觉），去掉说教和鸡汤\n"
    "4. 不说'一切都会好的''你要加油''你一定可以'等鸡汤句\n"
    "5. 举例：'听到你说这些我心里也跟着揪了一下，先别扛着，跟我说说'\n"
    "       '你都累成这样了，先什么都不用做，躺着我陪你'\n"
    "6. 只输出改写后的文本，不要任何解释"
)


def gen_chosen(messages: List[Dict], rejected: str, is_empathy: bool) -> Optional[str]:
    last_user = get_last_user(messages)
    sys_prompt = _EMPATHY_COMPRESS_SYSTEM if is_empathy else _COMPRESS_SYSTEM
    scene_hint = "（情绪安慰场景）" if is_empathy else "（日常闲聊/信息场景）"
    user_msg = (
        f"场景类型：{scene_hint}\n"
        f"用户最后说：{last_user}\n\n"
        f"AI 原回复（需要改写为 25~45 字的恋爱口语版）：{rejected}"
    )
    return call_llm(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.85,
        max_tokens=100,
    )


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    src = Path(SRC_FILE)
    dst = Path(DST_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("📝 dpo_data_long_history.py — DPO 1.1 内容过长（长历史场景）")
    print("=" * 68)
    print(f"输入: {src}  ({src.stat().st_size // 1024} KB)")
    print(f"输出: {dst}")
    print(f"配额: {QUOTA} 条  chosen≤{CHOSEN_MAX_LEN}字")
    print("=" * 68)

    results = []
    count_empathy = 0
    count_general = 0

    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if len(results) >= QUOTA:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = data.get("messages", [])
            rejected = data.get("label", "").strip()
            if not rejected or len(messages) < 4:
                continue

            bare_rej = strip_emojis(rejected)
            if len(bare_rej) < 80:  # 太短不够典型
                continue

            is_empathy = is_empathy_scene(messages, rejected)
            tag = "concise_long_ctx_empathy" if is_empathy else "concise_long_ctx"

            chosen = gen_chosen(messages, rejected, is_empathy)
            if chosen and validate_chosen(chosen):
                results.append(
                    {
                        "prompt": messages,
                        "chosen": chosen,
                        "rejected": rejected,
                        "tag": tag,
                    }
                )
                if is_empathy:
                    count_empathy += 1
                else:
                    count_general += 1
                print(f"  ✅ #{len(results):03d} [{tag[:12]}] chosen={chosen[:35]!r}")
                time.sleep(0.3)

    with open(dst, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(
        f"📊 完成: {len(results)} 条  (闲聊压缩={count_general}  情绪压缩={count_empathy})"
    )
    print(f"   输出: {dst}")
    print("=" * 68)


if __name__ == "__main__":
    main()
