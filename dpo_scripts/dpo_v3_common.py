#!/usr/bin/env python3
"""dpo_v3_common.py — v3 DPO 生成脚本公共工具。"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dpo_data_intent_clarity import _DEFAULT_SYS, _SYS_POOL
except ImportError:
    from dpo_scripts.dpo_data_intent_clarity import _DEFAULT_SYS, _SYS_POOL


REPO_ROOT = Path(__file__).resolve().parent.parent
_TIME_LINE_RE = re.compile(r"当前时间：[^\n]+")
_ASTRAL_RE = re.compile(r"[\U00010000-\U0010ffff]")


def bare_len(text: str) -> int:
    return len(_ASTRAL_RE.sub("", text).strip())


def sample_sys_prompt() -> str:
    return random.choice(_SYS_POOL) if _SYS_POOL else _DEFAULT_SYS


def sample_sys_prompt_with_time(day_name: str, period: str) -> str:
    prompt = sample_sys_prompt()
    time_line = f"当前时间：{day_name}， {period}"
    if "当前时间：" in prompt:
        return _TIME_LINE_RE.sub(time_line, prompt, count=1)
    return f"{prompt.rstrip()}\n\n## 时间信息\n{time_line}"


def build_prompt(
    turns: Sequence[Tuple[str, str]],
    last_user: str,
    sys_content: Optional[str] = None,
) -> List[Dict[str, str]]:
    prompt = [{"role": "system", "content": sys_content or sample_sys_prompt()}]
    for user_text, assistant_text in turns:
        prompt.append({"role": "user", "content": user_text})
        if assistant_text:
            prompt.append({"role": "assistant", "content": assistant_text})
    prompt.append({"role": "user", "content": last_user})
    return prompt


def sample_key(sample: Dict) -> str:
    prompt = json.dumps(sample["prompt"], ensure_ascii=False, sort_keys=True)
    return f"{sample['tag']}||{prompt}||{sample['chosen']}||{sample['rejected']}"


def validate_text(
    text: str,
    *,
    min_len: int = 8,
    max_len: int = 90,
    required_words: Optional[Iterable[str]] = None,
    forbidden_words: Optional[Iterable[str]] = None,
) -> bool:
    if not (min_len <= bare_len(text) <= max_len):
        return False
    if re.search(r"\*[^*]+\*", text):
        return False
    if required_words and not all(word in text for word in required_words):
        return False
    if forbidden_words and any(word in text for word in forbidden_words):
        return False
    return True


def write_jsonl(samples: Sequence[Dict], dst_file: Path) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
