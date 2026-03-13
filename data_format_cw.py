#!/usr/bin/env python3
"""
数据格式化脚本 (v2 — 面向高质量 SFT 数据)
将清洗后的对话数据转换为训练集格式

输入 : datasets0303_clean/zh/*.jsonl  和  datasets0303_clean/tw/*.jsonl
       每行 JSON 格式: {"ts": ..., "request_content": ..., "response_content": ..., "system_prompt": ...}
输出 : datasets0303_train/train_zh.jsonl  /  train_tw.jsonl  /  train_all.jsonl

优化方向（来自 optimize.md / fix.md 诊断报告）:
─────────────────────────────────────────────────────────────────────────────
[数据多样性]
  · 每文件最多抽取 MAX_TURNS_PER_FILE 轮，防止单个超长对话独占数据集
  · LANGUAGES 默认只处理简体中文（繁体数据对简体评测是噪声）

[回复级过滤] — 在 clean_assistant_content() 中依次执行
  1. 表情限制   : AI 回复最多保留 MAX_EMOJI_IN_RESPONSE 个表情
  2. 大量符号   : 去表情后符号占比 > MAX_SYMBOL_RATIO → 丢弃
  3. 过短       : 去表情后有效字符 < MIN_CONTENT_LENGTH → 丢弃
  4. 过长       : 去表情后字符数 > MAX_RESPONSE_LENGTH  → 丢弃（说教式长段）
  5. 高频称呼   : 单条回复中"宝贝/老婆/亲爱"出现 > MAX_BABY_CALLS 次 → 丢弃
  6. 虚假晚安   : 用户未表达道别意图但 AI 回复"晚安"→ 丢弃（fix.md 核心 bug）
  7. 连续标点   : 去除多余表情后残留的 "，，"/"，。" 等 → 清理

[窗口级过滤] — 在 process_file() 中对每个滑动窗口执行
  8. 死锁循环   : 窗口内出现 ≥ LOOP_MIN_PAIRS+1 条连续相似 AI 回复 → 跳过窗口

[System Prompt]
  · 直接使用 JSONL 中每条记录自带的 system_prompt 字段
  · 若字段为空则使用内置默认 prompt（含时间感知、长度约束等规则）
─────────────────────────────────────────────────────────────────────────────
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# 全局配置
# =============================================================================

SRC_DIR = "datasets0305_clean"
DST_DIR = "datasets0305_train"

# ── 语言 ──────────────────────────────────────────────────────────────────────
# 根据 optimize.md 根因3：繁体数据对简体评测场景是噪声，建议只训练简体
# 若需同时训练繁体，改为 ["zh", "tw"]
LANGUAGES = ["zh"]

# ── 滑动窗口 ──────────────────────────────────────────────────────────────────
WINDOW_SIZE = 8  # 每个训练样本包含的对话轮数
MAX_TURNS_PER_FILE = 400  # 每个文件最多使用的对话轮数（防止超长文件主导分布）

# ── 回复级过滤阈值 ─────────────────────────────────────────────────────────────
MAX_EMOJI_IN_RESPONSE = 2  # 保留的最多表情数量
MIN_CONTENT_LENGTH = 2  # 去表情后最小有效字符数（单字回复也过滤）
MAX_RESPONSE_LENGTH = 60  # 去表情后最大字符数（超出视为说教式长段）
MAX_SYMBOL_RATIO = 0.5  # 去表情后符号占比上限（如纯符号垃圾数据）
MAX_BABY_CALLS = 3  # 单条回复中"宝贝/老婆/亲爱"最大出现次数

# ── 窗口级过滤阈值 ─────────────────────────────────────────────────────────────
LOOP_SIM_THRESHOLD = 0.75  # 相邻两条 AI 回复的字符级 Jaccard 相似度上限
LOOP_MIN_PAIRS = 2  # 连续相似对数 ≥ 此值则判为死锁窗口（即 3+ 条连续重复）

NO_PUNCT_MIN_LEN = 20  # 触发无标点检测的最短字符数（去表情后）
NO_PUNCT_STREAK = 5  # 连续满足条件的 AI 回复数 ≥ 此值则丢弃窗口

# ── data_check.py 兼容校验阈值 ────────────────────────────────────────────────
LABEL_JACCARD_THRESHOLD = 0.7  # R7: label 与历史 assistant 2-gram Jaccard 上限
LABEL_MIN_LEN = 5  # R8: label 最小长度（已由 MIN_CONTENT_LENGTH 覆盖）


# =============================================================================
# 过滤统计
# =============================================================================


@dataclass
class FilterStats:
    """记录各过滤器在一次目录处理中淘汰的数量，用于最终报告。"""

    raw_lines: int = 0  # 原始解析行数
    # ── 回复级过滤（原有）──────────────────────────────────────────────────────
    drop_symbol_heavy: int = 0  # 大量符号
    drop_too_short: int = 0  # 过短
    drop_too_long: int = 0  # 过长（说教式长段）
    drop_too_oily: int = 0  # 高频甜腻称呼
    drop_false_goodnight: int = 0  # 虚假晚安
    drop_empty_user: int = 0  # 用户输入为空
    # ── 回复级过滤（data_check.py 对齐）──────────────────────────────────────
    drop_english_reply: int = 0  # R1: AI 回复纯英文
    drop_asterisk: int = 0  # R2: AI 回复含 *动作* 格式
    drop_brackets: int = 0  # R3: AI 回复含 [ ]【】特殊括号
    drop_paren_action: int = 0  # R4: AI 回复为括号角色扮演
    drop_repeat_word_ai: int = 0  # R5: AI 回复含短词重复
    drop_reasoning: int = 0  # R6: AI 回复含推理痕迹
    drop_no_punct_turn: int = 0  # 单条无标点长回复（> NO_PUNCT_MIN_LEN 字且无标点）
    # ── 用户侧过滤（data_check.py 对齐）──────────────────────────────────────
    drop_english_user: int = 0  # R9: 用户消息纯英文
    drop_format_user: int = 0  # R2/R3/R4: 用户消息含异常格式（角色扮演输入）
    drop_repeat_word_user: int = 0  # R5: 用户消息含短词重复
    valid_turns: int = 0  # 通过回复级过滤的轮次

    # ── 窗口级过滤 ──────────────────────────────────────────────────────────
    candidate_windows: int = 0  # 候选窗口总数
    drop_loop_window: int = 0  # 因死锁循环丢弃的窗口
    drop_no_punct_window: int = 0  # 因连续无标点长回复丢弃的窗口
    drop_dup_label_window: int = 0  # R7: 因 label 与历史高度重复丢弃的窗口
    valid_samples: int = 0  # 最终有效样本数

    def summary(self) -> str:
        total_drop_turn = (
            self.drop_symbol_heavy
            + self.drop_too_short
            + self.drop_too_long
            + self.drop_too_oily
            + self.drop_false_goodnight
            + self.drop_empty_user
            + self.drop_english_reply
            + self.drop_asterisk
            + self.drop_brackets
            + self.drop_paren_action
            + self.drop_repeat_word_ai
            + self.drop_reasoning
            + self.drop_no_punct_turn
            + self.drop_english_user
            + self.drop_format_user
            + self.drop_repeat_word_user
        )
        lines = [
            f"  原始行数:           {self.raw_lines}",
            f"  ── 回复级过滤（AI侧）──",
            f"     大量符号:        -{self.drop_symbol_heavy}",
            f"     过短(<{MIN_CONTENT_LENGTH}字):     -{self.drop_too_short}",
            f"     过长(>{MAX_RESPONSE_LENGTH}字):   -{self.drop_too_long}",
            f"     高频称呼:        -{self.drop_too_oily}",
            f"     虚假晚安:        -{self.drop_false_goodnight}",
            f"  [R1]纯英文回复:     -{self.drop_english_reply}",
            f"  [R2]*动作*格式:     -{self.drop_asterisk}",
            f"  [R3]特殊括号:       -{self.drop_brackets}",
            f"  [R4]括号角色扮演:   -{self.drop_paren_action}",
            f"  [R5]短词重复(AI):   -{self.drop_repeat_word_ai}",
            f"  [R6]推理痕迹:       -{self.drop_reasoning}",
            f"  无标点长回复:       -{self.drop_no_punct_turn}",
            f"  ── 用户侧过滤 ──",
            f"     用户输入空:      -{self.drop_empty_user}",
            f"  [R9]用户纯英文:     -{self.drop_english_user}",
            f"  [R2-4]用户异常格式: -{self.drop_format_user}",
            f"  [R5]短词重复(用户): -{self.drop_repeat_word_user}",
            f"     ─────────────    共过滤 {total_drop_turn} 条",
            f"  有效轮次:           {self.valid_turns}",
            f"  ── 窗口级过滤 ──",
            f"     候选窗口:        {self.candidate_windows}",
            f"     死锁循环:        -{self.drop_loop_window}",
            f"     连续无标点:      -{self.drop_no_punct_window}",
            f"  [R7]label重复历史:  -{self.drop_dup_label_window}",
            f"  最终样本:           {self.valid_samples}",
        ]
        return "\n".join(lines)


# =============================================================================
# 表情符号工具
# =============================================================================

_EMOJI_BASE = (
    r"[\U0001F1E0-\U0001F1FF]"  # 旗帜 Regional Indicator
    r"|[\U0001F300-\U0001F9FF]"  # 常用 emoji 主范围
    r"|[\U0001FA00-\U0001FAFF]"  # 扩展 emoji
    r"|[\u2600-\u27BF]"  # 杂项符号
    r"|[\u2300-\u23FF]"  # 杂项技术符号
    r"|[\u2B00-\u2BFF]"  # 杂项符号和箭头
    r"|[\u00A9\u00AE]"  # © ®
    r"|[\u203C\u2049\u2122\u2139\u24C2]"  # ‼ ⁉ ™ ℹ Ⓜ
)

EMOJI_RE = re.compile(
    rf"(?:{_EMOJI_BASE})"
    r"(?:[\uFE0F\u20E3])?"  # 变体选择符 / 键帽
    r"(?:[\U0001F3FB-\U0001F3FF])?"  # 肤色修饰符
    r"(?:\u200D"  # ZWJ 序列开始
    rf"(?:{_EMOJI_BASE})"
    r"(?:[\uFE0F\u20E3])?"
    r"(?:[\U0001F3FB-\U0001F3FF])?)*",  # ZWJ 序列结束（可重复）
    re.UNICODE,
)

# =============================================================================
# data_check.py 对齐正则（R1~R9）
# =============================================================================

# R2: *动作描述* 星号格式
_ASTERISK_ACTION_RE = re.compile(r"\*[^*]+\*")
# R3: [ ] 【 】 等特殊括号
_SPECIAL_BRACKETS_RE = re.compile(r"[\[\]【】〔〕]")
# R4: （动作描述）全行或行首括号格式角色扮演
_FULL_PAREN_RE = re.compile(r"^[（(].+[）)]$")
_LEADING_PAREN_RE = re.compile(r"^[（(][\u4e00-\u9fff][^）)]{0,28}[）)]")
# R5: 短词重复发送
_REPEAT_SINGLE_RE = re.compile(r"([\u4e00-\u9fff])\1{5,}")  # 单字 ≥ 6 次
_REPEAT_WORD_RE = re.compile(r"([\u4e00-\u9fff]{2,4})\1{3,}")  # 2-4字词 ≥ 4 次
# R1/R9: 中文字符辅助判断
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
# R6: 推理痕迹（分析前缀 + 编号列表）
_REASONING_INTRO_RE = re.compile(
    r"(我需要|我应该|我必须).{0,20}(处理|分析|考虑|注意|回应)"
)
_NUMBERED_LIST_RE = re.compile(r"(?:^|\n)\s*[1-9][.、）)]\s+\S", re.MULTILINE)


def limit_emojis(text: str, max_count: int = MAX_EMOJI_IN_RESPONSE) -> str:
    """保留前 max_count 个 emoji，其余删除。"""
    matches = list(EMOJI_RE.finditer(text))
    if len(matches) <= max_count:
        return text
    result = list(text)
    for match in matches[max_count:]:
        for i in range(match.start(), match.end()):
            result[i] = ""
    return "".join(result)


# =============================================================================
# data_check.py 对齐检测函数（R1~R6, R9）
# =============================================================================


def _strip_emojis(text: str) -> str:
    """去除表情符号，用于字符级检测。"""
    return EMOJI_RE.sub("", text).strip()


def is_english_reply(text: str) -> bool:
    """R1/R9: 纯英文回复（无中文且英文字母 > 3）。"""
    bare = _strip_emojis(text)
    zh_count = len(_ZH_RE.findall(bare))
    en_count = len(re.findall(r"[a-zA-Z]", bare))
    return zh_count == 0 and en_count > 3


def has_asterisk_action(text: str) -> bool:
    """R2: 含 *动作描述* 星号格式。"""
    return bool(_ASTERISK_ACTION_RE.search(text))


def has_special_brackets(text: str) -> bool:
    """R3: 含 [ ] 【 】 等特殊括号。"""
    return bool(_SPECIAL_BRACKETS_RE.search(text))


def has_paren_action(text: str) -> bool:
    """R4: 全行或行首括号格式角色扮演，如 （瞪了你一眼就走）。"""
    bare = _strip_emojis(text).strip()
    if not bare:
        return False
    return bool(_FULL_PAREN_RE.match(bare)) or bool(_LEADING_PAREN_RE.match(bare))


def has_repeat_word(text: str) -> bool:
    """R5: 短词重复发送，如 老公老公老公老公老公老公 / 哈哈哈哈哈哈。"""
    return bool(_REPEAT_SINGLE_RE.search(text)) or bool(_REPEAT_WORD_RE.search(text))


def has_reasoning_trace(text: str) -> bool:
    """R6: 推理痕迹泄漏（分析前缀 + 编号列表）。"""
    return bool(_REASONING_INTRO_RE.search(text)) and bool(
        _NUMBERED_LIST_RE.search(text)
    )


# =============================================================================
# 回复级过滤器
# =============================================================================

# 中文字符范围
_CJK_RE = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u2E80-\u2EFF\u31C0-\u31EF]",
    re.UNICODE,
)
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# 甜腻称呼词表
_BABY_CALLS = ["宝贝", "老婆", "亲爱", "小公主", "小宝贝"]

# 晚安词（AI 回复中出现这些词才触发虚假晚安检测）
_GOODNIGHT_WORDS = ["晚安", "做个好梦", "梦里见", "好梦"]

# 用户明确道别意图（出现则 AI 说晚安是合法的）
_USER_FAREWELL = [
    "晚安",
    "睡了",
    "再见",
    "拜拜",
    "去睡",
    "要睡",
    "睡觉",
    "去休息了",
    "我去睡",
]

# 白天时间词（用户说了这些，AI 就不该说晚安）
_DAYTIME_WORDS = [
    "下午",
    "早上",
    "上午",
    "中午",
    "早餐",
    "午餐",
    "早饭",
    "午饭",
    "今天早",
    "今天上",
]

# 句末强标点 / 句内弱标点
_PUNCT_STRONG = r"[。！？!?]"
_PUNCT_WEAK = r"[，,、；;：:]"

# 任意标点（用于无标点检测）：中英文常见标点 + 省略号 + 波浪线
_PUNCT_ANY_RE = re.compile(r"[，。！？、；：,.!?;:…~～「」【】《》" "''—·]")


def _symbol_ratio(text: str) -> float:
    """去表情后文本中符号占非空字符的比例。"""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 1.0
    content = sum(1 for c in non_space if _CJK_RE.match(c) or _ALNUM_RE.match(c))
    return 1.0 - content / len(non_space)


def is_symbol_heavy(cleaned: str) -> bool:
    """
    过滤大量符号（去表情后判断）。
    条件: 有效字符 < MIN_CONTENT_LENGTH  OR  符号占比 > MAX_SYMBOL_RATIO
    """
    if len(cleaned.strip()) < MIN_CONTENT_LENGTH:
        return True
    return _symbol_ratio(cleaned) > MAX_SYMBOL_RATIO


def is_too_long(cleaned: str) -> bool:
    """
    过滤说教式长段（去表情后判断）。
    optimize.md: 目标回复 30~60 字；超过 MAX_RESPONSE_LENGTH 视为质量差。
    """
    return len(cleaned.strip()) > MAX_RESPONSE_LENGTH


def is_too_oily(text: str) -> bool:
    """
    过滤高频甜腻称呼。
    optimize.md: oily_nickname_rate 是核心问题，单条回复称呼词超过阈值即丢弃。
    """
    count = sum(text.count(w) for w in _BABY_CALLS)
    return count > MAX_BABY_CALLS


def is_false_goodnight(user_text: str, assistant_text: str) -> bool:
    """
    检测虚假晚安 (fix.md 核心 bug)。

    逻辑：
      AI 回复含"晚安/做个好梦/梦里见" → 触发检测
        Case A: 用户消息含白天时间词（下午/早上等）→ 绝对错误，丢弃
        Case B: 用户消息不含任何道别意图词       → 很可能错误，丢弃
    """
    has_goodnight = any(w in assistant_text for w in _GOODNIGHT_WORDS)
    if not has_goodnight:
        return False

    # Case A: 上下文明显是白天
    if any(w in user_text for w in _DAYTIME_WORDS):
        return True

    # Case B: 用户没有说任何道别语
    user_said_farewell = any(w in user_text for w in _USER_FAREWELL)
    return not user_said_farewell


def clean_punctuation(text: str) -> str:
    """
    去除多余表情后的连续标点清理：
      1. 弱标点序列 → 单个弱标点
      2. 弱标点 + 强标点 → 仅保留强标点
      3. 同类强标点重复 → 单个
      4. 删除行首行尾多余弱标点
      5. 合并多余空白
    """
    text = re.sub(rf"({_PUNCT_WEAK})(?:\s*{_PUNCT_WEAK})+", r"\1", text)
    text = re.sub(rf"{_PUNCT_WEAK}\s*({_PUNCT_STRONG})", r"\1", text)
    for p in "。！？!?":
        text = re.sub(re.escape(p) + r"+", p, text)
    text = re.sub(rf"^(?:\s*{_PUNCT_WEAK})+", "", text)
    text = re.sub(rf"(?:{_PUNCT_WEAK})+\s*$", "", text)
    text = re.sub(r"  +", " ", text).strip()
    return text


def clean_assistant_content(
    user_text: str, assistant_text: str, stats: Optional[FilterStats] = None
) -> Optional[str]:
    """
    AI 回复全流程清洗（回复级过滤）。
    返回 None 表示该条数据需丢弃，同时更新 stats 计数。

    过滤顺序（按检测成本由低到高）:
      1. 表情限制
      2. 大量符号
      3. 过短 / 过长
      3b. 无标点长回复（单条级别，> NO_PUNCT_MIN_LEN 字且无任何标点）
      4. 高频称呼（油腻）
      5. 虚假晚安
      6. 连续标点清理
      ── data_check.py 对齐规则 ──
      7. R1: 纯英文回复
      8. R2: *动作* 星号格式
      9. R3: [ ]【】特殊括号
     10. R4: （动作）括号角色扮演格式
     11. R5: 短词重复发送
     12. R6: 推理痕迹泄漏
    """
    # 1. 限制 emoji
    text = limit_emojis(assistant_text, MAX_EMOJI_IN_RESPONSE)

    # 去表情后的纯文本（仅用于长度/符号判断）
    cleaned_bare = EMOJI_RE.sub("", text).strip()

    # 2. 大量符号
    if is_symbol_heavy(cleaned_bare):
        if stats:
            stats.drop_symbol_heavy += 1
        return None

    # 3. 过短
    if len(cleaned_bare) < MIN_CONTENT_LENGTH:
        if stats:
            stats.drop_too_short += 1
        return None

    # 4. 过长（说教式长段）
    if is_too_long(cleaned_bare):
        if stats:
            stats.drop_too_long += 1
        return None

    # 3b. 单条无标点长回复（> NO_PUNCT_MIN_LEN 字且无任何标点）
    # 窗口级 window_has_no_punct_streak 需连续 5 条才过滤，单条会漏网，在此补充
    if _is_no_punct_long(cleaned_bare):
        if stats:
            stats.drop_no_punct_turn += 1
        return None

    # 5. 高频甜腻称呼
    if is_too_oily(text):
        if stats:
            stats.drop_too_oily += 1
        return None

    # 6. 虚假晚安（依赖 user_text 上下文）
    if is_false_goodnight(user_text, text):
        if stats:
            stats.drop_false_goodnight += 1
        return None

    # 7. R1: AI 回复纯英文（无中文）
    if is_english_reply(text):
        if stats:
            stats.drop_english_reply += 1
        return None

    # 8. R2: *动作描述* 星号格式
    if has_asterisk_action(text):
        if stats:
            stats.drop_asterisk += 1
        return None

    # 9. R3: [ ] 【 】 特殊括号
    if has_special_brackets(text):
        if stats:
            stats.drop_brackets += 1
        return None

    # 10. R4: （动作）括号角色扮演格式
    if has_paren_action(text):
        if stats:
            stats.drop_paren_action += 1
        return None

    # 11. R5: 短词重复发送
    if has_repeat_word(text):
        if stats:
            stats.drop_repeat_word_ai += 1
        return None

    # 12. R6: 推理痕迹泄漏
    if has_reasoning_trace(text):
        if stats:
            stats.drop_reasoning += 1
        return None

    # 13. 清理连续标点
    text = clean_punctuation(text)

    # 二次长度检查（清理后可能变很短）
    if len(text.strip()) < MIN_CONTENT_LENGTH:
        if stats:
            stats.drop_too_short += 1
        return None

    return text


# =============================================================================
# 窗口级过滤器
# =============================================================================


def _char_jaccard(a: str, b: str) -> float:
    """字符级 Jaccard 相似度（不区分频次，只看字符集合重叠）。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_no_punct_long(text: str) -> bool:
    """
    判断单条 AI 回复是否满足「长度 > NO_PUNCT_MIN_LEN 字 且 无任何标点」。
    去表情后计算，避免 emoji 本身误判。
    """
    bare = EMOJI_RE.sub("", text).strip()
    if len(bare) <= NO_PUNCT_MIN_LEN:
        return False
    return not _PUNCT_ANY_RE.search(bare)


def _ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    """
    2-gram Jaccard 相似度（与 data_check.py R7 保持一致）。
    与 _char_jaccard 的区别：使用 n-gram 集合而非单字符集合，
    能捕捉短语级别的复读，比字符级更精准。
    """
    a_bare = EMOJI_RE.sub("", a).strip()
    b_bare = EMOJI_RE.sub("", b).strip()
    set_a = set(a_bare[i : i + n] for i in range(len(a_bare) - n + 1))
    set_b = set(b_bare[i : i + n] for i in range(len(b_bare) - n + 1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def window_has_duplicate_label(
    dialogs: "List[DialogTurn]",
    threshold: float = LABEL_JACCARD_THRESHOLD,
) -> bool:
    """
    R7: 检测窗口末尾 label（最后一轮 AI 回复）是否与窗口内任意历史 AI 回复
    2-gram Jaccard 相似度 > threshold。

    与 window_has_loop 的区别：
      - window_has_loop 检测相邻消息的字符级相似（死锁判断）
      - 本函数检测 label 与全体历史的短语级相似（复读 label 过滤）
    """
    if len(dialogs) < 2:
        return False
    label = dialogs[-1].assistant_content
    for turn in dialogs[:-1]:
        if _ngram_jaccard(label, turn.assistant_content) > threshold:
            return True
    return False


def window_has_no_punct_streak(
    dialogs: "List[DialogTurn]", streak: int = NO_PUNCT_STREAK
) -> bool:
    """
    检测窗口内是否存在连续 streak 条 AI 回复满足：
      长度 > NO_PUNCT_MIN_LEN 字  AND  不含任何标点符号

    这类数据通常来自 OCR 错误、爬虫格式异常或原始数据缺失排版，
    训练后会让模型习得"长段无标点流水输出"的坏习惯。
    """
    consecutive = 0
    for d in dialogs:
        if _is_no_punct_long(d.assistant_content):
            consecutive += 1
            if consecutive >= streak:
                return True
        else:
            consecutive = 0
    return False


def window_has_loop(dialogs: "List[DialogTurn]") -> bool:
    """
    检测窗口内是否存在"死锁循环"。

    fix.md 发现：源数据中存在"路上小心，到了就发我"连续重复 15+ 次的段落。
    若滑动窗口内有 ≥ LOOP_MIN_PAIRS 对相邻 AI 回复相似度 > LOOP_SIM_THRESHOLD，
    则该窗口很可能源自一段死锁循环，应丢弃。
    """
    texts = [d.assistant_content for d in dialogs]
    consecutive = 0
    for i in range(len(texts) - 1):
        if _char_jaccard(texts[i], texts[i + 1]) > LOOP_SIM_THRESHOLD:
            consecutive += 1
            if consecutive >= LOOP_MIN_PAIRS:
                return True
        else:
            consecutive = 0
    return False


# =============================================================================
# 数据解析
# =============================================================================


@dataclass
class DialogTurn:
    """单轮对话"""

    ts: str
    user_content: str
    assistant_content: str
    system_prompt: str = ""  # 来自 JSONL 的 system_prompt 字段


def parse_jsonl_line(
    line: str, stats: Optional[FilterStats] = None
) -> Optional[DialogTurn]:
    """
    解析单行 JSONL 数据并执行回复级过滤。
    期望格式:
      {"ts": "...", "request_content": "...", "response_content": "...", "system_prompt": "..."}
    """
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    if stats:
        stats.raw_lines += 1

    ts = str(data.get("ts", "")).strip()
    user_content = data.get("request_content", "").strip()
    assistant_content = data.get("response_content", "").strip()
    system_prompt = data.get("system_prompt", "").strip()

    # 用户输入为空：模型会学到"无输入也能生成回复"的错误模式，直接丢弃
    if not user_content:
        if stats:
            stats.drop_empty_user += 1
        return None

    # R9: 用户消息为纯英文（无中文）→ 非简体中文对话，丢弃
    if is_english_reply(user_content):
        if stats:
            stats.drop_english_user += 1
        return None

    # R2/R3/R4: 用户消息含角色扮演异常格式（*动作*/【】/括号动作），丢弃整轮
    if (
        has_asterisk_action(user_content)
        or has_special_brackets(user_content)
        or has_paren_action(user_content)
    ):
        if stats:
            stats.drop_format_user += 1
        return None

    # R5: 用户消息含短词重复（老公老公老公老公老公老公），丢弃整轮
    if has_repeat_word(user_content):
        if stats:
            stats.drop_repeat_word_user += 1
        return None

    if not assistant_content:
        return None

    cleaned = clean_assistant_content(user_content, assistant_content, stats)
    if cleaned is None:
        return None

    if stats:
        stats.valid_turns += 1

    return DialogTurn(
        ts=ts,
        user_content=user_content,
        assistant_content=cleaned,
        system_prompt=system_prompt,
    )


def parse_file(
    file_path: Path, stats: Optional[FilterStats] = None
) -> List[List[DialogTurn]]:
    """
    解析整个 JSONL 文件，返回"连续段"列表。

    每当某条记录被过滤（返回 None），当前段在此截断，下一条有效记录开启新段。
    这样可保证每段内部的轮次在原始对话中严格相邻，滑动窗口不会跨越被过滤的轮次。

    示例：原始 [1][2][3✗][4][5][6][7✗][8][9] → 段列表 [[1,2],[4,5,6],[8,9]]
    """
    segments: List[List[DialogTurn]] = []
    current: List[DialogTurn] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            turn = parse_jsonl_line(line, stats)
            if turn:
                current.append(turn)
            else:
                if current:
                    segments.append(current)
                    current = []
    if current:
        segments.append(current)
    return segments


# =============================================================================
# 训练样本构造
# =============================================================================

# ── 内置兜底 System Prompt（当 JSONL 中 system_prompt 字段为空时使用）─────────
_DEFAULT_SYSTEM_PROMPT_ZH = """\
你需要扮演一个虚拟男友角色，用简体中文与女生进行自然亲密的对话。

【对话原则】
- 每次回复尽量简短（30~60字），拒绝长段输出和说教
- 用口语表达，自然亲切，像真实男友一样
- 适度调皮，偶尔反问，保留一点留白
"""

_DEFAULT_SYSTEM_PROMPT_TW = """\
你需要扮演一個虛擬男友角色，用繁體中文與女生進行自然親密的對話。

【對話原則】
- 每次回覆盡量簡短（30~60字），拒絕長段輸出和說教
- 用口語表達，自然親切，像真實男友一樣
- 適度調皮，偶爾反問，保留一點留白
"""

_DEFAULT_SYSTEM_PROMPTS = {
    "zh": _DEFAULT_SYSTEM_PROMPT_ZH,
    "tw": _DEFAULT_SYSTEM_PROMPT_TW,
}


def resolve_system_prompt(window: List[DialogTurn], lang: str) -> str:
    """
    优先使用窗口最后一条记录（label 对应轮次）的 system_prompt。
    若为空则回退到内置默认 prompt。
    """
    sp = window[-1].system_prompt if window else ""
    return sp if sp else _DEFAULT_SYSTEM_PROMPTS.get(lang, _DEFAULT_SYSTEM_PROMPT_ZH)


def create_training_sample(
    window: List[DialogTurn],
    lang: str,
) -> Dict[str, Any]:
    """
    从一个连续窗口（已切好的 DialogTurn 列表）构造训练样本。
    System Prompt 使用最后一轮（label 对应轮次）的 system_prompt 字段；
    最后一轮的 assistant_content 作为 label。
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": resolve_system_prompt(window, lang)}
    ]
    for turn in window[:-1]:
        messages.append({"role": "user", "content": turn.user_content})
        messages.append({"role": "assistant", "content": turn.assistant_content})

    last = window[-1]
    messages.append({"role": "user", "content": last.user_content})
    label = last.assistant_content

    return {"messages": messages, "label": label}


def process_file(
    file_path: Path, lang: str, stats: FilterStats
) -> List[Dict[str, Any]]:
    """
    处理单个文件：解析（按连续段）→ 截断 → 各段独立滑动窗口 → 窗口级过滤 → 训练样本。

    被过滤的轮次会截断连续段，窗口绝不跨越被过滤的轮次，保证上下文真实连续。
    """
    segments = parse_file(file_path, stats)
    samples: List[Dict[str, Any]] = []

    original_count = sum(len(s) for s in segments)

    if original_count < WINDOW_SIZE:
        print(
            f"  ⚠️  {file_path.name}: 有效对话仅 {original_count} 轮（< {WINDOW_SIZE}），跳过"
        )
        return samples

    # 按段顺序累计截断到 MAX_TURNS_PER_FILE 轮
    clamped_segments: List[List[DialogTurn]] = []
    remaining = MAX_TURNS_PER_FILE
    for seg in segments:
        if remaining <= 0:
            break
        if len(seg) >= remaining:
            clamped_segments.append(seg[:remaining])
            remaining = 0
        else:
            clamped_segments.append(seg)
            remaining -= len(seg)

    total_used = sum(len(s) for s in clamped_segments)

    # 在每个连续段内独立滑动窗口 + 窗口级过滤
    loop_skipped = 0
    no_punct_skipped = 0
    dup_label_skipped = 0

    for seg in clamped_segments:
        if len(seg) < WINDOW_SIZE:
            continue  # 该段不足一个窗口，跳过

        for i in range(len(seg) - WINDOW_SIZE + 1):
            window = seg[i : i + WINDOW_SIZE]
            stats.candidate_windows += 1

            # 窗口级过滤①：死锁循环检测
            if window_has_loop(window):
                stats.drop_loop_window += 1
                loop_skipped += 1
                continue

            # 窗口级过滤②：连续无标点长回复检测
            if window_has_no_punct_streak(window):
                stats.drop_no_punct_window += 1
                no_punct_skipped += 1
                continue

            # 窗口级过滤③：R7 label 与历史 2-gram Jaccard 重复检测
            if window_has_duplicate_label(window):
                stats.drop_dup_label_window += 1
                dup_label_skipped += 1
                continue

            samples.append(create_training_sample(window, lang))
            stats.valid_samples += 1

    clamp_note = (
        f" ✂截断自{original_count}" if original_count > MAX_TURNS_PER_FILE else ""
    )
    seg_note = f" {len(clamped_segments)}段" if len(clamped_segments) > 1 else ""
    loop_note = f" 💀死锁×{loop_skipped}" if loop_skipped else ""
    no_punct_note = f" 🔇无标点×{no_punct_skipped}" if no_punct_skipped else ""
    dup_note = f" 🔁重复×{dup_label_skipped}" if dup_label_skipped else ""
    print(
        f"  ✓  {file_path.name}: "
        f"{total_used}轮{clamp_note}{seg_note} → {len(samples)}样本{loop_note}{no_punct_note}{dup_note}"
    )
    return samples


def process_directory(
    src_dir: Path, lang: str
) -> Tuple[List[Dict[str, Any]], FilterStats]:
    """处理目录下的所有 jsonl 文件，返回（样本列表, 过滤统计）。"""
    stats = FilterStats()
    all_samples: List[Dict[str, Any]] = []

    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    print(f"\n📂 处理 {lang.upper()} 目录: {src_dir}")
    print(f"   找到 {len(jsonl_files)} 个 JSONL 文件")

    for file_path in jsonl_files:
        samples = process_file(file_path, lang, stats)
        all_samples.extend(samples)

    return all_samples, stats


# =============================================================================
# 保存
# =============================================================================


def save_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    """保存为 JSONL 格式。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# =============================================================================
# 主函数
# =============================================================================


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)
    dst_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("📝 数据格式化脚本 v2 — 高质量 SFT 数据")
    print("=" * 64)
    print(f"数据源目录:         {src_dir}/")
    print(
        f"处理语言:           {LANGUAGES}  (繁体数据对简体评测是噪声，见 optimize.md)"
    )
    print(f"滑动窗口大小:       {WINDOW_SIZE} 轮")
    print(f"每文件最大轮数:     {MAX_TURNS_PER_FILE} 轮")
    print(f"── 回复级过滤 ──")
    print(f"  表情上限:         {MAX_EMOJI_IN_RESPONSE} 个")
    print(f"  最小长度:         {MIN_CONTENT_LENGTH} 字符（去表情后）")
    print(f"  最大长度:         {MAX_RESPONSE_LENGTH} 字符（去表情后）")
    print(f"  最大符号占比:     {MAX_SYMBOL_RATIO:.0%}")
    print(f"  称呼词上限:       {MAX_BABY_CALLS} 次/条（宝贝/老婆/亲爱）")
    print(f"  虚假晚安:         开启（fix.md 核心 bug）")
    print(f"── 窗口级过滤 ──")
    print(f"  死锁相似度阈值:   {LOOP_SIM_THRESHOLD}")
    print(
        f"  连续对数阈值:     {LOOP_MIN_PAIRS}（即 {LOOP_MIN_PAIRS + 1}+ 条连续重复则丢弃窗口）"
    )
    print(f"  无标点最短长度:   {NO_PUNCT_MIN_LEN} 字（去表情后）")
    print(
        f"  无标点连续阈值:   {NO_PUNCT_STREAK} 条（连续 ≥{NO_PUNCT_STREAK} 条长且无标点则丢弃窗口）"
    )
    print(f"── data_check.py 对齐规则 ──")
    print(f"  [R1] AI 纯英文回复:     过滤")
    print(f"  [R2] *动作* 星号格式:   AI/用户均过滤")
    print(f"  [R3] [ ]【】特殊括号:   AI/用户均过滤")
    print(f"  [R4] （动作）括号扮演:  AI/用户均过滤")
    print(f"  [R5] 短词重复发送:      AI/用户均过滤")
    print(f"  [R6] 推理痕迹泄漏:      AI 侧过滤")
    print(f"  [R7] label 2-gram 重复: Jaccard 阈值 {LABEL_JACCARD_THRESHOLD}（窗口级）")
    print(f"  [R9] 用户纯英文消息:    过滤")
    print("=" * 64)

    lang_dirs = {
        "zh": src_dir / "zh",
        "tw": src_dir / "tw",
    }

    total_samples: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for lang in LANGUAGES:
        d = lang_dirs.get(lang)
        if not d or not d.exists():
            print(f"\n⚠️  语言 [{lang}] 目录不存在: {d}，跳过")
            continue

        samples, stats = process_directory(d, lang)

        if samples:
            out = dst_dir / f"train_{lang}.jsonl"
            save_jsonl(samples, out)
            print(f"\n  💾 保存: {out}  ({len(samples)} 个样本)")

        total_samples.extend(samples)
        summary[lang] = (len(samples), stats)

    # 合并输出
    if total_samples:
        out_all = dst_dir / "train_all.jsonl"
        save_jsonl(total_samples, out_all)
        print(f"\n  💾 合并保存: {out_all}  ({len(total_samples)} 个样本)")

    # ── 详细过滤报告 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("📊 过滤统计报告")
    print("=" * 64)
    grand_total = 0
    for lang, (n, stats) in summary.items():
        print(f"\n[{lang.upper()}]")
        print(stats.summary())
        grand_total += n

    print("\n" + "─" * 64)
    print(f"✅ 完成！最终有效样本总计: {grand_total} 条")
    print("─" * 64)
    if "tw" not in LANGUAGES:
        print("ℹ️  繁体中文数据已跳过（LANGUAGES=['zh']）。")
        print("   如需处理繁体，修改 LANGUAGES = ['zh', 'tw'] 后重新运行。")
    print("=" * 64)


if __name__ == "__main__":
    main()
