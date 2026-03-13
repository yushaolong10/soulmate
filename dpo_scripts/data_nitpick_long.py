#!/usr/bin/env python3
"""
data_selected.py — 挑选超长回复作为 DPO 反向样本（rejected）

逻辑：
  读取清洗后的对话数据，用滑动窗口构造对话上下文，
  当窗口最后一轮的 AI 回复（去表情后）字符数 > REJECT_MIN_LEN 时，
  将该上下文 + 超长回复 保存为 DPO rejected 样本。

输入 : SRC_DIR/zh/*.jsonl
       每行格式: {"ts": ..., "request_content": ..., "response_content": ..., "system_prompt": ...}
输出 : DST_FILE（jsonl，每行格式如下）
       {"messages": [...], "label": "...", "window_size": N}

用法：直接运行，或修改配置区后运行。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# =============================================================================
# 配置区
# =============================================================================

# 输入目录（与 data_format.py 一致）
SRC_DIR = "datasets0305_clean"
LANGUAGES = ["zh"]  # 只处理简体中文

# 输出文件
DST_FILE = "datasets0305_train/dpo_src/nitpick_too_long.jsonl"

# 滑动窗口大小配置：
#   - 单个整数：固定窗口，如 8
#   - 二元组/列表：窗口范围，如 (4, 8)，会生成 4~8 轮的动态窗口
WINDOW_SIZE: Union[int, Sequence[int]] = (3, 8)

# 每文件最多使用的对话轮数（防止单个超长文件主导分布）
MAX_TURNS_PER_FILE = 200

# 触发"反向样本"的最小回复长度（去表情后字符数）
REJECT_MIN_LEN = 120

# 触发"反向样本"的最大回复长度（去表情后字符数，0 = 不限制）
REJECT_MAX_LEN = 300

# 最大样本数上限（0 = 不限制）
MAX_SAMPLES = 0

# =============================================================================
# 表情工具（与 data_format.py 保持一致）
# =============================================================================

_EMOJI_BASE = (
    r"[\U0001F1E0-\U0001F1FF]"
    r"|[\U0001F300-\U0001F9FF]"
    r"|[\U0001FA00-\U0001FAFF]"
    r"|[\u2600-\u27BF]"
    r"|[\u2300-\u23FF]"
    r"|[\u2B00-\u2BFF]"
    r"|[\u00A9\u00AE]"
    r"|[\u203C\u2049\u2122\u2139\u24C2]"
)

EMOJI_RE = re.compile(
    rf"(?:{_EMOJI_BASE})"
    r"(?:[\uFE0F\u20E3])?"
    r"(?:[\U0001F3FB-\U0001F3FF])?"
    r"(?:\u200D"
    rf"(?:{_EMOJI_BASE})"
    r"(?:[\uFE0F\u20E3])?"
    r"(?:[\U0001F3FB-\U0001F3FF])?)*",
    re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """去除所有 emoji，返回纯文本。"""
    return EMOJI_RE.sub("", text).strip()


def resolve_window_sizes(window_size: Union[int, Sequence[int]]) -> List[int]:
    """
    解析窗口配置，返回去重后的窗口大小列表。
    支持:
      - 8
      - (4, 8) / [4, 8]  -> 解析为 [4, 5, 6, 7, 8]
    """
    if isinstance(window_size, int):
        if window_size < 2:
            raise ValueError("WINDOW_SIZE 至少需要为 2")
        return [window_size]

    if len(window_size) != 2:
        raise ValueError("WINDOW_SIZE 范围配置必须是长度为 2 的列表或元组")

    min_size, max_size = window_size
    if not isinstance(min_size, int) or not isinstance(max_size, int):
        raise ValueError("WINDOW_SIZE 范围配置必须为整数")
    if min_size < 2 or max_size < 2:
        raise ValueError("WINDOW_SIZE 范围下限和上限都至少需要为 2")
    if min_size > max_size:
        raise ValueError("WINDOW_SIZE 范围下限不能大于上限")

    return list(range(min_size, max_size + 1))


# =============================================================================
# System Prompt 兜底
# =============================================================================

_DEFAULT_SYSTEM_PROMPT = """\
你需要扮演一个虚拟男友角色，用简体中文与女生进行自然亲密的对话。

【对话原则】
- 每次回复尽量简短（30~60字），拒绝长段输出和说教
- 用口语表达，自然亲切，像真实男友一样
- 适度调皮，偶尔反问，保留一点留白
"""


# =============================================================================
# 数据解析
# =============================================================================


def parse_jsonl_file(file_path: Path) -> List[Dict[str, str]]:
    """
    读取 JSONL 文件，返回有效对话轮次列表。
    每个元素: {"user": ..., "assistant": ..., "system_prompt": ...}
    只做基础过滤（空行、空内容），不做长度限制。
    """
    turns = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            user = data.get("request_content", "").strip()
            assistant = data.get("response_content", "").strip()
            system_prompt = data.get("system_prompt", "").strip()

            if not user or not assistant:
                continue

            turns.append(
                {
                    "user": user,
                    "assistant": assistant,
                    "system_prompt": system_prompt,
                }
            )
    return turns


# =============================================================================
# 核心筛选逻辑
# =============================================================================

# 星号动作描述检测（如 *进浴室*、*叹气* 等）
_ASTERISK_ACTION_RE = re.compile(r"\*[^*]+\*")

# 特殊括号符号
_SPECIAL_BRACKETS_RE = re.compile(r"[\[\]【】〔〕]")

# 括号动作描述检测（如 （瞪了你一眼就走）、(转身离开) 等）
# 规则1：整条消息（去表情空格后）完全被括号包裹 —— 最典型的角色扮演动作描述
_FULL_PAREN_RE = re.compile(r"^[（(].+[）)]$")
# 规则2：消息以括号动作开头（如"（笑）好啊"、"（叹气）算了"），括号内含中文且长度1~30字
_LEADING_PAREN_RE = re.compile(r"^[（(][\u4e00-\u9fff][^）)]{0,28}[）)]")

# 用户重复发送检测（如 "老公老公老公老公"、"宝宝宝宝宝宝宝宝"）
# 两档阈值：
#   单字（哈/嗯/啊）：重复 6 次及以上才触发（避免误杀"哈哈哈哈"等正常语气词）
#   2~4字词（老公/宝宝/喜欢）：重复 4 次及以上即触发
_REPEAT_SINGLE_RE = re.compile(r"([\u4e00-\u9fff])\1{5,}")  # 单字重复 ≥6 次
_REPEAT_WORD_RE = re.compile(r"([\u4e00-\u9fff]{2,4})\1{3,}")  # 多字词重复 ≥4 次

# 简繁体粗检测（复用其他 DPO 脚本的字符集合）
_TRADITIONAL = set(
    "妳們這來說時進還會經過請問後給種讓實對開點頭發將題學習樣東書對寫從見"
    "體關於為裡麼現話應訊機變準區離聽師親絡複雜辦處隨頻連線優質選擇環境號碼"
    "錢買賣價貴識認廠廳衛護許設劃統領導圖館訂閱數據軟硬碟駕駛證籤費單據營執照"
    "護照銀賬戶購嗎臺個"
)
_SIMPLIFIED = set(
    "们这来说时进还会经过请问后给种让实对开点头发将题学习样东书对写从见"
    "体关于为里么现话应讯机变准区离听师亲络复杂办处随频连线优质选择环境号码"
    "钱买卖价贵识认厂厅卫护许设划统领导图馆订阅数据软硬盘驾驶证签费单据营执照"
    "护照银账户购吗个"
)
_TRADITIONAL_TOPIC_WORDS = ("繁体", "繁體", "繁中", "正體")

# 中文字符检测
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

# 推理痕迹检测：含"我需要/我应该/我必须"开头的分析段落 + 编号列表
_REASONING_INTRO_RE = re.compile(
    r"(我需要|我应该|我必须).{0,20}(处理|分析|考虑|注意|回应)"
)
_NUMBERED_LIST_RE = re.compile(r"(?:^|\n)\s*[1-9][.、）)]\s+\S", re.MULTILINE)

# Jaccard 相似度阈值（label 与历史 assistant 消息的去重阈值）
JACCARD_THRESHOLD = 0.7


def is_english_reply(text: str) -> bool:
    """判断文本是否为英文回复：去表情后无中文字符且含有英文字母。"""
    bare = strip_emojis(text)
    zh_count = len(_ZH_RE.findall(bare))
    en_count = len(re.findall(r"[a-zA-Z]", bare))
    # 无中文 且 英文字母数 > 3 视为英文回复
    return zh_count == 0 and en_count > 3


def has_asterisk_action(text: str) -> bool:
    """判断文本是否含有 *动作描述* 格式的内容。"""
    return bool(_ASTERISK_ACTION_RE.search(text))


def has_special_brackets(text: str) -> bool:
    """判断文本是否含有 [ ] 【 】 等特殊括号符号。"""
    return bool(_SPECIAL_BRACKETS_RE.search(text))


def has_repeat_word(text: str) -> bool:
    """
    检测消息是否为短词重复发送，分两档：
      - 单字（哈/嗯/啊）重复 ≥6 次：哈哈哈哈哈哈
      - 2~4字词（老公/宝宝/喜欢）重复 ≥4 次：老公老公老公老公
    """
    return bool(_REPEAT_SINGLE_RE.search(text)) or bool(_REPEAT_WORD_RE.search(text))


def detect_script(text: str) -> str:
    """粗判断文本更接近简体还是繁体。"""
    tw = sum(1 for c in text if c in _TRADITIONAL)
    zh = sum(1 for c in text if c in _SIMPLIFIED)
    return "tw" if tw > zh else "zh"


def is_traditional_user_case(text: str) -> bool:
    """
    过滤用户侧繁体相关 case：
      1. 用户明确提到繁体/繁中/正体
      2. 用户文本本身明显是繁体
    """
    bare = strip_emojis(text)
    if not bare:
        return False
    if any(word in bare for word in _TRADITIONAL_TOPIC_WORDS):
        return True

    trad_hits = sum(1 for c in bare if c in _TRADITIONAL)
    simp_hits = sum(1 for c in bare if c in _SIMPLIFIED)
    return trad_hits >= 2 and detect_script(bare) == "tw" and trad_hits > simp_hits


def has_paren_action(text: str) -> bool:
    """
    检测消息是否为括号包裹的角色扮演动作描述，如：
      - （瞪了你一眼就走）      ← 整条消息被括号包裹
      - （叹气）算了            ← 以括号动作开头
    """
    bare = strip_emojis(text).strip()
    if not bare:
        return False
    # 整条消息被括号包裹
    if _FULL_PAREN_RE.match(bare):
        return True
    # 以括号动作开头（括号内含中文）
    if _LEADING_PAREN_RE.match(bare):
        return True
    return False


def is_window_clean(window: List[Dict[str, str]]) -> bool:
    """
    检查整个窗口中所有 user/assistant 消息是否干净（无需过滤的内容）。
    过滤规则（任意一条消息命中则整个窗口丢弃）：
      1. 含有 *动作描述* 星号格式
      2. 含有 [ ] 【 】 等特殊括号
      3. 任意一句为纯英文（无中文且英文字母数 > 3）
      4. 含有 （动作描述） 括号格式的角色扮演消息
      5. 短词重复发送（如 "老公老公老公老公"、"宝宝宝宝宝宝宝宝"）
      6. 用户明确要求/使用繁体中文
    """
    for turn in window:
        if is_traditional_user_case(turn["user"]):
            return False
        for text in (turn["user"], turn["assistant"]):
            if has_asterisk_action(text):
                return False
            if has_special_brackets(text):
                return False
            if is_english_reply(text):
                return False
            if has_paren_action(text):
                return False
            if has_repeat_word(text):
                return False
    return True


def has_reasoning_trace(text: str) -> bool:
    """
    检测 label 是否混入了模型推理痕迹（<think> 泄漏）。
    特征：含有"我需要/我应该/我必须 + 处理/分析..."的分析性前缀，
    且紧跟编号列表（1. 2. 3.）。
    """
    return bool(_REASONING_INTRO_RE.search(text)) and bool(
        _NUMBERED_LIST_RE.search(text)
    )


def jaccard_similarity(a: str, b: str) -> float:
    """计算两段文本的字符级 Jaccard 相似度（以2-gram为单位）。"""

    def ngrams(s: str, n: int = 2):
        s = strip_emojis(s)
        return set(s[i : i + n] for i in range(len(s) - n + 1))

    set_a = ngrams(a)
    set_b = ngrams(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_duplicate_of_history(label: str, window: List[Dict[str, str]]) -> bool:
    """
    检测 label 是否与窗口中历史 assistant 消息（不含最后一轮）高度重复。
    Jaccard 相似度 > JACCARD_THRESHOLD 则视为复读，过滤掉。
    """
    for turn in window[:-1]:  # 排除最后一轮（最后一轮的 assistant 就是 label 本身）
        sim = jaccard_similarity(label, turn["assistant"])
        if sim > JACCARD_THRESHOLD:
            return True
    return False


def is_too_long_rejected(assistant_text: str) -> bool:
    """判断 AI 回复（去表情后）长度是否在 [REJECT_MIN_LEN, REJECT_MAX_LEN] 范围内，应作为反向样本。"""
    bare = strip_emojis(assistant_text)
    bare_len = len(bare)
    if bare_len <= REJECT_MIN_LEN:
        return False
    if REJECT_MAX_LEN > 0 and bare_len > REJECT_MAX_LEN:
        return False
    return True


def build_messages(window: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    将滑动窗口（除最后一轮）构造成 messages 列表。
    System Prompt 取窗口第一轮的 system_prompt；若为空则用内置默认。
    """
    system_prompt = window[0]["system_prompt"] or _DEFAULT_SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # 除最后一轮外，其余作为历史上下文
    for turn in window[:-1]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # 最后一轮只放用户消息（assistant 作为 rejected）
    messages.append({"role": "user", "content": window[-1]["user"]})
    return messages


def format_window_size_config(window_sizes: List[int]) -> str:
    """格式化窗口配置，便于打印日志。"""
    if not window_sizes:
        return "[]"
    if len(window_sizes) == 1:
        return f"{window_sizes[0]} 轮"
    return f"{window_sizes[0]}~{window_sizes[-1]} 轮（动态）"


def select_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """处理单个文件，返回符合条件的 rejected 样本列表。"""
    turns = parse_jsonl_file(file_path)
    window_sizes = resolve_window_sizes(WINDOW_SIZE)
    if len(turns) < min(window_sizes):
        return []

    # 截断：防止单个超长文件独占数据集
    if len(turns) > MAX_TURNS_PER_FILE:
        turns = turns[:MAX_TURNS_PER_FILE]

    selected = []
    for window_size in window_sizes:
        if len(turns) < window_size:
            continue

        for i in range(len(turns) - window_size + 1):
            window = turns[i : i + window_size]
            last_assistant = window[-1]["assistant"]

            # 过滤超长判断
            if not is_too_long_rejected(last_assistant):
                continue

            # 过滤整个窗口中含英文句、星号动作或特殊括号的消息
            if not is_window_clean(window):
                continue

            # 过滤 label 含推理痕迹（<think> 泄漏）
            if has_reasoning_trace(last_assistant):
                continue

            # 过滤 label 与历史 assistant 消息高度重复（复读行为）
            if is_duplicate_of_history(last_assistant, window):
                continue

            messages = build_messages(window)

            selected.append(
                {
                    "messages": messages,
                    "label": last_assistant,
                    "window_size": window_size,
                }
            )

    return selected


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst_path = Path(DST_FILE)
    window_sizes = resolve_window_sizes(WINDOW_SIZE)

    print("=" * 64)
    print("🔍 data_selected.py — 超长回复反向样本挑选")
    print("=" * 64)
    print(f"输入目录:     {src_dir}")
    print(f"输出文件:     {dst_path}")
    print(f"处理语言:     {LANGUAGES}")
    print(f"滑动窗口:     {format_window_size_config(window_sizes)}")
    print(f"每文件上限:   {MAX_TURNS_PER_FILE} 轮")
    print(
        f"反向阈值:     回复去表情后 > {REJECT_MIN_LEN} 字  ≤ {'不限' if REJECT_MAX_LEN == 0 else REJECT_MAX_LEN} 字"
    )
    print(f"样本上限:     {'无限制' if MAX_SAMPLES == 0 else MAX_SAMPLES}")
    print("=" * 64)

    all_selected: List[Dict[str, Any]] = []

    for lang in LANGUAGES:
        lang_dir = src_dir / lang
        if not lang_dir.exists():
            print(f"\n⚠️  目录不存在，跳过: {lang_dir}")
            continue

        jsonl_files = sorted(lang_dir.glob("*.jsonl"))
        print(f"\n📂 [{lang.upper()}] {lang_dir}  ── 共 {len(jsonl_files)} 个文件")

        for file_path in jsonl_files:
            samples = select_from_file(file_path)
            all_selected.extend(samples)
            if samples:
                lens = [len(strip_emojis(s["label"])) for s in samples]
                print(
                    f"  ✓  {file_path.name}: "
                    f"命中 {len(samples)} 条  "
                    f"长度范围 [{min(lens)}~{max(lens)}]"
                )
            else:
                print(f"  ·  {file_path.name}: 无命中")

            # 样本上限检查
            if MAX_SAMPLES > 0 and len(all_selected) >= MAX_SAMPLES:
                print(f"\n⚡ 已达样本上限 {MAX_SAMPLES} 条，提前结束")
                all_selected = all_selected[:MAX_SAMPLES]
                break

        if MAX_SAMPLES > 0 and len(all_selected) >= MAX_SAMPLES:
            break

    # 保存
    if all_selected:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            for sample in all_selected:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计报告
    print("\n" + "=" * 64)
    print(f"✅ 完成！共挑选 {len(all_selected)} 条超长反向样本")
    if all_selected:
        lens = [len(strip_emojis(s["label"])) for s in all_selected]
        print(f"   回复长度分布:")
        ranges = [(120, 150), (150, 200), (200, 300), (300, float("inf"))]
        range_labels = ["120~150", "150~200", "200~300", "300+"]
        for (lo, hi), rl in zip(ranges, range_labels):
            cnt = sum(1 for l in lens if lo < l <= hi)
            bar = "█" * min(cnt // max(1, len(all_selected) // 20), 20)
            print(f"   {rl:>8} 字: {cnt:>5} 条  {bar}")
        print(f"   平均长度: {sum(lens) / len(lens):.1f} 字")
        print(f"   最长回复: {max(lens)} 字")
        print(f"   输出文件: {dst_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
