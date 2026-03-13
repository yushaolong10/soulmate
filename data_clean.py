#!/usr/bin/env python3
"""
数据清洗脚本 (v2 — JSONL 格式)
对 datasets0303_src/ 中的 .jsonl 文件进行清洗，输出到 datasets0303_clean/

每行 JSON 格式:
  {"ts": "...", "request_content": "...", "response_content": "...", "system_prompt": "..."}

清洗规则:
  1. 限制 request_content / response_content 中的 emoji ≤ MAX_EMOJI 个（多余用 ，替换）
     → 替换后清理连续标点，如 "，，"/"，。" 等
  2. 去除 response_content / request_content 中的转义残留（\" → "，\\n → 换行 等）
  3. 去除 response_content / request_content 中的空格（普通空格 + 全角空格）
  2. 根据 response_content + request_content 内容自动判断简体/繁体
     → 简体输出到 datasets0303_clean/zh/
     → 繁体输出到 datasets0303_clean/tw/
  3. system_prompt、ts 字段原样保留
  4. 跳过 JSON 解析失败或字段缺失的行
"""

import json
import re
import sys
from pathlib import Path

# =============================================================================
# 配置
# =============================================================================

SRC_DIR = "datasets0305_src"
DST_DIR = "datasets0305_clean"
MAX_EMOJI = 2

# =============================================================================
# 简繁体检测
# =============================================================================

# 繁体特征字（简体中不用）
_TRADITIONAL = set(
    "妳們這來說時進還會經過請問後給種讓實對開點頭發將題學習樣東書對寫從見"
    "體關於為裡麼現該話讓應訊機關變準區與離聽師親網絡複雜辦處隨視頻連線優質選擇環境號碼"
    "錢買賣價貴識認場廠廳衛護許設劃統領導圖館訂閱數據庫軟硬碟駕駛證籤費單據營執照護照簽銀賬戶購"
    "嗎臺個"
)

# 简体特征字（繁体中不用）
_SIMPLIFIED = set(
    "们这来说时进还会经过请问后给种让实对开点头发将题学习样东书对写从见"
    "体关于为里么现该话让应讯机关变准区与离听师亲网络复杂办处随视频连线优质选择环境号码"
    "钱买卖价贵识认场厂厅卫护许设划统领导图馆订阅数据库软硬盘驾驶证签费单据营执照护照银账户购"
)


def detect_script(text: str) -> str:
    """
    统计特征字出现次数，返回 'zh'（简体）或 'tw'（繁体）。
    """
    tw = sum(1 for c in text if c in _TRADITIONAL)
    zh = sum(1 for c in text if c in _SIMPLIFIED)
    return "tw" if tw > zh else "zh"


# =============================================================================
# Emoji 工具
# =============================================================================

# 复用 format_data.py 中精确的 emoji 正则，避免依赖 emoji 库
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


def limit_emojis(text: str, max_count: int = MAX_EMOJI) -> str:
    """保留前 max_count 个 emoji，多余的替换为 ，（中文逗号）。"""
    matches = list(EMOJI_RE.finditer(text))
    if len(matches) <= max_count:
        return text
    result = list(text)
    for m in matches[max_count:]:
        # 将整个 emoji 序列（含 ZWJ/肤色等修饰符）替换为单个逗号
        for i in range(m.start(), m.end()):
            result[i] = ""
        result[m.start()] = "，"
    return "".join(result)


# =============================================================================
# 单行处理
# =============================================================================


def remove_spaces(text: str) -> str:
    """去除文本中的所有空格（中文对话中空格通常为噪声）。"""
    return text.replace(" ", "").replace("\u3000", "")  # 普通空格 + 全角空格


# 句末强标点 / 句内弱标点
_PUNCT_STRONG = r"[。！？!?]"
_PUNCT_WEAK = r"[，,、；;：:]"


def clean_punctuation(text: str) -> str:
    """
    清理 emoji 替换后残留的连续标点：
      1. 连续弱标点 → 保留一个（，，→ ，）
      2. 弱标点 + 强标点 → 仅保留强标点（，。→ 。）
      3. 同类强标点重复 → 保留一个（。。→ 。）
      4. 删除行首/行尾多余弱标点
    """
    # 1. 连续弱标点 → 单个
    text = re.sub(rf"({_PUNCT_WEAK})(?:\s*{_PUNCT_WEAK})+", r"\1", text)
    # 2. 弱标点 + 强标点 → 仅保留强标点
    text = re.sub(rf"{_PUNCT_WEAK}\s*({_PUNCT_STRONG})", r"\1", text)
    # 3. 同类强标点重复 → 单个
    for p in "。！？!?":
        text = re.sub(re.escape(p) + r"+", p, text)
    # 4. 行首/行尾多余弱标点
    text = re.sub(rf"^(?:\s*{_PUNCT_WEAK})+", "", text)
    text = re.sub(rf"(?:{_PUNCT_WEAK})+\s*$", "", text)
    return text


# 常见转义序列 → 目标字符（按优先级由长到短替换，避免 \\ 先被替换导致误处理）
_UNESCAPE_MAP = [
    ('\\"', '"'),  # \" → "
    ("\\'", "'"),  # \' → '
    ("\\n", "\n"),  # \n → 换行
    ("\\t", "\t"),  # \t → 制表符
    ("\\r", ""),  # \r → 删除（统一为 \n 换行）
    ("\\\\", "\\"),  # \\ → \（最后处理，避免误替换）
]


def unescape_content(text: str) -> str:
    """
    去除内容中多余的转义符，例如：
      \\" → "   \\n → 换行   \\\\ → \\
    适用于 JSON 双重编码导致的转义残留。
    """
    for escaped, replacement in _UNESCAPE_MAP:
        text = text.replace(escaped, replacement)
    return text


def clean_record(record: dict) -> dict:
    """
    清洗单条 JSON 记录。
    - request_content / response_content：去除转义残留、限制 emoji、清理连续标点、去除空格
    - ts / system_prompt：原样保留
    """
    record["request_content"] = remove_spaces(
        clean_punctuation(
            limit_emojis(unescape_content(record.get("request_content", "")), MAX_EMOJI)
        )
    )
    record["response_content"] = remove_spaces(
        clean_punctuation(
            limit_emojis(
                unescape_content(record.get("response_content", "")), MAX_EMOJI
            )
        )
    )
    return record


# =============================================================================
# 文件处理
# =============================================================================


def clean_file(src_path: Path, dst_dir: Path) -> tuple[str, int, int]:
    """
    清洗单个 .jsonl 文件。
    返回: (script_type, total_lines, valid_lines)
    """
    valid_records = []
    total = 0
    skip = 0

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skip += 1
                continue

            # 必须包含核心字段
            if not record.get("request_content") and not record.get("response_content"):
                skip += 1
                continue

            valid_records.append(clean_record(record))

    if not valid_records:
        return "zh", total, 0

    # 用所有 response + request 内容合并判断简繁
    sample_text = " ".join(
        r.get("response_content", "") + r.get("request_content", "")
        for r in valid_records[:100]  # 取前100条即可判断
    )
    script_type = detect_script(sample_text)

    # 确定输出路径（保持原文件名，扩展名为 .jsonl）
    out_dir = dst_dir / script_type
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / src_path.name  # 已是 .jsonl，直接保留

    with open(dst_path, "w", encoding="utf-8") as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return script_type, total, len(valid_records)


# =============================================================================
# 主函数
# =============================================================================


def main() -> None:
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)

    if not src_dir.exists():
        print(f"❌ 源目录不存在: {src_dir}")
        sys.exit(1)

    dst_dir.mkdir(exist_ok=True)

    jsonl_files = sorted(src_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ 在 {src_dir}/ 中未找到 .jsonl 文件")
        sys.exit(1)

    print("=" * 60)
    print("🧹 数据清洗脚本 v2 — JSONL 格式")
    print("=" * 60)
    print(f"源目录:  {src_dir}/")
    print(f"输出目录: {dst_dir}/zh/  和  {dst_dir}/tw/")
    print(f"emoji 上限: {MAX_EMOJI} 个/条")
    print(f"处理文件: {len(jsonl_files)} 个")
    print("=" * 60)

    zh_files = zh_lines = 0
    tw_files = tw_lines = 0
    total_raw = total_valid = 0

    for src_path in jsonl_files:
        script_type, raw, valid = clean_file(src_path, dst_dir)
        total_raw += raw
        total_valid += valid

        if script_type == "zh":
            zh_files += 1
            zh_lines += valid
        else:
            tw_files += 1
            tw_lines += valid

        skip_note = f" (跳过 {raw - valid} 行)" if raw != valid else ""
        print(
            f"  [{script_type.upper()}] {src_path.name}: {raw} 行 → {valid} 行{skip_note}"
        )

    print()
    print("=" * 60)
    print("✅ 完成！")
    print(f"   简体 (zh): {zh_files:4d} 个文件，{zh_lines:,} 行 → {dst_dir}/zh/")
    print(f"   繁体 (tw): {tw_files:4d} 个文件，{tw_lines:,} 行 → {dst_dir}/tw/")
    print(
        f"   原始总行: {total_raw:,}  有效输出: {total_valid:,}  跳过: {total_raw - total_valid:,}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
