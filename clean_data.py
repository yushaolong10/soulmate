#!/usr/bin/env python3
"""
数据清洗脚本
1. 将每行存在的空格变成 `,`（仅在时间戳后的逗号分隔字段中）
2. 限制用户问题及AI回答的emoji个数，不能超过2个
3. 根据文件内容判断简体/繁体，分别输出到 zh/tw 子目录

输入: datasets0211_src/ 目录
输出: datasets0211_clean/zh/ (简体) 或 datasets0211_clean/tw/ (繁体)
"""

import os
import re
import emoji
from pathlib import Path

# 配置
SRC_DIR = "datasets0211_src"
DST_DIR = "datasets0211_clean"
MAX_EMOJI = 2

# 繁体中文特征字（常见繁体字，简体中不使用）
TRADITIONAL_CHARS = set(
    "臺個這們來說時進還會經過請問後給種說讓實對開點問頭發將題學習會樣東書說對寫從見"
    "體關於為這會國來時過對說經開問請見發進還會學過給時說讓後習於實種點對將頭題動會"
    "沒說過學關給發經這請來時問見進還會讓習於實對種點將頭題開樣動書東寫從會國為體個"
    "們後說對會發學關給時這經過來問見進還讓習實種點對將頭題開樣動書東寫從為體國個們"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請說"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請機"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請變"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請準"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請區"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請與"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請離"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請聽"
    "說對會發經給時這過來問見進還讓習實種點將頭題開樣動書東寫從為體國個們後學關請師"
    # 常用繁体字
    "嗎妳們這來說經過請問後給種說讓實對開點問頭發將題學習會樣東書說對寫從見體關於為"
    "裡麼還現該話讓應資訊機關變準區與離聽師親愛網絡複雜辦處隨視頻連線優質選擇環境號碼"
    "錢買賣價貴識認場廠廳應該衛護許設計劃總統領導圖書館員訂閱號數據庫軟體硬碟機車駕駛證書籤話費單據營業執照證件護照簽證銀行賬戶開戶銷戶購物車訂單運費貨運"
)

# 简体中文特征字（简体中常用，繁体中不使用）
SIMPLIFIED_CHARS = set(
    "这个来说时进还会经过请问后给种说让实对开点问头发将题学习会样东书说对写从见体关于为"
    "里么还现该话让应资讯机关变准区与离听师亲爱网络复杂办处随视频连线优质选择环境号码"
    "钱买卖价贵识认场厂厅应该卫护许设计划总统领导图书馆员订阅号数据库软体硬盘机车驾驶证书签话费单据营业执照证件护照签证银行账户开户销户购物车订单运费货运"
)


def detect_script(text: str) -> str:
    """
    检测文本是简体还是繁体
    返回: 'zh' (简体) 或 'tw' (繁体)
    """
    traditional_count = 0
    simplified_count = 0

    for char in text:
        if char in TRADITIONAL_CHARS:
            traditional_count += 1
        elif char in SIMPLIFIED_CHARS:
            simplified_count += 1

    # 根据特征字数量判断
    if traditional_count > simplified_count:
        return "tw"
    else:
        return "zh"


def count_emojis(text: str) -> int:
    """统计文本中的 emoji 数量"""
    return len([c for c in text if c in emoji.EMOJI_DATA])


def limit_emojis(text: str, max_count: int = MAX_EMOJI) -> str:
    """限制 emoji 数量，超过的移除"""
    if not text:
        return text

    result = []
    emoji_count = 0

    for char in text:
        if char in emoji.EMOJI_DATA:
            if emoji_count < max_count:
                result.append(char)
                emoji_count += 1
            # 超过限制则跳过
        else:
            result.append(char)

    return "".join(result)


def clean_line(line: str) -> str:
    """
    清洗单行数据
    格式: ts,request_content,response_content

    注意: request_content 和 response_content 中可能包含逗号，需要特殊处理
    数据中使用 ➡️ 作为 request 和 response 的分隔符
    """
    line = line.strip()
    if not line:
        return ""

    # 跳过表头
    if line.startswith("ts,"):
        return line

    # 找到第一个逗号（时间戳后）
    first_comma = line.find(",")
    if first_comma == -1:
        return line

    ts = line[:first_comma]
    rest = line[first_comma + 1 :]

    # 使用 ➡️ 分隔 request 和 response
    # 格式: request_content ➡️ ,response_content
    separator = " ➡️ ,"
    sep_idx = rest.find(separator)

    if sep_idx == -1:
        # 尝试其他可能的分隔符格式
        separator = "➡️,"
        sep_idx = rest.find(separator)

    if sep_idx == -1:
        # 如果找不到分隔符，返回原行
        return line

    request_content = rest[:sep_idx]
    response_content = rest[sep_idx + len(separator) :]

    # 1. 将空格替换为中文逗号（只在 request 和 response 内容中）
    request_content = request_content.replace(" ", "，")
    response_content = response_content.replace(" ", "，")

    # 2. 限制 emoji 数量
    request_content = limit_emojis(request_content, MAX_EMOJI)
    response_content = limit_emojis(response_content, MAX_EMOJI)

    # 重新组装
    return f"{ts},{request_content} ➡️ ,{response_content}"


def clean_file(src_path: Path, dst_dir: Path) -> str:
    """
    清洗单个文件，并根据内容判断简繁体
    返回: 'zh' 或 'tw'
    """
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 读取全部内容用于判断简繁体
    full_text = "".join(lines)
    script_type = detect_script(full_text)

    # 确定输出目录
    output_dir = dst_dir / script_type
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = output_dir / src_path.name

    cleaned_lines = []
    for line in lines:
        cleaned = clean_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    print(f"  [{script_type.upper()}] {src_path.name} → {len(cleaned_lines)} 行")
    return script_type


def main():
    src_dir = Path(SRC_DIR)
    dst_dir = Path(DST_DIR)

    if not src_dir.exists():
        print(f"❌ 源目录不存在: {src_dir}")
        return

    # 创建输出目录
    dst_dir.mkdir(exist_ok=True)

    # 获取所有 txt 文件
    txt_files = list(src_dir.glob("*.txt"))
    print(f"📁 找到 {len(txt_files)} 个文件")
    print(f"📝 清洗规则:")
    print(f"   1. 空格 → 中文逗号")
    print(f"   2. emoji 数量限制 ≤ {MAX_EMOJI}")
    print(f"   3. 自动分类: 简体 → zh/, 繁体 → tw/")
    print()

    zh_count = 0
    tw_count = 0

    for src_path in txt_files:
        script_type = clean_file(src_path, dst_dir)
        if script_type == "zh":
            zh_count += 1
        else:
            tw_count += 1

    print()
    print(f"✅ 完成!")
    print(f"   简体 (zh): {zh_count} 个文件 → {dst_dir}/zh/")
    print(f"   繁体 (tw): {tw_count} 个文件 → {dst_dir}/tw/")


if __name__ == "__main__":
    main()
