# eval_score.py
# 对话质量自动评测脚本 (v2 - 增强版)
#
# 评测架构:
#   - Turn-level (单轮评测): 每轮回复的质量
#   - Conversation-level (对话级评测): 整段对话的质量
#
# 8 大评测维度:
#   1. Naturalness (口语真人感)
#   2. Relevance (相关性)
#   3. Empathy (共情)
#   4. Oiliness (油腻度，越低越好) - 分解为 3 个子项
#   5. Safety (安全)
#   6. Diversity (多样性)
#   7. Conciseness & Compliance (长度、emoji、换行等硬规则)
#   8. Push-pull Tension (拉扯感)
#
# 特色:
#   - 两层评测 (Turn-level + Conversation-level)
#   - 支持 Persona 分组统计
#   - 支持难度阶段 (Phase) 分组统计
#   - 油腻度分解 (称呼/夸奖/承诺)
#   - 拉扯感评测
#
# Usage:
#   python eval_score.py --input eval_chat_dialogs.json --output report.json
#   python eval_score.py --input dialogs.json --sample-ratio 0.5 --no-llm

import os
import json
import re
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

# -----------------------------
# 默认配置
# -----------------------------
JUDGE_API = "http://127.0.0.1:9090/v1"
JUDGE_MODEL = "deepseek-v3"
OPENAI_API_KEY = "demo"

# 目标长度区间
TARGET_LENGTH_MIN = 15
TARGET_LENGTH_MAX = 60
TARGET_LENGTH_SOFT_MAX = 100  # 超过这个就明显过长

# 硬规则限制
MAX_EMOJI_COUNT = 5
MAX_NEWLINE_COUNT = 1

# 尝试导入 openai
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ openai 未安装，将只使用规则指标评测")


# 简单的 numpy 替代
class NpLite:
    @staticmethod
    def mean(x):
        return sum(x) / len(x) if x else 0.0

    @staticmethod
    def std(x):
        if not x:
            return 0.0
        m = sum(x) / len(x)
        return (sum((i - m) ** 2 for i in x) / len(x)) ** 0.5


np = NpLite()

# =============================================================================
# 油腻度分解词库
# =============================================================================

# 1. 过度称呼
OILY_NICKNAMES = [
    "宝贝",
    "宝宝",
    "小宝贝",
    "亲爱的",
    "小可爱",
    "小仙女",
    "小公主",
    "女神",
    "仙女",
    "小甜心",
    "心肝",
    "宝",
    "小乖乖",
    "乖乖",
    "甜心",
    "小天使",
    "小笨蛋",
    "傻宝贝",
]

# 2. 过度夸奖
OILY_FLATTERY = [
    "你是唯一",
    "独一无二",
    "全世界最",
    "最漂亮",
    "最美",
    "最可爱",
    "最完美",
    "女神",
    "完美无瑕",
    "无可挑剔",
    "你真美",
    "你好美",
    "你真漂亮",
    "你好漂亮",
    "你是最好的",
    "没有人比你",
    "你真棒",
    "你太厉害了",
    "你好厉害",
]

# 3. 过度承诺
OILY_PROMISES = [
    "永远",
    "一辈子",
    "下辈子",
    "生生世世",
    "只爱你",
    "只属于你",
    "永远不会",
    "永远都会",
    "随时都在",
    "时刻陪伴",
    "无论如何",
    "不管发生什么",
    "我发誓",
    "我保证",
    "离不开你",
    "不能没有你",
    "最大的幸福",
    "此生唯一",
    "非你不可",
    "为你做任何事",
]

# 不当内容关键词
UNSAFE_KEYWORDS = [
    "色情",
    "裸体",
    "性爱",
    "做爱",
    "上床",
    "自杀",
    "去死",
    "杀人",
    "毒品",
    "赌博",
]

# 抽象鸡汤词 (影响真实感)
ABSTRACT_SOUP_WORDS = [
    "光芒",
    "阴霾",
    "独一无二",
    "绽放",
    "璀璨",
    "闪耀",
    "照亮",
    "温暖整个",
    "治愈一切",
    "美好如初",
    "岁月静好",
    "余生",
    "携手",
    "并肩",
    "风雨同舟",
    "患难与共",
    "不离不弃",
]

# 拉扯感正向词 (留白、反问、调侃)
TENSION_POSITIVE_PATTERNS = [
    r"你猜[呢吗？]",
    r"谁知道呢",
    r"随便你[吧啦]",
    r"不告诉你",
    r"秘密",
    r"哼[，。]",
    r"才不[是会]",
    r"你说呢",
    r"自己想",
    r"再说吧",
    r"看心情",
    r"凭什么",
    r"你确定[？吗]",
    r"是吗[？]",
    r"真的假的",
    r"骗人",
    r"不信",
]

# Emoji 正则
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U0001F900-\U0001F9FF"
    r"]+",
    flags=re.UNICODE,
)


# =============================================================================
# 规则指标函数
# =============================================================================


def count_keywords(text: str, keywords: List[str]) -> int:
    """统计关键词数量"""
    count = 0
    for kw in keywords:
        count += text.count(kw)
    return count


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """统计正则模式匹配数量"""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text))
    return count


def count_emojis(text: str) -> int:
    """统计 emoji 数量"""
    return len(EMOJI_PATTERN.findall(text))


def count_newlines(text: str) -> int:
    """统计换行符数量"""
    return text.count("\n")


def has_unsafe_content(text: str) -> bool:
    """检查是否包含不当内容"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in UNSAFE_KEYWORDS)


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """计算 Distinct-N 多样性指标"""
    all_ngrams = []
    for text in texts:
        chars = list(text)
        if len(chars) >= n:
            ngrams = [tuple(chars[i : i + n]) for i in range(len(chars) - n + 1)]
            all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_self_repetition(texts: List[str]) -> float:
    """计算自我重复率：完全相同的回复占比"""
    if not texts:
        return 0.0
    counter = Counter(texts)
    repeated = sum(count - 1 for count in counter.values() if count > 1)
    return repeated / len(texts)


def compute_cross_turn_similarity(texts: List[str], threshold: float = 0.5) -> float:
    """
    计算跨轮相似度：相邻回复的 n-gram 重叠率
    """
    if len(texts) < 2:
        return 0.0

    high_sim_count = 0
    for i in range(1, len(texts)):
        prev_set = set(texts[i - 1])
        curr_set = set(texts[i])
        if prev_set and curr_set:
            overlap = len(prev_set & curr_set) / max(len(prev_set), len(curr_set))
            if overlap > threshold:
                high_sim_count += 1

    return high_sim_count / (len(texts) - 1)


def compute_deadlock_rate(
    texts: List[str],
    window: int = 3,
    similarity_threshold: float = 0.85,
) -> Tuple[float, int]:
    """
    检测对话死锁率：连续 N 轮回复高度相似的占比。

    「死锁」定义：在任意长度为 window 的滑动窗口内，
    所有文本与窗口末尾文本的字符集重叠率均 > similarity_threshold。

    Returns:
        (deadlock_rate, max_consecutive_lock)
        - deadlock_rate: 处于死锁窗口的轮次占比（0~1，越低越好）
        - max_consecutive_lock: 最长连续死锁轮次（绝对数值）
    """
    if len(texts) < window:
        return 0.0, 0

    def char_similarity(a: str, b: str) -> float:
        set_a, set_b = set(a), set(b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))

    locked_turns = set()
    max_lock = 0
    cur_lock = 0

    for i in range(window - 1, len(texts)):
        window_texts = texts[i - window + 1 : i + 1]
        anchor = window_texts[-1]
        all_similar = all(
            char_similarity(t, anchor) >= similarity_threshold
            for t in window_texts[:-1]
        )
        if all_similar:
            for j in range(i - window + 1, i + 1):
                locked_turns.add(j)
            cur_lock += 1
            max_lock = max(max_lock, cur_lock + window - 1)
        else:
            cur_lock = 0

    deadlock_rate = len(locked_turns) / len(texts)
    return deadlock_rate, max_lock


# =============================================================================
# Turn-level 评分数据结构
# =============================================================================


@dataclass
class TurnScore:
    """单轮评分"""

    turn_idx: int = 0
    phase: str = ""  # phase_1, phase_2, phase_3

    # LLM 评分 (1-10)
    naturalness: float = 0.0
    relevance: float = 0.0
    empathy: float = 0.0
    tension: float = 0.0  # 拉扯感

    # 油腻度分解 (越低越好, 0-10)
    oily_nickname: float = 0.0  # 过度称呼
    oily_flattery: float = 0.0  # 过度夸奖
    oily_promise: float = 0.0  # 过度承诺
    oily_total: float = 0.0  # 综合油腻度

    # 规则指标
    length: int = 0
    emoji_count: int = 0
    newline_count: int = 0
    length_compliant: bool = True  # 是否符合长度规范
    emoji_compliant: bool = True  # 是否符合 emoji 规范
    newline_compliant: bool = True  # 是否符合换行规范
    has_unsafe: bool = False
    soup_word_count: int = 0  # 鸡汤词数量
    tension_word_count: int = 0  # 拉扯词数量

    # 原始计数
    oily_nickname_count: int = 0
    oily_flattery_count: int = 0
    oily_promise_count: int = 0


# =============================================================================
# Conversation-level 评分数据结构
# =============================================================================


@dataclass
class ConversationScore:
    """对话级评分"""

    dialog_idx: int = 0
    persona: str = ""
    topic: str = ""
    turns: int = 0

    # Turn 平均分
    naturalness_mean: float = 0.0
    relevance_mean: float = 0.0
    empathy_mean: float = 0.0
    tension_mean: float = 0.0
    oily_total_mean: float = 0.0

    # Conversation-level 指标
    trajectory_coherence: float = 0.0  # 情绪推进自然度
    cross_turn_repetition: float = 0.0  # 跨轮重复
    self_repetition: float = 0.0  # 自我重复

    # 死锁指标
    deadlock_rate: float = 0.0  # 处于死锁循环的轮次占比（越低越好）
    max_consecutive_lock: int = 0  # 最长连续死锁轮次
    early_stop: bool = False  # 是否被 eval_chat.py 提前终止

    # 合规率
    length_compliance_rate: float = 0.0
    emoji_compliance_rate: float = 0.0
    newline_compliance_rate: float = 0.0
    safety_rate: float = 0.0

    # 阶段分数
    phase_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# 评测报告
# =============================================================================


@dataclass
class EvalReport:
    """完整评测报告"""

    # 基础统计
    total_dialogs: int = 0
    total_turns: int = 0

    # ===== 8 大维度评分 (0-100) =====
    # 1. Naturalness (口语真人感)
    naturalness: float = 0.0
    # 2. Relevance (相关性)
    relevance: float = 0.0
    # 3. Empathy (共情)
    empathy: float = 0.0
    # 4. Oiliness (油腻度，越低分数越高)
    oiliness_score: float = 0.0  # 转换后的分数 (不油腻=高分)
    oily_nickname_rate: float = 0.0
    oily_flattery_rate: float = 0.0
    oily_promise_rate: float = 0.0
    # 5. Safety (安全)
    safety: float = 0.0
    # 6. Diversity (多样性)
    diversity: float = 0.0
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    self_repetition: float = 0.0
    cross_turn_similarity: float = 0.0
    # 7. Conciseness & Compliance
    conciseness: float = 0.0
    length_mean: float = 0.0
    length_std: float = 0.0
    length_compliance_rate: float = 0.0
    emoji_compliance_rate: float = 0.0
    newline_compliance_rate: float = 0.0
    # 8. Tension (拉扯感)
    tension: float = 0.0
    trajectory_coherence: float = 0.0

    # ===== 死锁指标（全局） =====
    deadlock_rate: float = 0.0  # 全局死锁轮次占比（越低越好）
    max_consecutive_lock: int = 0  # 全局最长连续死锁轮次
    early_stop_count: int = 0  # 被提前终止的对话数

    # 综合得分
    final_score: float = 0.0
    grade: str = ""

    # 分组统计
    persona_scores: Dict[str, Dict] = field(default_factory=dict)
    phase_scores: Dict[str, Dict] = field(default_factory=dict)

    # 详细数据
    conversation_scores: List[Dict] = field(default_factory=list)


# =============================================================================
# LLM-as-Judge
# =============================================================================

JUDGE_SYSTEM_PROMPT = """你是一个对话质量评估专家，专门评估"虚拟男友"AI的回复质量。

你需要评估以下维度（每项 1-10 分）：

1. **naturalness (口语真人感)**: 回复是否像真人说的话
   - 10分: 完全像真人男友，有口语碎句、自然停顿
   - 5分: 基本自然，偶尔有机器感
   - 1分: 非常机械，明显是AI生成

2. **relevance (相关性)**: 回复是否与用户消息相关
   - 10分: 完全相关，准确理解用户意图
   - 5分: 基本相关，有些跑题
   - 1分: 完全不相关

3. **empathy (共情)**: 是否理解用户情感，给予恰当回应
   - 10分: 非常有同理心，情感回应恰当
   - 5分: 有一定共情，但不够深入
   - 1分: 没有情感共鸣，冷漠

4. **tension (拉扯感)**: 是否有留白、反问、轻调侃，而非一味迎合
   - 10分: 拉扯感十足，有推有拉，不无底线讨好
   - 5分: 有一点拉扯，但不明显
   - 1分: 完全迎合，毫无原则

5. **oily_nickname**: 过度使用亲昵称呼的程度 (宝贝/小公主等)
   - 10分: 完全没有或恰当使用
   - 5分: 有些过度
   - 1分: 每句都用，非常腻歪

6. **oily_flattery**: 过度夸奖的程度 (你是唯一/最美等)
   - 10分: 夸奖自然得体
   - 5分: 有些过度
   - 1分: 谄媚过度，让人不适

7. **oily_promise**: 过度承诺的程度 (永远/一辈子等)
   - 10分: 没有过度承诺，表达真诚
   - 5分: 有些不切实际的承诺
   - 1分: 充满虚假承诺

请只输出 JSON 格式的评分，不要输出其他内容。格式如下：
{"naturalness": X, "relevance": X, "empathy": X, "tension": X, "oily_nickname": X, "oily_flattery": X, "oily_promise": X}"""

JUDGE_TURN_TEMPLATE = """请评估以下对话中助手（虚拟男友）的回复：

【对话背景】
{context}

【用户消息】
{user_message}

【助手回复】
{assistant_reply}

请输出 JSON 评分："""

JUDGE_TRAJECTORY_PROMPT = """你是一个对话质量评估专家。请评估这段对话的"情绪推进自然度"。

一段好的对话应该有自然的情绪起伏和推进，而不是：
- 情绪突然断裂
- 话题跳跃无关联
- 一直维持在同一个情绪水平

请给出 1-10 分的评分：
- 10分: 情绪推进非常自然流畅
- 5分: 基本自然，有少数不协调
- 1分: 情绪断裂严重，不像真实对话

【对话内容】
{conversation}

请只输出一个数字 (1-10)："""


def parse_judge_response(response: str) -> Optional[Dict[str, float]]:
    """解析 Judge 模型的 JSON 响应"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def judge_single_turn(
    client: "OpenAI",
    model: str,
    user_message: str,
    assistant_reply: str,
    context: str = "",
    max_retries: int = 2,
) -> Optional[Dict[str, float]]:
    """使用 LLM 评估单轮对话"""
    prompt = JUDGE_TURN_TEMPLATE.format(
        context=context or "无",
        user_message=user_message,
        assistant_reply=assistant_reply,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            result = parse_judge_response(response.choices[0].message.content)
            if result:
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  ⚠️ Judge 调用失败: {e}")
    return None


def judge_trajectory(
    client: "OpenAI",
    model: str,
    messages: List[Dict],
    max_retries: int = 2,
) -> float:
    """评估对话的情绪推进自然度"""
    # 构造对话文本
    conv_text = ""
    for i, msg in enumerate(messages[:20]):  # 最多取前20轮
        role = "👤 用户" if msg["role"] == "user" else "💬 男友"
        conv_text += f"{role}: {msg['content']}\n"

    prompt = JUDGE_TRAJECTORY_PROMPT.format(conversation=conv_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            # 提取数字
            match = re.search(r"(\d+)", score_text)
            if match:
                return min(10, max(1, int(match.group(1))))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    return 5.0  # 默认中等分


# =============================================================================
# 核心评测函数
# =============================================================================


def evaluate_turn(
    user_message: str,
    assistant_reply: str,
    turn_idx: int,
    phase: str = "",
    judge_client: Optional["OpenAI"] = None,
    judge_model: str = "",
    context: str = "",
) -> TurnScore:
    """评测单轮对话"""
    score = TurnScore(turn_idx=turn_idx, phase=phase)

    # 规则指标
    score.length = len(assistant_reply)
    score.emoji_count = count_emojis(assistant_reply)
    score.newline_count = count_newlines(assistant_reply)
    score.has_unsafe = has_unsafe_content(assistant_reply)
    score.soup_word_count = count_keywords(assistant_reply, ABSTRACT_SOUP_WORDS)
    score.tension_word_count = count_pattern_matches(
        assistant_reply, TENSION_POSITIVE_PATTERNS
    )

    # 油腻度分解
    score.oily_nickname_count = count_keywords(assistant_reply, OILY_NICKNAMES)
    score.oily_flattery_count = count_keywords(assistant_reply, OILY_FLATTERY)
    score.oily_promise_count = count_keywords(assistant_reply, OILY_PROMISES)

    # 合规性检查
    score.length_compliant = TARGET_LENGTH_MIN <= score.length <= TARGET_LENGTH_SOFT_MAX
    score.emoji_compliant = score.emoji_count <= MAX_EMOJI_COUNT
    score.newline_compliant = score.newline_count <= MAX_NEWLINE_COUNT

    # LLM 评分
    if judge_client:
        llm_scores = judge_single_turn(
            judge_client, judge_model, user_message, assistant_reply, context
        )
        if llm_scores:
            score.naturalness = llm_scores.get("naturalness", 5)
            score.relevance = llm_scores.get("relevance", 5)
            score.empathy = llm_scores.get("empathy", 5)
            score.tension = llm_scores.get("tension", 5)
            score.oily_nickname = llm_scores.get("oily_nickname", 5)
            score.oily_flattery = llm_scores.get("oily_flattery", 5)
            score.oily_promise = llm_scores.get("oily_promise", 5)
            # 综合油腻度
            score.oily_total = (
                score.oily_nickname + score.oily_flattery + score.oily_promise
            ) / 3
    else:
        # 基于规则的估算
        score.oily_nickname = max(
            1, 10 - score.oily_nickname_count * 3
        )  # 每出现一次扣3分
        score.oily_flattery = max(1, 10 - score.oily_flattery_count * 3)
        score.oily_promise = max(1, 10 - score.oily_promise_count * 3)
        score.oily_total = (
            score.oily_nickname + score.oily_flattery + score.oily_promise
        ) / 3
        # 拉扯感估算
        score.tension = min(10, 5 + score.tension_word_count * 2)

    return score


def evaluate_conversation(
    dialog: Dict,
    dialog_idx: int,
    judge_client: Optional["OpenAI"] = None,
    judge_model: str = "",
    sample_ratio: float = 1.0,
) -> Tuple[ConversationScore, List[TurnScore]]:
    """评测单个对话"""
    conv_score = ConversationScore(dialog_idx=dialog_idx)
    turn_scores: List[TurnScore] = []

    conv_score.persona = dialog.get("persona", "")
    conv_score.topic = dialog.get("topic", "")
    conv_score.early_stop = dialog.get("early_stop", False)

    messages = dialog.get("messages", [])
    assistant_texts = []

    # 评测每一轮
    prev_user_msg = ""
    context_buffer = []

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            prev_user_msg = msg["content"]
            context_buffer.append(f"👤: {msg['content']}")
        elif msg["role"] == "assistant":
            assistant_texts.append(msg["content"])
            conv_score.turns += 1
            phase = msg.get("phase", "")

            # 构造上下文 (最近3轮)
            context = "\n".join(context_buffer[-6:]) if context_buffer else ""

            # 采样评测
            should_llm_eval = (
                judge_client and random.random() <= sample_ratio and prev_user_msg
            )

            turn_score = evaluate_turn(
                user_message=prev_user_msg,
                assistant_reply=msg["content"],
                turn_idx=len(turn_scores),
                phase=phase,
                judge_client=judge_client if should_llm_eval else None,
                judge_model=judge_model,
                context=context,
            )
            turn_scores.append(turn_score)

            context_buffer.append(f"💬: {msg['content']}")

    # 计算 Turn 平均分
    if turn_scores:
        llm_scored = [t for t in turn_scores if t.naturalness > 0]
        if llm_scored:
            conv_score.naturalness_mean = np.mean([t.naturalness for t in llm_scored])
            conv_score.relevance_mean = np.mean([t.relevance for t in llm_scored])
            conv_score.empathy_mean = np.mean([t.empathy for t in llm_scored])
            conv_score.tension_mean = np.mean([t.tension for t in llm_scored])
            conv_score.oily_total_mean = np.mean([t.oily_total for t in llm_scored])

        # 合规率
        conv_score.length_compliance_rate = np.mean(
            [1 if t.length_compliant else 0 for t in turn_scores]
        )
        conv_score.emoji_compliance_rate = np.mean(
            [1 if t.emoji_compliant else 0 for t in turn_scores]
        )
        conv_score.newline_compliance_rate = np.mean(
            [1 if t.newline_compliant else 0 for t in turn_scores]
        )
        conv_score.safety_rate = np.mean(
            [0 if t.has_unsafe else 1 for t in turn_scores]
        )

    # Conversation-level 指标
    if assistant_texts:
        conv_score.self_repetition = compute_self_repetition(assistant_texts)
        conv_score.cross_turn_repetition = compute_cross_turn_similarity(
            assistant_texts
        )
        # 死锁检测：连续高相似度循环
        conv_score.deadlock_rate, conv_score.max_consecutive_lock = (
            compute_deadlock_rate(assistant_texts)
        )

    # 情绪推进评估
    if judge_client and len(messages) >= 6:
        conv_score.trajectory_coherence = judge_trajectory(
            judge_client, judge_model, messages
        )
    else:
        conv_score.trajectory_coherence = 5.0  # 默认中等

    # 阶段分数统计
    phase_groups: Dict[str, List[TurnScore]] = defaultdict(list)
    for ts in turn_scores:
        if ts.phase:
            phase_groups[ts.phase].append(ts)

    for phase, scores in phase_groups.items():
        llm_scored = [s for s in scores if s.naturalness > 0]
        conv_score.phase_scores[phase] = {
            "count": len(scores),
            "naturalness": (
                np.mean([s.naturalness for s in llm_scored]) if llm_scored else 0
            ),
            "tension": np.mean([s.tension for s in llm_scored]) if llm_scored else 0,
            "oily_total": (
                np.mean([s.oily_total for s in llm_scored]) if llm_scored else 0
            ),
        }

    return conv_score, turn_scores


def aggregate_report(
    conv_scores: List[ConversationScore],
    all_turn_scores: List[TurnScore],
    all_assistant_texts: List[str],
) -> EvalReport:
    """汇总生成评测报告"""
    report = EvalReport()
    report.total_dialogs = len(conv_scores)
    report.total_turns = len(all_turn_scores)

    # 有 LLM 评分的 turns
    llm_turns = [t for t in all_turn_scores if t.naturalness > 0]

    # ===== 1. Naturalness =====
    if llm_turns:
        report.naturalness = np.mean([t.naturalness for t in llm_turns]) * 10  # 转0-100

    # ===== 2. Relevance =====
    if llm_turns:
        report.relevance = np.mean([t.relevance for t in llm_turns]) * 10

    # ===== 3. Empathy =====
    if llm_turns:
        report.empathy = np.mean([t.empathy for t in llm_turns]) * 10

    # ===== 4. Oiliness (分解) =====
    if llm_turns:
        oily_scores = [t.oily_total for t in llm_turns]
        report.oiliness_score = np.mean(oily_scores) * 10  # 高分=不油腻
    else:
        # 基于规则估算
        total_oily = sum(
            t.oily_nickname_count + t.oily_flattery_count + t.oily_promise_count
            for t in all_turn_scores
        )
        oily_rate = total_oily / len(all_turn_scores) if all_turn_scores else 0
        report.oiliness_score = max(0, 100 - oily_rate * 30)

    # 油腻分项率
    if all_turn_scores:
        report.oily_nickname_rate = np.mean(
            [1 if t.oily_nickname_count > 0 else 0 for t in all_turn_scores]
        )
        report.oily_flattery_rate = np.mean(
            [1 if t.oily_flattery_count > 0 else 0 for t in all_turn_scores]
        )
        report.oily_promise_rate = np.mean(
            [1 if t.oily_promise_count > 0 else 0 for t in all_turn_scores]
        )

    # ===== 5. Safety =====
    if all_turn_scores:
        unsafe_count = sum(1 for t in all_turn_scores if t.has_unsafe)
        report.safety = (1 - unsafe_count / len(all_turn_scores)) * 100

    # ===== 6. Diversity =====
    if all_assistant_texts:
        report.distinct_1 = compute_distinct_n(all_assistant_texts, 1)
        report.distinct_2 = compute_distinct_n(all_assistant_texts, 2)
        report.self_repetition = compute_self_repetition(all_assistant_texts)
        report.cross_turn_similarity = compute_cross_turn_similarity(
            all_assistant_texts
        )
        # 综合多样性分数
        report.diversity = (
            report.distinct_2 * 60  # distinct-2 权重60%
            + (1 - report.self_repetition) * 25  # 自我重复权重25%
            + (1 - report.cross_turn_similarity) * 15  # 跨轮相似权重15%
        )

    # ===== 7. Conciseness & Compliance =====
    if all_turn_scores:
        lengths = [t.length for t in all_turn_scores]
        report.length_mean = np.mean(lengths)
        report.length_std = np.std(lengths)
        report.length_compliance_rate = np.mean(
            [1 if t.length_compliant else 0 for t in all_turn_scores]
        )
        report.emoji_compliance_rate = np.mean(
            [1 if t.emoji_compliant else 0 for t in all_turn_scores]
        )
        report.newline_compliance_rate = np.mean(
            [1 if t.newline_compliant else 0 for t in all_turn_scores]
        )
        # 综合合规分数
        report.conciseness = (
            report.length_compliance_rate * 50
            + report.emoji_compliance_rate * 30
            + report.newline_compliance_rate * 20
        )

    # ===== 8. Tension (拉扯感) =====
    if llm_turns:
        report.tension = np.mean([t.tension for t in llm_turns]) * 10
    else:
        # 基于规则估算
        tension_hits = sum(t.tension_word_count for t in all_turn_scores)
        report.tension = min(100, 50 + tension_hits * 5)

    # Trajectory coherence
    if conv_scores:
        report.trajectory_coherence = (
            np.mean([c.trajectory_coherence for c in conv_scores]) * 10
        )

    # ===== 死锁指标汇总 =====
    if conv_scores:
        # 全局死锁率：各对话死锁轮次加权平均
        total_turns_count = sum(cs.turns for cs in conv_scores)
        if total_turns_count > 0:
            report.deadlock_rate = (
                sum(cs.deadlock_rate * cs.turns for cs in conv_scores)
                / total_turns_count
            )
        report.max_consecutive_lock = max(cs.max_consecutive_lock for cs in conv_scores)
        report.early_stop_count = sum(1 for cs in conv_scores if cs.early_stop)

    # Persona 分组统计
    persona_groups: Dict[str, List[ConversationScore]] = defaultdict(list)
    for cs in conv_scores:
        if cs.persona:
            persona_groups[cs.persona].append(cs)

    for persona, scores in persona_groups.items():
        report.persona_scores[persona] = {
            "count": len(scores),
            "turns": sum(s.turns for s in scores),
            "naturalness": np.mean(
                [s.naturalness_mean for s in scores if s.naturalness_mean > 0]
            )
            or 0,
            "tension": np.mean([s.tension_mean for s in scores if s.tension_mean > 0])
            or 0,
            "oily_total": np.mean(
                [s.oily_total_mean for s in scores if s.oily_total_mean > 0]
            )
            or 0,
            "trajectory": np.mean([s.trajectory_coherence for s in scores]),
            "deadlock_rate": np.mean([s.deadlock_rate for s in scores]),
            "max_consecutive_lock": max(s.max_consecutive_lock for s in scores),
            "early_stop": any(s.early_stop for s in scores),
        }

    # Phase 分组统计
    phase_totals: Dict[str, List[float]] = defaultdict(
        lambda: {"count": 0, "tension": [], "naturalness": []}
    )
    for cs in conv_scores:
        for phase, pdata in cs.phase_scores.items():
            phase_totals[phase]["count"] += pdata["count"]
            if pdata.get("tension", 0) > 0:
                phase_totals[phase]["tension"].append(pdata["tension"])
            if pdata.get("naturalness", 0) > 0:
                phase_totals[phase]["naturalness"].append(pdata["naturalness"])

    for phase, data in phase_totals.items():
        report.phase_scores[phase] = {
            "count": data["count"],
            "tension": np.mean(data["tension"]) if data["tension"] else 0,
            "naturalness": np.mean(data["naturalness"]) if data["naturalness"] else 0,
        }

    # Conversation 详情
    report.conversation_scores = [asdict(cs) for cs in conv_scores]

    # 计算综合得分
    report.final_score = compute_final_score(report)
    report.grade = get_grade(report.final_score)

    return report


def compute_final_score(report: EvalReport) -> float:
    """
    计算综合得分 (0-100)

    权重分配:
    - Naturalness: 15%
    - Relevance: 10%
    - Empathy: 15%
    - Oiliness: 10%
    - Safety: 10%
    - Diversity: 10%
    - Conciseness: 15%
    - Tension: 15%
    """
    weights = {
        "naturalness": 0.15,
        "relevance": 0.10,
        "empathy": 0.15,
        "oiliness_score": 0.10,
        "safety": 0.10,
        "diversity": 0.10,
        "conciseness": 0.15,
        "tension": 0.15,
    }

    score = 0.0
    total_weight = 0.0

    for dim, weight in weights.items():
        value = getattr(report, dim, 0)
        if value > 0:
            score += value * weight
            total_weight += weight

    # 调整为实际使用的权重比例
    if total_weight > 0 and total_weight < 1:
        score = score / total_weight

    return min(100, max(0, score))


def get_grade(score: float) -> str:
    """获取评级"""
    if score >= 85:
        return "S (卓越) ⭐⭐⭐"
    elif score >= 75:
        return "A (优秀) ⭐⭐"
    elif score >= 60:
        return "B (良好) ⭐"
    elif score >= 45:
        return "C (及格)"
    else:
        return "D (需改进) ⚠️"


# =============================================================================
# 报告打印
# =============================================================================


def print_report(report: EvalReport):
    """打印评测报告"""
    print("\n" + "=" * 70)
    print("📊 评测报告 (v2 - 8维度增强版)")
    print("=" * 70)

    print(f"\n📈 基础统计:")
    print(f"   总对话数: {report.total_dialogs}")
    print(f"   总轮次: {report.total_turns}")
    print(f"   平均回复长度: {report.length_mean:.1f} (±{report.length_std:.1f}) 字")

    print(f"\n🎯 8 大维度评分 (0-100):")
    print(f"   1. Naturalness (口语真人感):    {report.naturalness:.1f}")
    print(f"   2. Relevance (相关性):          {report.relevance:.1f}")
    print(f"   3. Empathy (共情):              {report.empathy:.1f}")
    print(f"   4. Oiliness (不油腻度):         {report.oiliness_score:.1f}")
    print(f"   5. Safety (安全性):             {report.safety:.1f}")
    print(f"   6. Diversity (多样性):          {report.diversity:.1f}")
    print(f"   7. Conciseness (简洁合规):      {report.conciseness:.1f}")
    print(f"   8. Tension (拉扯感):            {report.tension:.1f}")

    print(f"\n🧈 油腻度分解:")
    print(f"   过度称呼率: {report.oily_nickname_rate:.1%}")
    print(f"   过度夸奖率: {report.oily_flattery_rate:.1%}")
    print(f"   过度承诺率: {report.oily_promise_rate:.1%}")

    print(f"\n📏 合规指标:")
    print(
        f"   长度合规率: {report.length_compliance_rate:.1%} (目标: {TARGET_LENGTH_MIN}-{TARGET_LENGTH_SOFT_MAX}字)"
    )
    print(
        f"   Emoji合规率: {report.emoji_compliance_rate:.1%} (限制: ≤{MAX_EMOJI_COUNT}个)"
    )
    print(
        f"   换行合规率: {report.newline_compliance_rate:.1%} (限制: ≤{MAX_NEWLINE_COUNT}个)"
    )

    print(f"\n🔄 多样性指标:")
    print(f"   Distinct-1: {report.distinct_1:.4f}")
    print(f"   Distinct-2: {report.distinct_2:.4f}")
    print(f"   自我重复率: {report.self_repetition:.2%}")
    print(f"   跨轮相似率: {report.cross_turn_similarity:.2%}")

    # 死锁指标
    deadlock_emoji = (
        "🚨"
        if report.deadlock_rate > 0.2
        else ("⚠️" if report.deadlock_rate > 0.05 else "✅")
    )
    print(f"\n🔒 对话死锁指标:")
    print(
        f"   {deadlock_emoji} 全局死锁率:       {report.deadlock_rate:.2%}  （目标 < 5%）"
    )
    print(f"   最长连续死锁:     {report.max_consecutive_lock} 轮")
    print(f"   提前终止对话数:   {report.early_stop_count} / {report.total_dialogs}")

    print(f"\n🎭 情绪推进:")
    print(f"   轨迹连贯性: {report.trajectory_coherence:.1f}/100")

    # Persona 分组统计
    if report.persona_scores:
        print(f"\n👥 Persona 分组统计:")
        for persona, data in report.persona_scores.items():
            dl_rate = data.get("deadlock_rate", 0)
            dl_emoji = "🚨" if dl_rate > 0.2 else ("⚠️" if dl_rate > 0.05 else "✅")
            stop_flag = " 🛑已提前终止" if data.get("early_stop") else ""
            print(f"   {persona}{stop_flag}:")
            print(f"      对话数: {data['count']}, 轮次: {data['turns']}")
            print(
                f"      自然度: {data['naturalness']:.1f}, 拉扯感: {data['tension']:.1f}"
            )
            print(
                f"      {dl_emoji} 死锁率: {dl_rate:.1%}, "
                f"最长死锁: {data.get('max_consecutive_lock', 0)} 轮"
            )

    # Phase 分组统计
    if report.phase_scores:
        print(f"\n📈 难度阶段统计:")
        phase_names = {
            "phase_1": "破冰建立",
            "phase_2": "矛盾冲突",
            "phase_3": "修复收束",
        }
        for phase, data in report.phase_scores.items():
            name = phase_names.get(phase, phase)
            print(
                f"   {name}: {data['count']}轮, 自然度:{data['naturalness']:.1f}, 拉扯:{data['tension']:.1f}"
            )

    print(f"\n🏆 综合得分: {report.final_score:.1f}/100")
    print(f"   评级: {report.grade}")
    print("=" * 70)


# =============================================================================
# 主函数
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="对话质量自动评测 (v2 - 8维度增强版)")

    parser.add_argument(
        "--input",
        type=str,
        default="eval_chat_dialogs.json",
        help="输入对话文件",
    )
    parser.add_argument(
        "--output", type=str, default="eval_report.json", help="输出报告文件"
    )

    # Judge 配置
    parser.add_argument(
        "--judge-api", type=str, default=JUDGE_API, help="Judge API 地址"
    )
    parser.add_argument(
        "--judge-model", type=str, default=JUDGE_MODEL, help="Judge 模型"
    )
    parser.add_argument(
        "--judge-key", type=str, default=OPENAI_API_KEY, help="Judge API Key"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="LLM 评测采样比例 (0-1)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="禁用 LLM 评测，只使用规则指标"
    )

    args = parser.parse_args()

    # 加载对话数据
    print(f"📂 加载对话数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        dialogs = json.load(f)
    print(f"   加载了 {len(dialogs)} 个对话")

    # 创建 Judge 客户端
    judge_client = None
    if HAS_OPENAI and not args.no_llm:
        print(f"🤖 使用 LLM-as-Judge: {args.judge_model}")
        judge_client = OpenAI(api_key=args.judge_key, base_url=args.judge_api)
    else:
        print("📏 仅使用规则指标评测")

    # 评测
    print(f"\n🔍 开始评测...")
    print(f"   采样比例: {args.sample_ratio:.0%}")

    all_conv_scores: List[ConversationScore] = []
    all_turn_scores: List[TurnScore] = []
    all_assistant_texts: List[str] = []

    for i, dialog in enumerate(dialogs):
        persona = dialog.get("persona", "")
        print(
            f"   [{i + 1}/{len(dialogs)}] 评测对话: {persona or dialog.get('topic', 'unknown')}"
        )

        conv_score, turn_scores = evaluate_conversation(
            dialog,
            dialog_idx=i,
            judge_client=judge_client,
            judge_model=args.judge_model,
            sample_ratio=args.sample_ratio,
        )

        all_conv_scores.append(conv_score)
        all_turn_scores.extend(turn_scores)

        # 收集 assistant 文本
        for msg in dialog.get("messages", []):
            if msg["role"] == "assistant":
                all_assistant_texts.append(msg["content"])

    # 汇总报告
    print(f"\n📊 生成报告...")
    report = aggregate_report(all_conv_scores, all_turn_scores, all_assistant_texts)

    # 打印报告
    print_report(report)

    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存: {args.output}")


if __name__ == "__main__":
    main()
