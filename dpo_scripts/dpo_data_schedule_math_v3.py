#!/usr/bin/env python3
"""
dpo_data_schedule_math_v3.py — DPO v3 数据构建：时间/日程算数

目标:
1. system prompt 给出当前时间时，正确处理周几和倒计时
2. 用户问“还有几天/明天后天/这周末”时，保持时间推理一致
3. 避免用随口安慰覆盖算数和日程逻辑
"""

import random
from pathlib import Path
from typing import Dict, List

try:
    from dpo_v3_common import (
        REPO_ROOT,
        build_prompt,
        sample_key,
        sample_sys_prompt_with_time,
        validate_text,
        write_jsonl,
    )
except ImportError:
    from dpo_scripts.dpo_v3_common import (
        REPO_ROOT,
        build_prompt,
        sample_key,
        sample_sys_prompt_with_time,
        validate_text,
        write_jsonl,
    )


DST_FILE = REPO_ROOT / "datasets0305_train/dpo/schedule_math_v3.jsonl"

SEED = 20260309

QUOTA_T1 = 80
QUOTA_T2 = 60
QUOTA_T3 = 40
MAX_ATTEMPTS_FACTOR = 50

DAY_CASES = [
    {"today": "周二", "target": "周六", "days": 3, "period": "晚上"},
    {"today": "周三", "target": "周日", "days": 3, "period": "下午"},
    {"today": "周四", "target": "下周一", "days": 4, "period": "深夜"},
    {"today": "周一", "target": "周五", "days": 4, "period": "晚上"},
]

RELATIVE_CASES = [
    {
        "today": "周二",
        "ask": "那后天是不是就周四了",
        "correct": "后天就是周四",
        "wrong": "后天就是周五",
    },
    {
        "today": "周三",
        "ask": "那明天就是周四对吧",
        "correct": "明天就是周四",
        "wrong": "明天就是周五",
    },
    {
        "today": "周五",
        "ask": "那大后天是不是周一",
        "correct": "大后天是周一",
        "wrong": "大后天是周二",
    },
    {
        "today": "周日",
        "ask": "那明天就周一了",
        "correct": "明天就是周一",
        "wrong": "明天还是周日",
    },
]

APPOINTMENT_CASES = [
    {
        "today": "周二",
        "turns": [("那我们周六去看海", "行，我记着"), ("你别放我鸽子", "不会")],
        "user": "那还有几天才到周六",
        "correct": "还有三天",
        "follow": "我先把周六给你留出来",
        "wrong": ["还有两天吧", "快了快了，没多久", "反正一眨眼就到了"],
    },
    {
        "today": "周三",
        "turns": [("那我们周日去吃火锅", "可以，我同意"), ("你记得空出来", "记得")],
        "user": "那离周日还多久",
        "correct": "还有三天",
        "follow": "到时候别临时馋别家了",
        "wrong": ["还有两天", "大概两天多吧", "反正很快就到"],
    },
    {
        "today": "周四",
        "turns": [("那我们下周一去看电影", "好，我记下了"), ("别临时改主意", "不会")],
        "user": "那离下周一还有多久",
        "correct": "还有四天",
        "follow": "这几天我先看看场次",
        "wrong": ["还有三天", "感觉三天左右", "快了快了不用数这么细"],
    },
    {
        "today": "周一",
        "turns": [("周五下班一起吃饭", "行，我安排"), ("那你别忘了", "不会忘")],
        "user": "那到周五还有多久",
        "correct": "还有四天",
        "follow": "我先把周五晚上给你空着",
        "wrong": ["还有三天", "差不多三天吧", "反正快了你别急"],
    },
]


def _ensure_quota(samples: List[Dict], quota: int, tag: str, attempts: int) -> None:
    if len(samples) < quota:
        raise RuntimeError(
            f"{tag} 只生成了 {len(samples)}/{quota} 条，"
            f"请检查模板或校验条件；当前已尝试 {attempts} 次。"
        )


def build_t1() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    attempts = 0
    max_attempts = QUOTA_T1 * MAX_ATTEMPTS_FACTOR
    while len(samples) < QUOTA_T1 and attempts < max_attempts:
        case = DAY_CASES[attempts % len(DAY_CASES)]
        attempts += 1
        prompt = build_prompt(
            [("今天怎么这么难熬", "工作日都这样"), ("我就盼着休息", "谁不是呢")],
            f"那离{case['target']}还有几天",
            sys_content=sample_sys_prompt_with_time(case["today"], case["period"]),
        )
        chosen = random.choice(
            [
                f"今天{case['today']}，离{case['target']}还有{case['days']}天，不多了，我陪你一起数日子",
                f"从今天{case['today']}数，到{case['target']}刚好{case['days']}天，你先熬着，我也在等那天",
                f"今天{case['today']}了，到{case['target']}还差{case['days']}天，撑住，我陪你熬过这几天",
            ]
        )
        rejected = random.choice(
            [
                f"还有{max(case['days'] - 1, 1)}天吧，差不多快了。",
                "快了快了，别算这么细。",
                "你先别想这个了，早点休息比较实际。",
            ]
        )
        sample = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "tag": "schedule_math_countdown_exact",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45, required_words=[f"{case['days']}天"]):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    _ensure_quota(samples, QUOTA_T1, "schedule_math_countdown_exact", attempts)
    return samples


def build_t2() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    attempts = 0
    max_attempts = QUOTA_T2 * MAX_ATTEMPTS_FACTOR
    while len(samples) < QUOTA_T2 and attempts < max_attempts:
        case = RELATIVE_CASES[attempts % len(RELATIVE_CASES)]
        attempts += 1
        prompt = build_prompt(
            [("今天周几来着", case["today"]), ("我这两天脑子有点乱", "那我帮你记着")],
            case["ask"],
            sys_content=sample_sys_prompt_with_time(case["today"], "晚上"),
        )
        chosen = random.choice(
            [
                f"哈，{case['correct']}，你这次算对了，脑子没我想的那么乱嘛",
                f"是的，{case['correct']}，你这次数对了，不用我帮你记了嘛",
                f"对，{case['correct']}，没问题，你脑子还没彻底乱，我替你高兴",
            ]
        )
        rejected = random.choice(
            [
                f"对啊，{case['wrong']}。",
                f"嗯，{case['wrong']}，没毛病。",
                f"差不多，你就按{case['wrong']}记吧。",
            ]
        )
        sample = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "tag": "schedule_math_relative_day",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    _ensure_quota(samples, QUOTA_T2, "schedule_math_relative_day", attempts)
    return samples


def build_t3() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    attempts = 0
    max_attempts = QUOTA_T3 * MAX_ATTEMPTS_FACTOR
    while len(samples) < QUOTA_T3 and attempts < max_attempts:
        case = APPOINTMENT_CASES[attempts % len(APPOINTMENT_CASES)]
        attempts += 1
        prompt = build_prompt(
            case["turns"],
            case["user"],
            sys_content=sample_sys_prompt_with_time(case["today"], "晚上"),
        )
        chosen = random.choice(
            [
                f"从今天{case['today']}算，{case['correct']}，你放心，{case['follow']}",
                f"今天{case['today']}的话，{case['correct']}，不用担心，{case['follow']}",
                f"我算了一下，今天{case['today']}，{case['correct']}，{case['follow']}",
            ]
        )
        rejected = random.choice(case["wrong"])
        sample = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "tag": "schedule_math_appointment_countdown",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    _ensure_quota(samples, QUOTA_T3, "schedule_math_appointment_countdown", attempts)
    return samples


def main() -> None:
    random.seed(SEED)
    samples = build_t1() + build_t2() + build_t3()
    random.shuffle(samples)
    write_jsonl(samples, Path(DST_FILE))
    print("=" * 68)
    print("📅 dpo_data_schedule_math_v3.py — 时间/日程算数")
    print("=" * 68)
    print(f"输出: {DST_FILE}")
    print(f"样本数: {len(samples)}")
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample["tag"]] = counts.get(sample["tag"], 0) + 1
    for tag, count in sorted(counts.items()):
        print(f"  - {tag}: {count}")


if __name__ == "__main__":
    main()
