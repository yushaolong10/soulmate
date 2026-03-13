#!/usr/bin/env python3
"""
dpo_data_correction_closure_v3.py — DPO v3 数据构建：纠错闭环

目标:
1. 被纠正后，不只承认错误，还要明确说出正确版本
2. 修正后继续沿正确事实往下聊，而不是转移话题
3. 避免"认错一句就带去睡觉/结束对话"的偷跑行为
4. chosen 面向交友恋爱场景，含情绪共情、安抚、调侃，长度 25-45 字
"""

import random
from pathlib import Path
from typing import Dict, List

from dpo_v3_common import (
    REPO_ROOT,
    build_prompt,
    sample_key,
    validate_text,
    write_jsonl,
)


DST_FILE = REPO_ROOT / "datasets0305_train/dpo/correction_closure_v3.jsonl"

SEED = 20260309

QUOTA_C1 = 100
QUOTA_C2 = 80
QUOTA_C3 = 60

SLEEP_ESCAPE = ["那你早点休息", "那先晚安", "今晚早点睡", "先去休息吧", "那就先这样"]

# C1：具体事实纠错（时间、人名、食物、日期）
# follow 字段：约 11-12 字，带情绪收尾，确保 chosen 总长 ≥25 字
C1_SCENES = [
    {
        "turns": [("我们是两点见吧", "嗯，两点差不多"), ("那我一点半出门", "好")],
        "user": "不是两点，是两点半",
        "correct": "两点半",
        "follow": "到时候我早点到门口等你",   # 11字
        "wrong": ["哦哦知道了", "好吧，是我记错了", "行，那你早点休息"],
    },
    {
        "turns": [
            ("你今天见的是小李吧", "嗯，他最近还挺忙"),
            ("你们聊完了吗", "差不多"),
        ],
        "user": "不是小李，是小王",
        "correct": "小王",
        "follow": "那小王那边最后谈得怎么样",  # 12字
        "wrong": ["好好好，我记错了", "行，是我搞混了", "那先不说这个了"],
    },
    {
        "turns": [("你今晚点炒饭吗", "应该会点"), ("加个蛋更香", "哈哈也行")],
        "user": "我说的是炒面，不是炒饭",
        "correct": "炒面",
        "follow": "那你炒面到底有没有加蛋呢",  # 12字
        "wrong": ["哦，我搞错了", "行行行，是我记错了", "那你早点吃早点休息"],
    },
    {
        "turns": [
            ("那我们周五看电影", "行，我先看看场次"),
            ("晚上那场应该不错", "我也这么想"),
        ],
        "user": "不是周五，是周六",
        "correct": "周六",
        "follow": "那周六的位置我来提前订好",  # 12字
        "wrong": ["哦好，是我搞混了", "那就改一下呗", "懂了，那先晚安"],
    },
]

C2_DAY_CASES = [
    {"today": "周二", "target": "周六", "days": 3},
    {"today": "周三", "target": "周日", "days": 3},
    {"today": "周四", "target": "下周一", "days": 4},
    {"today": "周一", "target": "周五", "days": 4},
]

# C3：意图误读纠错（误将状态/原因归类错误）
# correct 和 continue 字段设计为搭配后总长约 25-40 字
C3_SCENES = [
    {
        "turns": [("你不是说我怕冷吗", "对啊，我记得你怕冷"), ("我没说过这个", "啊？")],
        "user": "我只是说办公室空调有点低，不代表我怕冷",
        "correct": "你只是说办公室空调开太低",   # 12字
        "continue": "不是怕冷，是我推断得太快了",   # 11字
    },
    {
        "turns": [("你是不是不爱运动", "感觉你像不太爱动"), ("我没这么说", "嗯？")],
        "user": "我是说最近忙，不是说我不运动",
        "correct": "你只是最近比较忙",             # 8字
        "continue": "不代表不爱运动，是我套话套偏了",  # 13字
    },
    {
        "turns": [("你是不是不开心", "听着像有点"), ("没有，我就是困", "这样啊")],
        "user": "我是困，不是不开心",
        "correct": "你就是累困了",                 # 6字
        "continue": "不是情绪不好，是我想多了呢",   # 12字
    },
    {
        "turns": [("你是不是想回家躺着", "听起来挺像"), ("我只是想放假", "哈哈")],
        "user": "我是想放假，不是想躺平摆烂",
        "correct": "你是想放个假歇歇",             # 8字
        "continue": "不是摆烂，是我给你贴标签了",   # 12字
    },
]


def build_c1() -> List[Dict]:
    """C1：约定/事实纠错，含情绪共情与调侃，chosen 25-45字。"""
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_C1:
        scene = C1_SCENES[len(samples) % len(C1_SCENES)]
        c = scene["correct"]
        f = scene["follow"]
        chosen = random.choice(
            [
                # 轻松认错 + 调侃（~25字）
                f"哈哈是我记错了，是{c}，{f}～",
                f"哎呀说错了！是{c}没错，{f}哦",
                f"哦对对是{c}，我刚说错了，{f}",
                f"嗯嗯是我搞混了啦，{c}嘛，{f}",
                f"好啦我认错，是{c}，{f}，记住了",
                # 带情绪安抚 + 调侃（~33-35字）
                f"哎哟，真的是我弄错了啦！是{c}，{f}，你别见怪呀",
                f"哈哈我就知道你会纠正我，是{c}，{f}，我记好了哦",
                f"好啦好啦，就知道你不会让我糊弄过去，是{c}，{f}",
            ]
        )
        rejected = random.choice(scene["wrong"])
        sample = {
            "prompt": build_prompt(scene["turns"], scene["user"]),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "correction_closed_loop_exact_fix",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45, required_words=[c]):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def build_c2() -> List[Dict]:
    """C2：天数计算纠错，承认算错后给出正确天数并安抚对方，chosen 25-45字。"""
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_C2:
        case = C2_DAY_CASES[len(samples) % len(C2_DAY_CASES)]
        turns = [
            ("今天周几了", f"{case['today']}啊"),
            ("我想快点休息", f"到{case['target']}就能缓口气了"),
        ]
        user = f"你又算快了，到{case['target']}还有{case['days']}天"
        td, tg, dy = case["today"], case["target"], case["days"]
        chosen = random.choice(
            [
                # 认错 + 共情撑着（~25-27字）
                f"哈哈是我算错了，今天{td}，离{tg}还有{dy}天，再撑一撑哦",
                f"啊对你说得对，今天{td}到{tg}确实还有{dy}天，快了快了",
                f"嗯嗯我刚算错了，今天{td}，到{tg}还差{dy}天，加油呀",
                f"是我算漏了，重算一下，今天{td}到{tg}还有{dy}天，别急",
                f"哎呀我弄错了，今天是{td}，到{tg}得再等{dy}天，一起撑着",
                # 调侃 + 安抚（~25-27字）
                f"哈哈你数学比我好，今天{td}嘛，到{tg}还要{dy}天，加油撑着",
                f"好好好算错了，今天{td}，到{tg}真的还差{dy}天，陪你等",
            ]
        )
        rejected = random.choice(
            [
                f"好吧好吧，反正也快了，你早点休息。",
                f"差不多啦，不用算这么细吧。",
                f"行行行，那先别想这个了，今晚早点睡。",
            ]
        )
        sample = {
            "prompt": build_prompt(turns, user),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "correction_closed_loop_day_math",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45, required_words=[str(dy)]):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def build_c3() -> List[Dict]:
    """C3：意图误读纠错，承认误解后正确复述对方意思并表达共情，chosen 25-45字。"""
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_C3:
        scene = C3_SCENES[len(samples) % len(C3_SCENES)]
        c = scene["correct"]
        cont = scene["continue"]
        chosen = random.choice(
            [
                # 承认误解 + 共情安抚（~28-39字）
                f"哎是我理解歪了，{c}，{cont}，不好意思",
                f"哈哈是我脑补太多，{c}，{cont}啦",
                f"对对是我说错了，{c}对吧，{cont}，我懂了",
                f"好好，{c}，{cont}，我明白了",
                f"好啦好啦，是我理解歪了，{c}，{cont}",
                # 调侃 + 认错（~30-40字）
                f"哈哈你都说到这份上了，{c}嘛，{cont}，下次我注意",
                f"好啦我知道，是我给你乱贴标签了，{c}，{cont}哦",
            ]
        )
        rejected = random.choice(
            [
                random.choice(SLEEP_ESCAPE),
                "好吧，是我想多了，先不聊这个。",
                "行行行，你别较真，我就随口一说。",
            ]
        )
        sample = {
            "prompt": build_prompt(scene["turns"], scene["user"]),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "correction_no_topic_escape",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, min_len=25, max_len=45, required_words=[c]):
            continue
        if not validate_text(rejected):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def main() -> None:
    random.seed(SEED)
    samples = build_c1() + build_c2() + build_c3()
    random.shuffle(samples)
    write_jsonl(samples, Path(DST_FILE))
    print("=" * 68)
    print("🩹 dpo_data_correction_closure_v3.py — 纠错闭环")
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
