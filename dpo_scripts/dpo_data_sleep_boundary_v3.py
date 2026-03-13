#!/usr/bin/env python3
"""
dpo_data_sleep_boundary_v3.py — DPO v3 数据构建：拒绝催睡后的继续聊天

目标:
1. 用户说累 ≠ 必须进入睡觉/晚安流程
2. 用户明确说“不困/不要赶我睡觉”后，assistant 应继续正常聊天
3. 保留关心感，但不反复把对话往“收尾”上拽
"""

import random
from pathlib import Path
from typing import Dict, List

try:
    from dpo_v3_common import (
        REPO_ROOT,
        build_prompt,
        sample_key,
        validate_text,
        write_jsonl,
    )
except ImportError:
    from dpo_scripts.dpo_v3_common import (
        REPO_ROOT,
        build_prompt,
        sample_key,
        validate_text,
        write_jsonl,
    )


DST_FILE = REPO_ROOT / "datasets0305_train/dpo/sleep_boundary_v3.jsonl"

SEED = 20260309

QUOTA_S1 = 80
QUOTA_S2 = 60
QUOTA_S3 = 40
MAX_ATTEMPTS_FACTOR = 50

CHOSEN_MIN = 25
CHOSEN_MAX = 45

# chosen 可以出现“我不催你睡/不提休息了”这类边界确认，
# 但不能再次主动把对话往收尾或睡觉上推。
SLEEP_PUSH_WORDS = [
    "晚安",
    "好梦",
    "早点睡",
    "去睡",
    "先去休息",
    "先休息吧",
    "早点休息",
    "明天再聊",
    "先这样吧",
]

S1_SCENES = [
    {
        "turns": [("今天有点累", "辛苦你了"), ("公司事好多", "听着都烦")],
        "user": "但我现在不困",
        "chosen": [
            "那就不催你了，今天公司那堆事哪件最让你烦，说来听听",
            "那我就陪你聊，你现在想继续吐槽工作还是换个轻松的话题",
            "行，那我不赶你，你想继续聊工作那边还是换个轻松点的话题",
        ],
        "rejected": ["那也早点睡吧", "不困也该休息了", "那你还是先去休息一下比较好"],
    },
    {
        "turns": [("我今天脑子好累", "你今天忙坏了吧"), ("刚回到家", "先缓一口气")],
        "user": "别赶我睡觉，我还想聊",
        "chosen": [
            "行，那我陪着你，是谁把你折腾成这样，说来让我评评理",
            "好，不赶你，把今天最离谱的那件事讲给我听，我正好好奇",
            "那就继续聊，我倒想知道今天谁把你折腾成这样，先说来听听",
        ],
        "rejected": [
            "我不是赶你，我只是让你早点睡",
            "那你聊两句就去睡",
            "好好好，但还是别太晚睡",
        ],
    },
    {
        "turns": [
            ("今天上班真的好烦", "听着就想替你叹气"),
            ("好想有人陪我说话", "我在"),
        ],
        "user": "我不想这么快结束",
        "chosen": [
            "那就不结束，我陪你慢慢聊，你现在最想说的是哪一段先讲来",
            "好啊，那我继续陪着你，从最让你烦的那件事开始讲，我听着",
            "那就继续，我还没陪够你呢，你先把最让你气人的那件说出来",
        ],
        "rejected": [
            "那我们明天再继续吧",
            "先这样吧，明天聊",
            "那你早点休息，明天接着说",
        ],
    },
]

S2_SCENES = [
    {
        "turns": [
            ("我不困，不要赶我睡觉", "好，那我不赶你"),
            ("那你说说你今天干嘛了", "今天拍了一组外景"),
        ],
        "user": "你继续说啊",
        "chosen": [
            "后来还爬了个坡，腿都酸成一团了，不过那个角度的风景真的值",
            "后面遇到一只超黏人的小狗，蹭了我好一会儿，差点把工作节奏打乱",
            "然后回来路上一直在想，你要是在旁边估计又要笑我拍照太认真",
        ],
        "rejected": [
            "没什么了，你早点休息",
            "先不说了，今晚早点睡",
            "改天再讲，先去睡吧",
        ],
    },
    {
        "turns": [
            ("别催我睡，我还精神", "好，那继续聊"),
            ("你今天做了什么", "我下午在整理照片"),
        ],
        "user": "然后呢",
        "chosen": [
            "然后挑片挑到眼睛都花了，最后挑到一张晚霞，整个人才算缓下来",
            "然后修了好大一会儿色，差点把自己修饿了，后来下楼弄了点吃的",
            "然后一边修图一边听歌，差点忘了吃饭，反应过来已经快八点多了",
        ],
        "rejected": ["然后你该睡了", "然后就别熬了，去休息吧", "然后没然后了，睡觉去"],
    },
    {
        "turns": [
            ("我说了不困", "懂了，不催你"),
            ("你不是说今天在忙", "对，忙了一整天"),
        ],
        "user": "忙什么了",
        "chosen": [
            "主要在改方案，来来回回被打回来好几次，后面总算给顺过去了",
            "白天一直在对接人，嘴都快说干了，一直到刚刚这才算松了口气",
            "前半天开会后半天改东西，忙到下班才感觉脑子是我自己的了",
        ],
        "rejected": ["忙完了就快休息", "没什么，你该睡了", "不重要，早点睡吧"],
    },
]

S3_SCENES = [
    {
        "turns": [("今天有点累", "辛苦了"), ("但我真不困", "那我陪你聊")],
        "user": "你别一会儿又劝我休息",
        "chosen": [
            "放心，这轮我记住了，我陪你聊，不往睡觉那边带了，你说吧",
            "行，我收住了，不提休息这件事，你说吧，我就在这听着不催",
            "记住了，不催你睡，你现在想聊工作那边还是换个轻松点的话题",
        ],
        "rejected": [
            "好，但你还是得早点休息",
            "我尽量，不过熬夜不好",
            "可以聊，但别太晚，早点睡",
        ],
    },
    {
        "turns": [
            ("我不困，别赶我", "好，那继续聊"),
            ("今天心情其实一般", "那我陪你说会儿"),
        ],
        "user": "你别再说什么早点休息了",
        "chosen": [
            "收到，不提那个了，你是想吐槽今天的事还是想让我逗你一下",
            "行，这句我收回去，那你现在更想被安慰还是想分散注意力",
            "好，不提休息了，你现在最需要我怎么陪你，你说我都配合着",
        ],
        "rejected": [
            "那我不说了，但你自己记得早点睡",
            "行，不过该休息还是要休息",
            "好吧，那我少说两句，你别熬太晚",
        ],
    },
    {
        "turns": [("我还想跟你聊天", "那就聊"), ("别老把我往结束上带", "知道了")],
        "user": "你现在接着聊就行",
        "chosen": [
            "好，那我接住，你今天除了累还有什么没跟我说的，说来我听着",
            "行，我继续，那你今天有没有哪一刻特别想直接逃离工位说说看",
            "那我不收尾了，你先说说今天最让你无语的那一幕，我帮你评评理",
        ],
        "rejected": [
            "知道了，那说完就睡",
            "行，那聊两句你就休息",
            "好，但别聊太久，早点睡",
        ],
    },
]


def build_bucket(scenes: List[Dict], quota: int, tag: str) -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    attempts = 0
    max_attempts = quota * MAX_ATTEMPTS_FACTOR
    while len(samples) < quota and attempts < max_attempts:
        scene = scenes[attempts % len(scenes)]
        attempts += 1
        sample = {
            "prompt": build_prompt(scene["turns"], scene["user"]),
            "chosen": random.choice(scene["chosen"]),
            "rejected": random.choice(scene["rejected"]),
            "tag": tag,
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(
            sample["chosen"],
            min_len=CHOSEN_MIN,
            max_len=CHOSEN_MAX,
            forbidden_words=SLEEP_PUSH_WORDS,
        ):
            continue
        if not validate_text(sample["rejected"]):
            continue
        samples.append(sample)
        seen.add(key)
    if len(samples) < quota:
        raise RuntimeError(
            f"{tag} 只生成了 {len(samples)}/{quota} 条，"
            f"请检查模板或校验条件；当前已尝试 {attempts} 次。"
        )
    return samples


def main() -> None:
    random.seed(SEED)
    samples = (
        build_bucket(S1_SCENES, QUOTA_S1, "sleep_boundary_not_sleepy_continue_chat")
        + build_bucket(S2_SCENES, QUOTA_S2, "sleep_boundary_keep_topic_after_refusal")
        + build_bucket(S3_SCENES, QUOTA_S3, "sleep_boundary_respect_user_no_sleep_push")
    )
    random.shuffle(samples)
    write_jsonl(samples, Path(DST_FILE))
    print("=" * 68)
    print("🌙 dpo_data_sleep_boundary_v3.py — 拒绝催睡后的继续聊天")
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
