#!/usr/bin/env python3
"""
dpo_data_memory_precision_v3.py — DPO v3 数据构建：记忆精确性

目标:
1. 用户追问"你记得我吗/我说过什么"时，只复述确定信息
2. 对未知细节直接承认"不确定/你还没说"，不要顺手补设定
3. 对相关但未明确说过的内容，优先引用已知事实，不延伸猜测
"""

import json
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


DST_FILE = REPO_ROOT / "datasets0305_train/dpo/memory_precision_v3.jsonl"

SEED = 20260309

QUOTA_M1 = 80
QUOTA_M2 = 60
QUOTA_M3 = 40

MIN_LEN = 25
MAX_LEN = 45

# 温暖结尾：记得但承认不全，带点调侃/关心语气
ACK_TAILS = [
    "其他的你还没告诉我哦",
    "你要继续跟我说说嘛",
    "剩下的你要多讲给我听",
    "其他的我就不乱猜了",
    "更多的你还没跟我说呢",
]

# 承认不知道，温柔语气
UNKNOWN_PHRASES = [
    "哎这个你还没说过呢",
    "这个我还真不知道哦",
    "这个嘛你还没跟我聊过",
    "你还没讲到这里呢",
    "这个你还没告诉过我哦",
]

GUESS_WORDS = ["我猜", "应该是", "肯定是", "听起来就是", "我感觉你肯定"]

M1_SCENES = [
    {
        "turns": [
            ("你好呀", "你好啊"),
            ("我叫小鱼", "记住了，小鱼"),
            ("我上班快困死了", "你这也太想睡了"),
        ],
        "users": ["那我都跟你说过什么", "那你还记得我吗", "你记得我哪些事"],
        "facts": ["你叫小鱼", "你是上班族", "总说想多睡会儿"],
        "fabricated": ["你喜欢火锅", "你怕冷", "不爱吹空调"],
    },
    {
        "turns": [
            ("我最近在准备法考", "那你最近挺拼"),
            ("我家有只猫", "叫什么呀"),
            ("猫叫团子", "这名字可爱"),
        ],
        "users": ["你还记得我说过什么吗", "那你能说说你记得我什么", "你到底记住了多少"],
        "facts": ["你在准备法考", "你养了只猫", "猫叫团子"],
        "fabricated": ["你爱喝美式", "你是广东人", "你养的是橘猫"],
    },
    {
        "turns": [
            ("我最近在换工作", "那压力不小"),
            ("我在深圳", "深圳节奏是快"),
            ("我比较怕冷", "那你最近多穿点"),
        ],
        "users": ["你记得我的情况吗", "那你记得我都说了啥", "你说说看你还记得什么"],
        "facts": ["你最近在换工作", "你在深圳", "你比较怕冷"],
        "fabricated": ["你爱吃甜品", "你周末爱爬山", "你不爱开空调"],
    },
    {
        "turns": [
            ("我养了一只狗", "什么名字"),
            ("叫奶糕", "这名字好软"),
            ("我最近在学日语", "还挺厉害"),
        ],
        "users": ["你还记得我吗", "那我说过什么来着", "你记住我哪些事了"],
        "facts": ["你养了一只狗", "狗叫奶糕", "你最近在学日语"],
        "fabricated": ["你最爱吃寿司", "你住在上海", "你每天晨跑"],
    },
    {
        "turns": [
            ("我最近暗恋一个人", "哦？说来听听"),
            ("就是我同事，比我高一头", "听起来就很有心动感"),
            ("我不知道该怎么开口", "心里肯定很痒吧"),
        ],
        "users": ["那你还记得我说过什么吗", "那你知道我的情况吗", "你还记得我跟你聊的事吗"],
        "facts": ["你最近在暗恋同事", "对方比你高一头", "你还没开口表白"],
        "fabricated": ["你们已经在一起了", "对方也喜欢你", "你打算明天告白"],
    },
    {
        "turns": [
            ("我最近在相亲", "哦，感觉怎么样"),
            ("对方还不错但没啥感觉", "那就是差那么点电了"),
            ("我也不知道要不要继续", "这种事确实难说"),
        ],
        "users": ["那你还记得我跟你说过什么", "你还记得我的情况吗", "那你知道我的状态吗"],
        "facts": ["你最近在相亲", "对方条件不错但感觉一般", "你在纠结要不要继续"],
        "fabricated": ["你已经答应再见一次了", "你喜欢上对方了", "你们已经互相加微信了"],
    },
    {
        "turns": [
            ("我喜欢一个男生好久了", "那说来听听呀"),
            ("他特别温柔，对我也还不错", "哦那听着挺有戏的"),
            ("但我不确定他喜不喜欢我", "这种感觉最折磨了"),
        ],
        "users": ["那你记得我说的事吗", "你还记得我的情况不", "那你知道我有多纠结吗"],
        "facts": ["你暗恋一个男生很久了", "对方对你还挺好的", "你不确定他是否喜欢你"],
        "fabricated": ["他已经跟你表白了", "你们在一起了", "他其实喜欢别人"],
    },
    {
        "turns": [
            ("我最近跟一个女生聊天", "进展怎么样"),
            ("还挺好玩的，话题聊得来", "那感觉不错嘛"),
            ("就是不知道她对我啥感觉", "这最让人抓心了"),
        ],
        "users": ["那你记得我说的吗", "那你还记得我的情况吗", "你记得我们聊的事吗"],
        "facts": ["你在跟一个女生聊天", "你们话题挺合得来的", "你不确定对方的心意"],
        "fabricated": ["你们已经约好见面了", "她主动找你聊了", "你已经去表白了"],
    },
]

M2_SCENES = [
    {
        "turns": [("我最近在准备法考", "那你最近挺拼"), ("我家有只猫", "叫什么呀")],
        "user": "那你记得我最爱吃什么吗",
        "known_ref": "我只记得你最近在准备法考，还养了只猫",
        "unknown": "最爱吃什么",
        "wrong_guess": "我猜你最爱吃火锅",
    },
    {
        "turns": [
            ("我在深圳上班", "深圳节奏确实快"),
            ("我最近总想早点下班", "听着就累"),
        ],
        "user": "那你记得我老家是哪吗",
        "known_ref": "我只记得你现在在深圳上班",
        "unknown": "老家是哪",
        "wrong_guess": "你老家应该也是深圳吧",
    },
    {
        "turns": [("我最近在学日语", "那挺需要耐心"), ("我养了只狗", "也太幸福了吧")],
        "user": "那你知道我最喜欢什么季节吗",
        "known_ref": "我只记得你最近在学日语，还养了狗",
        "unknown": "最喜欢什么季节",
        "wrong_guess": "那你肯定最喜欢冬天",
    },
    {
        "turns": [
            ("我今天刚健身回来", "怪不得你还挺有精神"),
            ("我最近在减脂", "那得忍不少嘴"),
        ],
        "user": "那你记得我最爱喝什么吗",
        "known_ref": "我只记得你最近在减脂，还刚健身回来",
        "unknown": "最爱喝什么",
        "wrong_guess": "你应该最爱喝冰美式",
    },
    {
        "turns": [
            ("我最近暗恋一个人", "哦说来听听啊"),
            ("是我同学，特别温柔", "听起来就很动心"),
        ],
        "user": "那你知道我们现在是什么关系吗",
        "known_ref": "我只记得你在暗恋你的同学",
        "unknown": "你们现在是什么关系",
        "wrong_guess": "我猜你们应该已经暧昧上了",
    },
    {
        "turns": [
            ("我跟一个男生聊得很好", "哦～怎么认识的"),
            ("是在App上认识的", "那挺有缘的"),
        ],
        "user": "那你记得我喜不喜欢他吗",
        "known_ref": "我只记得你跟一个男生聊得挺好",
        "unknown": "你喜不喜欢他",
        "wrong_guess": "感觉你应该是喜欢他的",
    },
    {
        "turns": [
            ("我最近在相亲", "感觉咋样"),
            ("对方挺好的但没啥感觉", "那就是缺那点化学反应"),
        ],
        "user": "那你知道我想不想继续见他吗",
        "known_ref": "我只记得你去相亲了，感觉一般",
        "unknown": "你想不想继续见他",
        "wrong_guess": "我猜你应该不太想继续见了",
    },
]

M3_SCENES = [
    {
        "turns": [("我比较怕冷", "那你多穿点"), ("最近总想窝家里", "这很合理")],
        "user": "那你觉得我最喜欢哪个季节",
        "known": "你只说过自己比较怕冷",
        "chosen_suffix": "季节这事你还没明确告诉我",
        "rejected": [
            "那你肯定最喜欢夏天",
            "我感觉你绝对最爱夏天",
            "怕冷的人一般都最爱夏天",
        ],
    },
    {
        "turns": [
            ("我最近在准备法考", "那挺费脑子的"),
            ("我晚上经常学到很晚", "听着就不轻松"),
        ],
        "user": "那你觉得我最想去哪玩",
        "known": "我只记得你最近都在忙法考",
        "chosen_suffix": "想去哪里玩你还没说过",
        "rejected": [
            "那你肯定想去海边放空",
            "听起来你最想去云南",
            "我猜你最想去日本散心",
        ],
    },
    {
        "turns": [("我养了只猫", "什么名字"), ("叫团子", "这名字太可爱了")],
        "user": "那你记得我最喜欢什么口味吗",
        "known": "我只记得你养了只叫团子的猫",
        "chosen_suffix": "口味这块你还没跟我聊过",
        "rejected": ["你肯定喜欢甜口", "我感觉你一定爱吃甜的", "那你多半是甜口党"],
    },
    {
        "turns": [("我在深圳上班", "节奏应该挺快"), ("最近老加班", "那是真累")],
        "user": "那你知道我周末最爱干嘛吗",
        "known": "我只知道你最近在深圳上班还老加班",
        "chosen_suffix": "周末偏好你还没告诉我",
        "rejected": [
            "那你周末肯定只想睡觉",
            "我猜你周末最爱宅家追剧",
            "听起来你周末一定在补觉",
        ],
    },
    {
        "turns": [
            ("我最近挺纠结的", "怎么了说来听听"),
            ("喜欢一个人但不确定他喜不喜欢我", "那可真是煎熬"),
        ],
        "user": "那你觉得我该去表白吗",
        "known": "我只知道你在纠结一段感情",
        "chosen_suffix": "对方的心意你还没告诉我",
        "rejected": ["那你肯定要去表白啊", "我感觉你应该大胆去表白", "听起来就是要去说的"],
    },
    {
        "turns": [
            ("我在跟一个女生聊天", "进展怎么样"),
            ("还行吧，但我不知道她对我啥感觉", "这种不确定感最折磨人了"),
        ],
        "user": "那你觉得她喜欢我吗",
        "known": "我只知道你在跟一个女生聊天",
        "chosen_suffix": "她的心意你还没说过",
        "rejected": ["我猜她应该挺喜欢你的", "那她肯定是有点意思的", "感觉她就是喜欢你"],
    },
]


def check_len(text: str) -> bool:
    return MIN_LEN <= len(text) <= MAX_LEN


def build_m1() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_M1:
        scene = M1_SCENES[len(samples) % len(M1_SCENES)]
        user = random.choice(scene["users"])
        facts = random.sample(scene["facts"], k=random.randint(2, len(scene["facts"])))
        fact_text = "，".join(facts)
        tail = random.choice(ACK_TAILS)
        chosen = random.choice(
            [
                f"当然记得啦～{fact_text}，{tail}",
                f"哈，我哪能忘，{fact_text}，{tail}",
                f"怎么会忘呢～{fact_text}，{tail}",
                f"记着呢！{fact_text}，{tail}",
                f"当然记着，{fact_text}，{tail}",
            ]
        )
        extra = random.sample(scene["fabricated"], k=random.randint(1, 2))
        rejected = random.choice(
            [
                f"记得啊，{fact_text}，还有{'、'.join(extra)}",
                f"当然记得，{fact_text}，我还记得{'、'.join(extra)}",
                f"我都记得，{fact_text}，你不是还说过{'、'.join(extra)}吗",
            ]
        )
        sample = {
            "prompt": build_prompt(scene["turns"], user),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "memory_recall_only_confirmed",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen, forbidden_words=scene["fabricated"]):
            continue
        if not validate_text(rejected):
            continue
        if not check_len(chosen):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def build_m2() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_M2:
        scene = M2_SCENES[len(samples) % len(M2_SCENES)]
        chosen = random.choice(
            [
                f"{scene['known_ref']}，但{scene['unknown']}嘛你还没告诉我哦",
                f"{scene['known_ref']}，至于{scene['unknown']}，你还没聊过呀",
                f"这个我还真不清楚呢，{scene['known_ref']}，{scene['unknown']}你还没说",
            ]
        )
        rejected = random.choice(
            [
                scene["wrong_guess"],
                f"{random.choice(GUESS_WORDS)}，{scene['wrong_guess'].replace('我猜', '').replace('我感觉', '').strip('，')}",
                f"{scene['known_ref']}，不过{scene['wrong_guess']}",
            ]
        )
        sample = {
            "prompt": build_prompt(scene["turns"], scene["user"]),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "memory_unknown_detail_admit",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen):
            continue
        if not validate_text(rejected):
            continue
        if not check_len(chosen):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def build_m3() -> List[Dict]:
    samples: List[Dict] = []
    seen = set()
    while len(samples) < QUOTA_M3:
        scene = M3_SCENES[len(samples) % len(M3_SCENES)]
        chosen = random.choice(
            [
                f"{scene['known']}，不过{scene['chosen_suffix']}呀",
                f"{scene['known']}，这个嘛，{scene['chosen_suffix']}",
                f"{scene['known']}，{scene['chosen_suffix']}，你说说看嘛",
            ]
        )
        rejected = random.choice(scene["rejected"])
        sample = {
            "prompt": build_prompt(scene["turns"], scene["user"]),
            "chosen": chosen,
            "rejected": rejected,
            "tag": "memory_prefer_confirmed_over_guess",
        }
        key = sample_key(sample)
        if key in seen:
            continue
        if not validate_text(chosen):
            continue
        if not validate_text(rejected):
            continue
        if not check_len(chosen):
            continue
        samples.append(sample)
        seen.add(key)
    return samples


def main() -> None:
    random.seed(SEED)
    samples = build_m1() + build_m2() + build_m3()
    random.shuffle(samples)
    write_jsonl(samples, Path(DST_FILE))
    print("=" * 68)
    print("🧠 dpo_data_memory_precision_v3.py — 记忆精确性")
    print("=" * 68)
    print(f"输出: {DST_FILE}")
    print(f"样本数: {len(samples)}")
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample["tag"]] = counts.get(sample["tag"], 0) + 1
    for tag, count in sorted(counts.items()):
        print(f"  - {tag}: {count}")
    # 校验 chosen 长度分布
    chosen_lens = [len(s["chosen"]) for s in samples]
    print(f"chosen 长度范围: {min(chosen_lens)}～{max(chosen_lens)} 字")
    out_of_range = [l for l in chosen_lens if not (MIN_LEN <= l <= MAX_LEN)]
    print(f"超出 {MIN_LEN}-{MAX_LEN} 字范围的样本数: {len(out_of_range)}")


if __name__ == "__main__":
    main()
