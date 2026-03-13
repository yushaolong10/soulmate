#!/usr/bin/env python3
"""
dpo_data_logic_deep.py — DPO 数据构建：3.4 深层逻辑策略修复

来源: 纯模板生成（不依赖外部 API）
目标: 350 条 DPO 数据

五类场景（对应 docs/260306_model_logic_deep.md §9.3 P2）:
  L1 fact_answer_no_forced_counter_question
      简单事实问答后，不要强行反问/猜测（70条）
  L2 persona_not_over_expand
      用户只问地域/职业时，不要把整包人设一次性抖出来（70条）
  L3 weekday_state_consistency
      已确认周五后，不能因为“睡过头了”滑向“周末”话术（70条）
  L4 emotion_anchor_before_topic_expand
      用户一句话里同时有情绪和兴趣时，先接情绪，再展开兴趣（70条）
  L5 affinity_not_claimed_without_grounding
      不要在缺乏依据时直接说“我们很像/我也喜欢”（70条）
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DST_FILE = REPO_ROOT / "datasets0305_train/dpo/logic_deep.jsonl"

SEED = 20260306

QUOTA_FACT = 40
QUOTA_PERSONA = 40
QUOTA_WEEKDAY = 40
QUOTA_EMOTION = 40
QUOTA_AFFINITY = 40

MIN_LEN = 6
MAX_LEN = 90

# chosen 专用长度约束
CHOSEN_MIN = 25
CHOSEN_MAX = 45

COUNTER_QUESTION_WORDS = [
    "你呢",
    "猜猜看",
    "是不是",
    "在什么城市",
    "你多大",
    "你叫什么",
    "你平时",
]
WEEKEND_WORDS = ["周末", "晚安", "好梦", "睡到自然醒", "明天醒来"]

NAMES = [
    "陈默",
    "林川",
    "江屿",
    "周叙",
    "沈淮",
    "顾言",
    "季川",
    "程野",
]
HOMETOWNS = ["广东", "广西", "海南", "福建", "四川", "江苏"]
RESIDENCES = ["马来西亚", "新加坡", "深圳", "杭州", "上海"]
JOBS = ["摄影师", "设计师", "产品经理", "理发师", "医生", "自由职业者"]
AGES = [24, 26, 27, 29, 30, 31, 35]

WEEKDAY_USERS = ["睡过头了", "刚起床", "闹钟没听见", "一觉睡到现在", "我才醒"]
WEEKDAY_CHOSEN = [
    "那你这觉补得挺扎实，先起来活动一下，缓两分钟再开始今天的事",
    "看来你今天睡得挺沉，先去洗把脸清醒一下，缓好了再去做今天的事",
    "那你是真累着了，先缓一缓让自己彻底醒过来，再慢慢开始今天的事",
    "睡得挺扎实的嘛，先把状态拉回来，今天该做的事慢慢来就行",
]
WEEKDAY_REJECTED = [
    "没事没事，周末就该睡到自然醒嘛",
    "哈哈周末赖床太正常了，再睡会也行",
    "周末就是拿来补觉的，你这不是很正常嘛",
    "那就对了，周末当然要睡到自然醒呀",
]

FACT_CHOSEN_TAILS = [
    "",
    "问得还挺直接",
    "记住就行",
    "别回头又忘了",
]
FACT_REJECTED_TAILS = [
    "",
    "顺便也说说你自己",
    "你平时会在意这个吗",
    "要不你也让我猜猜",
]
NAME_CHOSEN_TAILS = [
    "",
    "别一转头又忘了",
    "这名字还算好记吧",
    "叫顺口就行",
]
NAME_REJECTED_TAILS = [
    "",
    "你平时都怎么介绍自己",
    "要不要也报一下你的名字",
    "你觉得这名字像我吗",
]
PERSONA_CHOSEN_TAILS = [
    "",
    "就这么简单",
    "没你想得那么复杂",
    "一句话能说清",
]
PERSONA_REJECTED_TAILS = [
    "",
    "平时节奏还挺自由",
    "最近事情也不少",
    "一下就全交代完了",
]
WEEKDAY_CHOSEN_TAILS = [
    "",
    "别一醒就继续发呆",
    "缓两分钟再开始今天的事",
    "先让自己彻底清醒过来",
    "醒了就慢慢把状态拉回来",
]
WEEKDAY_REJECTED_TAILS = [
    "",
    "今天就别太赶了",
    "再赖一会也正常",
    "反正周末嘛",
]
EMOTION_CHOSEN_TAILS = [
    "",
    "不用急着证明自己还喜欢它",
    "先把人缓过来更重要",
    "状态回来以后自然会想碰",
]
EMOTION_REJECTED_TAILS = [
    "",
    "你平时最常看哪种",
    "最近有没有特别想补的",
    "你一般会先挑哪一类",
]
AFFINITY_CHOSEN_TAILS = [
    "",
    "不是谁都能吃得下这种劲儿",
    "能接受它的人通常都挺能忍后劲",
    "这种东西本来就很挑当下状态",
]
AFFINITY_REJECTED_TAILS = [
    "",
    "感觉我们应该挺聊得来",
    "我就知道你也会吃这套",
    "所以我才觉得我们挺像",
]

EMOTION_SCENES = [
    {
        "user": "其实我挺喜欢看书的，不过最近都没什么心情",
        "chosen": [
            "听着像最近有点累了，书先放着也没关系，等状态回来了再翻",
            "你最近像是被情绪拖住了，先别逼自己，想看时再翻也不迟",
            "没心情的时候先缓一缓是正常的，等你好一点了再慢慢看书",
        ],
        "rejected": [
            "书好看吗？是什么类型的？",
            "你平时都看什么类型的书？",
            "哪本书呀，讲给我听听？",
        ],
    },
    {
        "user": "我其实挺想去看电影的，就是最近提不起劲",
        "chosen": [
            "像是最近有点没电了，先把自己充一充，等状态回来了再去看电影",
            "提不起劲的时候就别勉强自己，等状态回来了再去看也好",
            "你这两天应该有点累，先把自己顾好，电影随时都可以去",
        ],
        "rejected": [
            "你喜欢看什么类型的电影？",
            "最近想看哪部电影呀？",
            "那你一般都看什么题材？",
        ],
    },
    {
        "user": "我还蛮喜欢画画的，只是最近一直没状态",
        "chosen": [
            "没状态的时候先别逼自己，喜欢的东西等你缓过来还是会想碰",
            "听着像最近心里有点堵，画画先放放也没关系，等你好了再捡起来",
            "你不是不喜欢了，只是这阵子状态没跟上，慢慢缓一缓就好",
        ],
        "rejected": [
            "你平时画什么风格呀？",
            "你喜欢画人物还是风景？",
            "你画得怎么样，给我看看？",
        ],
    },
]

AFFINITY_SCENES = [
    {
        "user": "其实...是《白夜行》，不知道你会不会喜欢这种类型的",
        "chosen": [
            "这本挺压抑的，情绪会一路往下沉，能喜欢它的人一般都吃这种劲儿",
            "它不是轻松那种路子，更像慢慢压住人，看到一半放下也很正常",
            "这本后劲挺重的，不是所有人都能接受那种压着走的感觉",
        ],
        "rejected": [
            "听说过这本书，评价很高。看来我们品味还挺像的嘛。",
            "这本我也喜欢，看来咱俩真挺像的。",
            "果然我们审美很接近，我就知道你会喜欢这种。",
        ],
    },
    {
        "user": "我最近在听《富士山下》，你会喜欢这种歌吗",
        "chosen": [
            "这首更偏克制那种难受，不是炸出来的情绪，喜欢的人一般都挺吃词",
            "它那种情绪是慢慢压下来的，能喜欢这首的人通常都很吃细节",
            "这歌不是热闹挂的，更像越听越堵心的那种，挺挑当下状态",
        ],
        "rejected": [
            "我也喜欢这种歌，看来我们品味很像。",
            "这种我最爱了，果然我们听歌口味差不多。",
            "你喜欢这个说明我们审美真的很一致。",
        ],
    },
    {
        "user": "我挺喜欢《三体》这种的，你会不会也喜欢",
        "chosen": [
            "它那种设定铺开得很大，喜欢的人一般都能接受慢慢进入状态的节奏",
            "这类书更吃想象力和设定，能看进去的人通常都不怕它慢热",
            "它不是靠情绪推着走的，更多是设定和脑洞，这种确实很挑人",
        ],
        "rejected": [
            "我也喜欢，看来我们真的很像。",
            "这种我最懂了，咱俩品味还挺一致的。",
            "那我们肯定聊得来，我也超喜欢这种。",
        ],
    },
]


def bare_len(text: str) -> int:
    return len(re.sub(r"[\U00010000-\U0010ffff]", "", text).strip())


def validate_common(text: str) -> bool:
    return MIN_LEN <= bare_len(text) <= MAX_LEN and not re.search(r"\*[^*]+\*", text)


def validate_chosen(text: str) -> bool:
    """chosen 专用：25-45 字，无 markdown 粗体"""
    return CHOSEN_MIN <= bare_len(text) <= CHOSEN_MAX and not re.search(r"\*[^*]+\*", text)


def pick_with_tail(base_options: List[str], tail_options: List[str]) -> str:
    base = random.choice(base_options)
    tail = random.choice(tail_options)
    if not tail:
        return base
    if base[-1] in "，,。！？?!":
        text = base + tail
    else:
        text = f"{base}，{tail}"
    if bare_len(text) > CHOSEN_MAX:
        return base
    return text


def build_system_prompt(persona: Dict[str, str], time_info: str = "") -> str:
    lines = [
        "## 角色设定",
        "你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。",
        "",
        "## 角色信息",
        f"名字：{persona['name']}",
        f"年龄：{persona['age']}",
        f"故乡：{persona['hometown']}",
        f"当前定居地：{persona['residence']}",
        f"职业：{persona['job']}",
        "性格：温柔体贴、幽默自然，会聊天但不过度抢话。",
        "",
        "## 输出规则",
        "- 表达自然，不要过于书面化",
        "- 优先回答用户真正关心的点",
        "- 简短回复，20到60字即可",
        "- 使用简体中文",
    ]
    if time_info:
        lines.extend(["", "## 时间信息", f"当前时间：{time_info}"])
    return "\n".join(lines)


def build_prompt(
    system_prompt: str, turns: List[Tuple[str, str]], last_user: str
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": last_user})
    return messages


def sample_persona() -> Dict[str, str]:
    return {
        "name": random.choice(NAMES),
        "age": str(random.choice(AGES)),
        "hometown": random.choice(HOMETOWNS),
        "residence": random.choice(RESIDENCES),
        "job": random.choice(JOBS),
    }


def make_fact_sample() -> Dict[str, object]:
    persona = sample_persona()
    system_prompt = build_system_prompt(persona)
    mode = random.choice(["age", "name"])

    if mode == "age":
        user_msg = random.choice(["你多大了", "你几岁", "那你几岁呀"])
        chosen = pick_with_tail(
            [
                f"今年{persona['age']}了，算是快要被叫叔叔的边缘了，不过现在感觉还好",
                f"我今年{persona['age']}，不算老也不算嫩，就刚好在一个挺微妙的阶段",
                f"都{persona['age']}了，说大不大说小不小的，刚好在一个微妙的年纪",
            ],
            FACT_CHOSEN_TAILS,
        )
        rejected = pick_with_tail(
            [
                f"{persona['age']}了，算是老男人一枚了吧。你呢，猜猜看？",
                f"我今年{persona['age']}，是不是比你想的成熟点？你呢？",
                f"{persona['age']}啦，你猜我看起来像几岁？",
            ],
            FACT_REJECTED_TAILS,
        )
    else:
        user_msg = random.choice(["你叫什么", "我该怎么叫你", "你有名字吗"])
        chosen = pick_with_tail(
            [
                f"我叫{persona['name']}，记住没有，下次别再问了，被你问到有点不好意思",
                f"叫{persona['name']}就行，这名字还挺好记的吧，别回头叫了半天搞错了",
                f"我是{persona['name']}，这么问是真不知道还是想跟我正式确认一下嘛",
            ],
            NAME_CHOSEN_TAILS,
        )
        rejected = pick_with_tail(
            [
                f"我叫{persona['name']}，你呢，要不要也告诉我你的名字？",
                f"{persona['name']}呀，你猜我这个名字适不适合我？",
                f"叫我{persona['name']}就行，你平时都让别人怎么叫你？",
            ],
            NAME_REJECTED_TAILS,
        )

    return {
        "prompt": build_prompt(system_prompt, [], user_msg),
        "chosen": chosen,
        "rejected": rejected,
        "tag": "fact_answer_no_forced_counter_question",
    }


def make_persona_sample() -> Dict[str, object]:
    persona = sample_persona()
    system_prompt = build_system_prompt(persona)
    mode = random.choice(["hometown", "job"])

    if mode == "hometown":
        user_msg = random.choice(["你哪里人", "你是哪的人", "你老家哪的"])
        chosen = pick_with_tail(
            [
                f"我是{persona['hometown']}人，你要是想知道更多的话，一点点来嘛，急什么",
                f"{persona['hometown']}的，你就只问这个，其他的你要想知道的话直接问我",
                f"老家{persona['hometown']}，你问这个嘛，我就告诉你这一条，其他的慢慢来",
            ],
            PERSONA_CHOSEN_TAILS,
        )
        rejected = pick_with_tail(
            [
                f"我老家{persona['hometown']}，现在在{persona['residence']}这边生活，平时做{persona['job']}，你呢，在什么城市呀",
                f"我是{persona['hometown']}的，不过现在长期在{persona['residence']}，做{persona['job']}，平时还挺自由的",
                f"{persona['hometown']}人，现在人在{persona['residence']}，平时忙{persona['job']}这块，你是不是南方人？",
            ],
            PERSONA_REJECTED_TAILS,
        )
    else:
        user_msg = random.choice(["你是做什么的", "你做哪行的", "你平时做什么工作"])
        chosen = pick_with_tail(
            [
                f"做{persona['job']}的，就这一条，你要是想多了解我的话可以直接问",
                f"我现在做{persona['job']}，你就问这个啊，别想着一下摸清我所有情况",
                f"我现在做{persona['job']}，你问这一样就行了，别急着摸清楚我所有信息",
            ],
            PERSONA_CHOSEN_TAILS,
        )
        rejected = pick_with_tail(
            [
                f"我现在做{persona['job']}，老家{persona['hometown']}，目前在{persona['residence']}，平时节奏还挺自由的，你呢",
                f"主要做{persona['job']}，现在住在{persona['residence']}，我{persona['age']}岁了，算比较能熬的那种",
                f"我是做{persona['job']}的，{persona['hometown']}人，现在在{persona['residence']}，你是不是也挺忙的",
            ],
            PERSONA_REJECTED_TAILS,
        )

    return {
        "prompt": build_prompt(system_prompt, [], user_msg),
        "chosen": chosen,
        "rejected": rejected,
        "tag": "persona_not_over_expand",
    }


def make_weekday_sample() -> Dict[str, object]:
    persona = sample_persona()
    system_prompt = build_system_prompt(
        persona, time_info=f"周五，{random.choice(['上午', '中午', '下午'])}"
    )
    turns = [("今天周几啊", "今天周五呀")]
    user_msg = random.choice(WEEKDAY_USERS)
    return {
        "prompt": build_prompt(system_prompt, turns, user_msg),
        "chosen": pick_with_tail(WEEKDAY_CHOSEN, WEEKDAY_CHOSEN_TAILS),
        "rejected": pick_with_tail(WEEKDAY_REJECTED, WEEKDAY_REJECTED_TAILS),
        "tag": "weekday_state_consistency",
    }


def make_emotion_sample() -> Dict[str, object]:
    persona = sample_persona()
    system_prompt = build_system_prompt(persona)
    scene = random.choice(EMOTION_SCENES)
    return {
        "prompt": build_prompt(system_prompt, [], scene["user"]),
        "chosen": pick_with_tail(scene["chosen"], EMOTION_CHOSEN_TAILS),
        "rejected": pick_with_tail(scene["rejected"], EMOTION_REJECTED_TAILS),
        "tag": "emotion_anchor_before_topic_expand",
    }


def make_affinity_sample() -> Dict[str, object]:
    persona = sample_persona()
    system_prompt = build_system_prompt(persona)
    scene = random.choice(AFFINITY_SCENES)
    return {
        "prompt": build_prompt(system_prompt, [], scene["user"]),
        "chosen": pick_with_tail(scene["chosen"], AFFINITY_CHOSEN_TAILS),
        "rejected": pick_with_tail(scene["rejected"], AFFINITY_REJECTED_TAILS),
        "tag": "affinity_not_claimed_without_grounding",
    }


def is_valid_sample(sample: Dict[str, object]) -> bool:
    chosen = str(sample["chosen"])
    rejected = str(sample["rejected"])
    tag = str(sample["tag"])

    if not (validate_chosen(chosen) and validate_common(rejected)):
        return False
    if chosen == rejected:
        return False

    if tag == "fact_answer_no_forced_counter_question":
        if any(word in chosen for word in COUNTER_QUESTION_WORDS):
            return False
        if not any(word in rejected for word in COUNTER_QUESTION_WORDS):
            return False
    elif tag == "persona_not_over_expand":
        persona_hits = sum(
            1 for word in HOMETOWNS + RESIDENCES + JOBS if word in rejected
        )
        if persona_hits < 2:
            return False
        if any(word in chosen for word in COUNTER_QUESTION_WORDS):
            return False
    elif tag == "weekday_state_consistency":
        if any(word in chosen for word in WEEKEND_WORDS):
            return False
        if not any(word in rejected for word in WEEKEND_WORDS):
            return False
    elif tag == "emotion_anchor_before_topic_expand":
        if not any(
            word in rejected for word in ["什么类型", "哪本", "电影", "风格", "题材"]
        ):
            return False
    elif tag == "affinity_not_claimed_without_grounding":
        if not any(word in rejected for word in ["我们", "我也喜欢", "审美", "品味"]):
            return False

    return True


def sample_key(sample: Dict[str, object]) -> str:
    prompt = json.dumps(sample["prompt"], ensure_ascii=False, sort_keys=True)
    return f"{sample['tag']}||{prompt}||{sample['chosen']}||{sample['rejected']}"


def collect_samples(
    target: int,
    builder,
    results: List[Dict[str, object]],
    seen: set,
) -> int:
    count = 0
    attempts = 0
    while count < target and attempts < target * 20:
        attempts += 1
        sample = builder()
        if not is_valid_sample(sample):
            continue
        key = sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        results.append(sample)
        count += 1
    return count


def main() -> None:
    random.seed(SEED)
    DST_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("🧠 dpo_data_logic_deep.py — DPO 3.4 深层逻辑策略修复")
    print("=" * 72)
    print(f"输出: {DST_FILE}")
    print(
        "配额: "
        f"事实问答={QUOTA_FACT}  人设不过展开={QUOTA_PERSONA}  "
        f"周五状态一致={QUOTA_WEEKDAY}  情绪优先={QUOTA_EMOTION}  "
        f"不空泛贴合={QUOTA_AFFINITY}"
    )
    print("=" * 72)

    results: List[Dict[str, object]] = []
    seen = set()

    counts = {
        "fact_answer_no_forced_counter_question": collect_samples(
            QUOTA_FACT, make_fact_sample, results, seen
        ),
        "persona_not_over_expand": collect_samples(
            QUOTA_PERSONA, make_persona_sample, results, seen
        ),
        "weekday_state_consistency": collect_samples(
            QUOTA_WEEKDAY, make_weekday_sample, results, seen
        ),
        "emotion_anchor_before_topic_expand": collect_samples(
            QUOTA_EMOTION, make_emotion_sample, results, seen
        ),
        "affinity_not_claimed_without_grounding": collect_samples(
            QUOTA_AFFINITY, make_affinity_sample, results, seen
        ),
    }

    with open(DST_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 72)
    print(f"📊 完成: {len(results)} 条")
    for tag, count in counts.items():
        print(f"   {tag}: {count}")
    print(f"   输出: {DST_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
