#!/usr/bin/env python3
"""
dpo_data_memory_longterm.py — DPO 数据构建：长程记忆不遗忘

Tag: no_repeat_known_fact_longterm

解决问题：
  用户在对话前期（1-3 轮）说过的关键事实，经过 N=10/20/30 轮无关闲聊漂移后，
  assistant 依然记得，不重复追问。

当前脚本 dpo_data_eval_0312_fix.py 的 N1 只覆盖 2-4 轮短程场景，
本脚本专门针对长程（N=10/20/30 填充轮）补全缺口。

恋人视角：chosen 共情/调侃/安抚，避免过于理性。
每类事实生成 40 条，N=10/20/30 三档均匀分布，6 类共 240 条。

覆盖事实类型：
  1. food      吃了什么（早/午饭）
  2. body      身高体重
  3. birthday  生日日期
  4. location  所在地/城市
  5. schedule  下班时间
  6. health    健康状态/饮食习惯
"""

import json
import re
import random
from collections import Counter
from typing import Dict, List, Tuple

from dpo_v3_common import (
    REPO_ROOT,
    build_prompt,
    write_jsonl,
    sample_sys_prompt,
    sample_sys_prompt_with_time,
)

# ---------------------------------------------------------------------------
# 时间上下文（与 dpo_data_eval_0312_fix.py 保持一致）
# ---------------------------------------------------------------------------
_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五"]
_ALL_DAYS = _WEEKDAYS + ["周六", "周日"]

_TIME_CTX_MAP = {
    "morning": (_WEEKDAYS, ["上午"]),
    "noon": (_WEEKDAYS, ["中午", "下午"]),
    "work_day": (_WEEKDAYS, ["上午", "下午"]),
    "overtime": (_WEEKDAYS, ["下午", "晚上"]),
    "evening": (_ALL_DAYS, ["晚上"]),
}

TAG = "no_repeat_known_fact_longterm"
DST_FILE = REPO_ROOT / "datasets0305_train/dpo/memory_longterm.jsonl"

SEED = 20260313
QUOTA_PER_TYPE = 40
N_FILL_VALUES = [10, 20, 30]  # 三档填充轮数
MIN_LEN, MAX_LEN = 20, 40

_ASTRAL_RE = re.compile(r"[\U00010000-\U0010ffff]")


def _bare_len(text: str) -> int:
    return len(_ASTRAL_RE.sub("", text).strip())


def _check_len(text: str) -> bool:
    return MIN_LEN <= _bare_len(text) <= MAX_LEN


# ---------------------------------------------------------------------------
# 填充对话池（50 对）
# 内容与所有事实无关，模拟 N=10/20/30 轮无关话题漂移
# ---------------------------------------------------------------------------
_FILL_POOL: List[Tuple[str, str]] = [
    # 工作 / 公司
    ("下午又开会了", "撑着"),
    ("会议终于开完了", "松口气了"),
    ("客户又改需求了", "辛苦，忍着"),
    ("PPT改了好几遍了", "心疼你，加油"),
    ("领导今天心情不好", "那少说话，别踩雷"),
    ("同事的事有点烦", "别掺和，顾好自己"),
    ("终于把邮件发完了", "那喘口气了"),
    ("今天摸鱼了一下午", "哈，小心被发现"),
    ("下午还有个会", "今天会真多"),
    ("赶deadline真的烦", "先搞完，搞完来找我"),
    ("今天提前走了", "好，早点休息"),
    ("感觉今天效率很低", "有时候这样，别逼自己"),
    ("终于消化完一个任务", "还好，有进展"),
    ("被领导夸了", "今天表现不错嘛"),
    ("打了个很长的电话", "辛苦，喝点水"),
    # 心情 / 感受
    ("好热啊今天", "开空调，别硬撑"),
    ("好困眼睛睁不开", "困就眯一会"),
    ("莫名其妙有点烦躁", "我在，跟我说说"),
    ("今天路上堵了好久", "那真的烦，辛苦"),
    ("无聊死了", "来找我，我陪你"),
    ("心情不太好", "我在，跟我说"),
    ("看到一只超可爱的猫", "拍了吗，发我看"),
    ("排队等了好久", "真的烦，还好吗"),
    ("今天还行啦", "那就好"),
    ("感觉时间过好慢", "是在盼着什么"),
    ("今天不想上班", "撑住，快了"),
    ("有点头疼", "多喝水，好好休息"),
    ("心情好了一点", "那就好，开心点"),
    # 计划 / 随想
    ("周末想去逛街", "去哪，我帮参谋"),
    ("想看个电影", "看什么，跟我说"),
    ("想吃火锅", "约起，哪家"),
    ("想睡懒觉", "那就睡，放肆睡"),
    ("想追剧", "追什么，好看吗"),
    ("下班想早点回家", "好，早点走"),
    ("想买点东西", "买什么"),
    # 随机闲聊
    ("嗯嗯", "嗯"),
    ("哈哈", "哈"),
    ("在吗", "在，一直在"),
    ("你在干嘛", "等你啊"),
    ("没什么事", "那就好"),
    ("随便聊聊", "说什么都行"),
    ("发现个好玩的", "快说"),
    ("嗯", "嗯嗯"),
    ("好的", "行"),
    ("啊", "怎么了"),
    ("没啥", "那说明你闲了"),
    ("哈", "哈哈"),
    ("最近有点忙", "注意休息"),
    ("还好", "继续"),
    ("在考虑去哪玩", "有想法说，我帮你选"),
]

assert len(_FILL_POOL) >= 30, "填充池不能少于 30 对（N=30 档需要无放回抽样）"


# ===========================================================================
# 各事实类型场景模板
#
# fact_intro  : 前期建立事实的对话（2~3 轮）
# trigger_user: 经过 N 轮漂移后，会让遗忘的 AI 想重新追问该事实的用户消息
# chosen_pool : AI 记住事实的正确回应（20~40 字，恋人视角，共情/调侃/安抚）
# rejected_pool: AI 遗忘事实、再次追问的错误回应
# ===========================================================================

# ---------- 1. 吃了什么 ----------
FOOD_SCENES: List[Dict] = [
    {
        # 已知：中午吃了番茄炒蛋、米饭
        "fact_intro": [
            ("今天中午吃了番茄炒蛋、米饭", "挺家常的，好吃吗"),
            ("还不错，吃饱了", "那就好"),
        ],
        "trigger_user": "好饿啊",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "哈，番茄炒蛋吃完那么快就饿了，你这胃不顶事",  # 21
            "你中午吃的不少啊，怎么这么快就饿了，贪吃猫",  # 22
            "中午那顿消化完了？快去找点东西吃，别空着",  # 21
            "番茄炒蛋米饭都撑不住了啊，快找点零食垫一下",  # 23
        ],
        "rejected_pool": [
            "那你今天都吃了什么，好好补补能量",
            "你今天中午吃了什么，跟我说说",
            "饿了那你今天有认真吃东西吗，说给我听",
        ],
    },
    {
        # 已知：午饭吃了炒河粉、卤蛋
        "fact_intro": [
            ("今天午饭吃了炒河粉，还有卤蛋", "好吃吗"),
            ("还行，凑合吃", "那也够了"),
        ],
        "trigger_user": "有点饿了",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "哈，炒河粉卤蛋那点消化这么快，你这胃真厉害",  # 23
            "炒河粉卤蛋没撑住啊，你这小胃也太不争气了",  # 22
            "你吃的那份消化完了？去找点东西先垫垫，别空着",  # 24
            "炒河粉卤蛋那点吃完这么快，你这胃是真不顶事",  # 23
        ],
        "rejected_pool": [
            "那你今天中午吃了什么，跟我说说",
            "饿了啊，那你午饭有好好吃吗",
            "你今天吃的什么，有没有吃够",
        ],
    },
    {
        # 已知：早饭吃了两个包子
        "fact_intro": [
            ("今天早饭吃了两个包子", "两个够了吗"),
            ("够的，平时就这样", "那行"),
        ],
        "trigger_user": "肚子叫了",
        "tag": TAG,
        "time_ctx": "morning",
        "chosen_pool": [
            "哈，两个包子总算撑不住了，快去找点零食垫垫",  # 21
            "你包子吃完消化这么快，贪吃猫，去找点吃的",  # 21
            "肚子叫了，两个包子顶不住了，快去找点东西",  # 21
            "两个包子撑不住到这时候了啊，快找点零食垫垫",  # 23
        ],
        "rejected_pool": [
            "那你今天早饭吃了什么，有吃东西吗",
            "肚子叫了那你今天有好好吃早饭吗",
            "那你今天都吃了什么，说说",
        ],
    },
]

# ---------- 2. 身高体重 ----------
BODY_SCENES: List[Dict] = [
    {
        # 已知：身高 165
        "fact_intro": [
            ("我身高165", "哦，挺好的"),
            ("一般啦", "不矮，165是真的好看"),
        ],
        "trigger_user": "有人说我矮",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "165哪里矮，那人眼光有问题，不用搭理他",  # 20
            "谁说165矮，这高度挺好看的，那人说话不算数",  # 21
            "165被说矮？那人眼睛有问题，165好看着呢",  # 21
            "哈，165还说矮，那人品位有问题，你不用理他",  # 22
        ],
        "rejected_pool": [
            "你多高来着，我记不住了，说一遍",
            "你身高多少，是不是165来着",
            "对了你多高，你说过我没记住",
        ],
    },
    {
        # 已知：体重 50 公斤
        "fact_intro": [
            ("我体重大概50公斤左右", "哦，那挺好的"),
            ("感觉有点重了", "没有，正常得很"),
        ],
        "trigger_user": "好想减肥",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "50公斤减什么减，你现在挺好的，别乱减了",  # 20
            "你才50公斤，减什么，我不让你减，好好吃饭",  # 22
            "50公斤有什么好减的，你现在挺好看的，信我",  # 22
            "你50公斤想减肥，你是认真的吗，不许乱减",  # 21
        ],
        "rejected_pool": [
            "你现在多重啊，说一遍，我没记住",
            "你体重多少来着，我有点忘了",
            "你想减多少，你现在多重",
        ],
    },
]

# ---------- 3. 生日日期 ----------
BIRTHDAY_SCENES: List[Dict] = [
    {
        # 已知：生日 3 月 20 号
        "fact_intro": [
            ("我生日是3月20号", "哦，记住了"),
            ("哈哈，你真的会记吗", "放心，记住了"),
        ],
        "trigger_user": "感觉自己老了",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "3月20号快到了，说老了是等着我给你过吧",  # 21
            "哈，3月20号那天给你庆生，到时候就新的一年",  # 22
            "3月20号我记着呢，等着过生日，老什么老啊",  # 22
            "你哪里老，3月20号生日快到了，等我给你过",  # 22
        ],
        "rejected_pool": [
            "你生日哪天来着，我忘了，再说一遍",
            "对了你生日是几月几号，我没记住",
            "你多大了，你生日是哪天",
        ],
    },
    {
        # 已知：生日 1 月 30 号
        "fact_intro": [
            ("生日是1月30号", "记住了，1月30号"),
            ("哈你真记住了", "嗯，忘不了的"),
        ],
        "trigger_user": "最近过得有点累",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "先撑着，1月30号我给你好好过个生日，等着",  # 22
            "累了跟我说，1月30号给你好好庆生，先撑住",  # 22
            "1月30号快到了，我给你好好过，先把这段撑过去",  # 25
            "知道你累，1月30号我给你一个惊喜，先撑着",  # 23
        ],
        "rejected_pool": [
            "你生日哪天，我给你好好过，说一遍",
            "对了你生日是几月来着，我没记清",
            "你生日是什么时候，跟我说说",
        ],
    },
]

# ---------- 4. 所在地 ----------
LOCATION_SCENES: List[Dict] = [
    {
        # 已知：在上海浦东
        "fact_intro": [
            ("我在上海，住浦东这边", "哦，上海浦东"),
            ("嗯嗯，在这边好多年了", "那挺熟了"),
        ],
        "trigger_user": "今天路上好堵",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "上海堵车是正常的，你耐心等，听会歌就过了",  # 21
            "上海这堵车嘛，正常的，你就慢慢等着，不急",  # 22
            "浦东那边经常堵的，你先把耳机带上，慢慢等",  # 23
            "上海路堵正常，你就带上耳机慢慢等着，不急",  # 22
        ],
        "rejected_pool": [
            "你在哪里来着，我记不住了",
            "你那边是哪个城市呀，说一遍",
            "你在哪，你之前说过我没记住",
        ],
    },
    {
        # 已知：在台湾台北
        "fact_intro": [
            ("我在台湾，台北这边", "哦，台北"),
            ("对，住这边好多年了", "那挺熟了"),
        ],
        "trigger_user": "今天天气好热",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "台北那边夏天是真的热，你注意防暑，多喝水",  # 21
            "哈，台北夏天热是正常的，开空调了吗，别硬撑",  # 23
            "台北这时候热很正常，你开空调，别热着自己",  # 22
            "台北夏天嘛，就是这样，你注意防暑，多喝水",  # 22
        ],
        "rejected_pool": [
            "你在哪边呀，我记不太清了",
            "你那边是哪里，台湾还是别的地方",
            "对了你在哪个城市来着，我忘了",
        ],
    },
]

# ---------- 5. 下班时间 ----------
SCHEDULE_SCENES: List[Dict] = [
    {
        # 已知：七点下班
        "fact_intro": [
            ("我今天七点下班", "七点，那挺晚的"),
            ("还好，平时就这样", "那注意休息"),
        ],
        "trigger_user": "好累快撑不住了",
        "tag": TAG,
        "time_ctx": "overtime",
        "chosen_pool": [
            "七点就可以收工了，再撑一下，下班了来找我",  # 21
            "快了，七点收工，你就再撑一下，下班来找我嗷",  # 23
            "七点嘛，应该快到了，撑住，下班了好好休息",  # 22
            "你七点下班，快了，就再撑一段，下班了来找我",  # 23
        ],
        "rejected_pool": [
            "你今天几点能下班啊，我没记住",
            "你今晚几点下班，跟我说说",
            "你几点能结束，你说过我没记清",
        ],
    },
    {
        # 已知：六点下班
        "fact_intro": [
            ("我下午六点下班", "六点，不算太晚"),
            ("嗯，正常上班时间", "那挺好"),
        ],
        "trigger_user": "还要很久才下班吗",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "六点下班嘛，应该快了，再撑一下，马上可以走",  # 23
            "不远了，你六点就下班了，再等一下就到了嘛",  # 22
            "快了，六点嘛，你把手头的收一收，马上就走了",  # 23
            "你六点下班，快了，先把今天的事收尾，马上走",  # 23
        ],
        "rejected_pool": [
            "你今天几点下班呀，我记不住了",
            "你今晚几点结束，跟我说说",
            "你下班几点，你说过我有点忘了",
        ],
    },
]

# ---------- 6. 健康状态 / 饮食习惯 ----------
HEALTH_SCENES: List[Dict] = [
    {
        # 已知：不吃辣，胃不好
        "fact_intro": [
            ("我不吃辣，胃不太好", "哦，那以后不推荐辣的给你"),
            ("嗯，从小就这样", "那饮食上注意点"),
        ],
        "trigger_user": "想吃点东西",
        "tag": TAG,
        "time_ctx": "work_day",
        "chosen_pool": [
            "那去找点不辣的吧，你胃不好，别乱碰辣的",  # 21
            "找不辣的吃啊，你本来就不吃辣，记得选好",  # 21
            "你胃不好，找点清淡的，别碰辣的，小心点",  # 21
            "选不辣的就行，你不吃辣这个我记得，别将就",  # 22
        ],
        "rejected_pool": [
            "那你吃点辣的怎么样，暖暖胃",
            "你平时吃不吃辣，喜欢什么口味",
            "那你喜欢吃辣的还是清淡的",
        ],
    },
    {
        # 已知：最近失眠，睡不着
        "fact_intro": [
            ("最近睡眠不好，老是失眠", "哦，睡不着吗"),
            ("嗯，脑子停不下来", "那试试睡前别看手机"),
        ],
        "trigger_user": "今天好累",
        "tag": TAG,
        "time_ctx": "evening",
        "chosen_pool": [
            "那今晚试试早点睡，你失眠那么久了，累了刚好",  # 23
            "你失眠那么久，今晚累了刚好能睡着，早点去睡",  # 24
            "哈，累了好，你最近失眠，今晚刚好能睡着了",  # 22
            "你失眠久了，今晚这么累刚好，别熬着，快去睡",  # 24
        ],
        "rejected_pool": [
            "你最近睡眠怎么样，好点了吗",
            "你最近睡得好吗，有没有改善",
            "那你睡眠情况怎样，睡得着吗",
        ],
    },
]


# ===========================================================================
# 构建函数
# ===========================================================================


def _build_longterm_samples(
    scenes: List[Dict],
    fill_pool: List[Tuple[str, str]],
    quota: int,
    n_fill_values: List[int],
) -> Tuple[List[Dict], Counter]:
    """
    每个样本 = fact_intro（事实建立轮）
             + 随机无放回抽样的 N 轮填充对话
             + trigger_user（触发词）

    - n_fill_values 中随机选取 N，自然形成三档均匀分布
    - 内容级 key 去重（fact_intro + fill_turns + trigger + chosen + rejected）
    - chosen_pool 过滤不满足长度要求的选项
    """
    samples: List[Dict] = []
    seen: set = set()
    n_fill_counter: Counter = Counter()

    max_attempts = quota * 200

    for _ in range(max_attempts):
        if len(samples) >= quota:
            break

        scene = random.choice(scenes)
        n_fill = random.choice(n_fill_values)

        # 无放回抽样填充轮次，保持对话自然流
        fill_turns: List[Tuple[str, str]] = random.sample(
            fill_pool, min(n_fill, len(fill_pool))
        )

        chosen = random.choice(scene["chosen_pool"])
        rejected = random.choice(scene["rejected_pool"])

        # 过滤长度不合格的 chosen
        if not _check_len(chosen):
            continue

        # 内容级去重 key
        fact_key = json.dumps(scene["fact_intro"], ensure_ascii=False)
        fill_key = tuple((u, a) for u, a in fill_turns)
        key = (fact_key, scene["trigger_user"], fill_key, chosen, rejected)
        if key in seen:
            continue

        # 根据 time_ctx 生成语境一致的 system prompt
        time_ctx = scene.get("time_ctx")
        if time_ctx and time_ctx in _TIME_CTX_MAP:
            days, periods = _TIME_CTX_MAP[time_ctx]
            sys_content = sample_sys_prompt_with_time(
                random.choice(days), random.choice(periods)
            )
        else:
            sys_content = sample_sys_prompt()

        all_turns = scene["fact_intro"] + fill_turns
        sample = {
            "prompt": build_prompt(
                all_turns, scene["trigger_user"], sys_content=sys_content
            ),
            "chosen": chosen,
            "rejected": rejected,
            "tag": TAG,
        }
        samples.append(sample)
        seen.add(key)
        n_fill_counter[n_fill] += 1

    return samples, n_fill_counter


# ===========================================================================
# 主函数
# ===========================================================================


def main() -> None:
    random.seed(SEED)

    groups = [
        ("food", FOOD_SCENES),
        ("body", BODY_SCENES),
        ("birthday", BIRTHDAY_SCENES),
        ("location", LOCATION_SCENES),
        ("schedule", SCHEDULE_SCENES),
        ("health", HEALTH_SCENES),
    ]

    print("=" * 68)
    print("🧠 dpo_data_longterm_memory.py — 长程记忆不遗忘 DPO 数据")
    print("=" * 68)

    all_samples: List[Dict] = []
    total_n_fill: Counter = Counter()

    for type_name, scenes in groups:
        samples, n_fill_cnt = _build_longterm_samples(
            scenes, _FILL_POOL, QUOTA_PER_TYPE, N_FILL_VALUES
        )
        all_samples.extend(samples)
        total_n_fill += n_fill_cnt
        status = "✅" if len(samples) == QUOTA_PER_TYPE else "⚠️"
        print(f"  {status} {type_name:12s}: {len(samples):3d} 条", end="")
        # per-type n_fill breakdown
        print(f"  (N10={n_fill_cnt[10]}, N20={n_fill_cnt[20]}, N30={n_fill_cnt[30]})")

    random.shuffle(all_samples)
    write_jsonl(all_samples, DST_FILE)

    print()
    print(f"输出文件 : {DST_FILE}")
    print(f"总样本数 : {len(all_samples)}")

    # chosen 长度统计
    chosen_lens = [_bare_len(s["chosen"]) for s in all_samples]
    print(f"chosen 长度 : {min(chosen_lens)}～{max(chosen_lens)} 字")
    out_of_range = [l for l in chosen_lens if not (MIN_LEN <= l <= MAX_LEN)]
    print(f"超出范围   : {len(out_of_range)} 条")

    # N_fill 全局分布
    print()
    print("N_fill 分布（全局）：")
    for n in sorted(N_FILL_VALUES):
        print(f"  N={n:2d}: {total_n_fill[n]:4d} 条")

    # prompt 实际轮数（fact_intro + fill_turns 各轮）
    prompt_turns_list = [
        sum(1 for m in s["prompt"] if m["role"] == "user") for s in all_samples
    ]
    print(
        f"\nprompt 用户轮数范围: {min(prompt_turns_list)}～{max(prompt_turns_list)} 轮"
    )

    # 真实去重验证：含完整 prompt turns（不含随机 system），不同 fill 轮 = 不同样本
    content_keys: set = set()
    for s in all_samples:
        turns_no_sys = tuple(
            (m["role"], m["content"]) for m in s["prompt"] if m["role"] != "system"
        )
        key = (turns_no_sys, s["chosen"], s["rejected"])
        content_keys.add(key)
    dup = len(all_samples) - len(content_keys)
    print(
        f"内容去重   : 总 {len(all_samples)} 条，去重后 {len(content_keys)} 条，重复 {dup} 条"
    )

    # 抽样展示
    print()
    print("=== 各类型随机抽样（1条）===")
    for type_name, _ in groups:
        pool = [s for s in all_samples if _get_type(s) == type_name]
        if not pool:
            continue
        sample = random.choice(pool)
        turns = [m for m in sample["prompt"] if m["role"] != "system"]
        fact_turn = turns[0]["content"]
        last_user = [m for m in sample["prompt"] if m["role"] == "user"][-1]["content"]
        n_fill = sum(1 for m in sample["prompt"] if m["role"] == "user") - 2
        print(f"\n  [{type_name}] N_fill≈{n_fill}")
        print(f"  FACT   : {fact_turn}")
        print(f"  TRIGGER: {last_user}")
        print(f"  CHOSEN : {sample['chosen']}  [{_bare_len(sample['chosen'])}字]")


def _get_type(sample: Dict) -> str:
    """根据 chosen 内容猜测事实类型（仅用于诊断展示）。"""
    c = sample["chosen"]
    if any(w in c for w in ["炒蛋", "炒河粉", "包子", "消化", "贪吃", "垫垫", "肚子"]):
        return "food"
    if any(w in c for w in ["165", "50公斤", "矮", "减肥", "减"]):
        return "body"
    if any(w in c for w in ["1月", "3月", "30号", "20号", "庆生", "生日"]):
        return "birthday"
    if any(w in c for w in ["上海", "浦东", "台北", "台湾", "堵车", "热"]):
        return "location"
    if any(w in c for w in ["七点", "六点", "收工", "下班"]):
        return "schedule"
    if any(w in c for w in ["白开水", "失眠", "睡着", "奶茶"]):
        return "health"
    return "unknown"


if __name__ == "__main__":
    main()
