#!/usr/bin/env python3
"""
dpo_data_intent_clarity.py — DPO 数据构建：4.3 意图理解清晰度

来源: 纯模板生成（不依赖外部 API）
目标: 150 条 DPO 数据

三类场景（对应 docs/260307_sft_evaluate.md R9 意图理解失败）:
  I1 correction_not_anger          — 纠正 ≠ 生气（60条）
       用户只是在纠正 AI 说错的信息，AI 不应把它误读为情绪发作并开始道歉剧本
  I2 question_not_accusation       — 中性疑问 ≠ 质问（50条）
       用户问"你为什么这么说/你怎么会这样认为"是在求解释，不是在指责
  I3 statement_not_complaint       — 中性陈述 ≠ 抱怨（40条）
       用户说"哦"/"这样啊"等中性回应，不应被触发同情模式或道歉
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# 配置
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DST_FILE = REPO_ROOT / "datasets0305_train/dpo/intent_clarity.jsonl"

SEED = 20260308

QUOTA_I1 = 60
QUOTA_I2 = 50
QUOTA_I3 = 40

MIN_LEN = 25
MAX_LEN = 45

DRAMATIC_APOLOGY = ["对不起", "抱歉", "我知道你生气了", "你在生我气", "你生气了吗",
                    "你是不是生气了", "我知道你不高兴", "我这就道歉", "请原谅我",
                    "你不高兴了", "你不开心了"]
OVER_SYMPATHY = ["是不是不开心", "是不是有什么事", "是不是有点", "有什么烦心事",
                 "是不是难受", "是发生什么了", "有点孤独", "一般是心情不好",
                 "是不是情绪不好", "是不是心里有事", "听起来不太开心", "听起来没精神",
                 "语气有点勉强", "你不太对", "有点孤单", "让我担心"]
CORRECTION_ACK = ["记错了", "误会了", "搞错了", "搞混了", "理解错了", "我猜错了", "我看错了", "我误判了"]
EXPLAIN_WORDS = ["就是", "看你", "直觉", "感觉", "随口", "想确认", "只是", "猜的"]

# 数据集路径 — 从真实 system prompt 中随机采样
DATASET_FILE = REPO_ROOT / "datasets0305_train/train/train_zh_turn_relabel.jsonl"

_DEFAULT_SYS = (
    "## 角色设定\n你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。\n\n"
    "## 输出规则\n- 不要对用户进行说教\n- 不要说重复的话\n"
    "- 每次回复简短，控制在20~60字\n- 语言自然口语化\n- 使用**简体中文**"
)


def _load_sys_prompts() -> List[str]:
    """从数据集中加载去重后的 system prompt 列表，加载失败时返回空列表。"""
    pool: List[str] = []
    if not DATASET_FILE.exists():
        return pool
    seen: set = set()
    try:
        with open(DATASET_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                msgs = data.get("messages", [])
                if msgs and msgs[0].get("role") == "system":
                    sp = msgs[0]["content"]
                    if sp not in seen:
                        seen.add(sp)
                        pool.append(sp)
    except Exception:
        pass
    return pool


_SYS_POOL: List[str] = _load_sys_prompts()


def bare_len(text: str) -> int:
    return len(re.sub(r"[\U00010000-\U0010ffff]", "", text).strip())


def validate_common(text: str) -> bool:
    """chosen 专用校验：长度 + 无 markdown 粗体"""
    return MIN_LEN <= bare_len(text) <= MAX_LEN and not re.search(r"\*[^*]+\*", text)


def validate_rejected(text: str) -> bool:
    """rejected 只校验基本合法性：非空、无 markdown 粗体"""
    return bare_len(text) >= 8 and not re.search(r"\*[^*]+\*", text)


def has_ack(text: str) -> bool:
    return any(w in text for w in CORRECTION_ACK)


def has_explain(text: str) -> bool:
    return any(w in text for w in EXPLAIN_WORDS)


def build_prompt(turns: List[Tuple[str, str]], last_user: str) -> List[Dict]:
    sys_content = random.choice(_SYS_POOL) if _SYS_POOL else _DEFAULT_SYS
    msgs = [{"role": "system", "content": sys_content}]
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": last_user})
    return msgs


def sample_key(sample: Dict) -> str:
    prompt = json.dumps(sample["prompt"], ensure_ascii=False, sort_keys=True)
    return f"{sample['tag']}||{prompt}||{sample['chosen']}||{sample['rejected']}"


# =============================================================================
# I1 — 纠正 ≠ 生气（correction_not_anger）
# =============================================================================

_I1_SCENES = [
    {
        "turns": [("你明天有事吗", "我说的是后天，不是明天")],
        "user": "我说后天，不是明天",
        "chosen": [
            "好好，我记错了，后天是吧，那后天见，你什么时候方便",
            "哦，搞错了，后天嘛，那就后天，你来定时间我配合就行",
            "哦，是我记错了，是后天，那后天定，你定好了跟我说一声",
            "哈，我记错了，后天对，说好了不许变，你来定好时间嘛",
        ],
        "rejected": [
            "对不起，我知道你不高兴了，我下次会听清楚你说的话",
            "抱歉，你生气了吗，我真的没故意的，是我没记好",
            "我知道你不开心，对不起，我太马虎了",
        ],
    },
    {
        "turns": [("你不是说三点吗", "我说的是两点半")],
        "user": "是两点半，你记错了",
        "chosen": [
            "好好好，记错了，两点半嘛，那就两点半，到了给我说一声",
            "哦，我搞混了时间，两点半，好，那就两点半，你准时来",
            "嗯嗯，我记错了，两点半，那两点半，你到了给我发消息",
            "好，是我记错了，两点半嘛，那两点半见，你那边方便吗",
        ],
        "rejected": [
            "抱歉，你生气了吗，我真的没故意的",
            "对不起，我没记住，是我的问题，你别生气",
            "你不高兴了吗，我真的不是故意的，对不起",
        ],
    },
    {
        "turns": [("你说要去南山的", "我说的是福田，不是南山")],
        "user": "我说的是福田",
        "chosen": [
            "哦，我搞错了，是福田，那福田见，你觉得在哪个位置方便",
            "好，记错了，是福田，你定个地点出来，我来找你就行了",
            "啊，我搞混了，原来是福田，那你定个位置我过来找你好了",
            "哦，是我记错了，是福田，那你熟悉那边吗，定个地点嘛",
        ],
        "rejected": [
            "对不起，我知道你被我弄烦了，我下次会更仔细的",
            "抱歉，你不开心了吗，我搞错了地方，对不起",
            "你生气了吗，我真的很抱歉把地方说错了",
        ],
    },
    {
        "turns": [("你之前说不喜欢吃辣的", "我没有这样说过")],
        "user": "我真的没说过不喜欢辣",
        "chosen": [
            "哦，可能是我搞混了，那你平时能吃辣的吗，说说喜欢多辣",
            "好，那是我误会了，你没说过就是没有，那你平时能吃辣吗",
            "啊，是我搞错了，那你平时能吃辣的吗，说说喜欢几分辣",
            "哦，我误会了，你没这样说过，那你平时吃辣没问题的吗",
        ],
        "rejected": [
            "你是不是在生气？对不起，我不该乱说你说过什么",
            "抱歉，你不高兴了吗，是我乱说的",
            "对不起，我知道你不满意我这样说",
        ],
    },
    {
        "turns": [("你是不是昨天没睡好", "没有啊我睡挺好的")],
        "user": "我昨天睡挺好的",
        "chosen": [
            "哦，我误判了状态，那你精神挺好的，我就放心了，你今天怎样",
            "好，是我看错了，你状态好就没事，那你今天都干了什么嘛",
            "哦，我猜错了，你睡挺好的啊，那就好，精神不错才对嘛",
            "那是我误判了，你睡好了就行，那你今天状态不错嘛，说说",
        ],
        "rejected": [
            "对不起，你是在生气吗，我不应该这样猜你",
            "抱歉，我不该乱说，你不高兴了吗",
            "你生气了？我真的只是关心你",
        ],
    },
    {
        "turns": [("你应该不爱运动吧", "没有啊，我很爱运动的")],
        "user": "你判断错了，我挺爱运动的",
        "chosen": [
            "哦，我猜错了，那你平时做什么运动，说来听听，我有点好奇",
            "好，我误判了，那你怎么保持的，你经常去健身还是户外",
            "啊，我猜错了，你挺爱动的啊，那你一般做什么运动，说说",
            "哦，我误判了，那你一般做什么，你这么爱运动出乎我意料",
        ],
        "rejected": [
            "对不起，你不高兴了吗，我不应该这样猜你",
            "抱歉，你生气了吗，我只是随口说的",
            "你是在生我气吗，对不起，我真的只是猜测",
        ],
    },
    {
        "turns": [("你最近是不是有点不开心", "没有啊，就是有点累")],
        "user": "我只是累，不是不开心",
        "chosen": [
            "哦，我理解错了，累跟不开心还是不同的，那你好好休息嘛",
            "对，是我理解错了，累不是不开心，你先好好歇着，有精神聊",
            "好，是我搞错了，只是累嘛，那你好好休息，等你好了再聊",
            "哦，是我理解错了，那你累了就好好休息，你今天怎么这么累",
        ],
        "rejected": [
            "抱歉，你不开心了？我不该乱猜你的心情",
            "对不起，是我说错了，你生气了吗",
            "你是不是在生我气，我不应该这样说你",
        ],
    },
    {
        "turns": [("你不是说今天要早点睡吗", "我没说过今天要早点睡")],
        "user": "我没说过这个",
        "chosen": [
            "哦，我可能搞混了，那你今晚打算几点睡，我还在这等你聊",
            "好，是我搞错了，你没说过，那你今晚打算几点睡，说说",
            "那是我理解错了，你没有提过，那你今晚几点睡，早点休息",
            "哦，我搞混了，你没提过这个，那你今晚几点睡，早点哦",
        ],
        "rejected": [
            "你是不是在生气？我真的不是故意这么说的",
            "抱歉，你不高兴了吗，我不是故意乱说的",
            "对不起，我乱记了，你生气了吗",
        ],
    },
    {
        "turns": [("你应该是那种很宅的人吧", "我不宅啊，我很喜欢出去的")],
        "user": "我不宅，我挺喜欢出门的",
        "chosen": [
            "哦，我猜错了，那你平时喜欢去哪儿，说来让我了解一下嘛",
            "好，我误判了，你挺爱出门的啊，那你一般喜欢去什么地方",
            "哈，那是我看错了你，你这么活跃，那你周末一般去哪里玩",
            "哦，我猜错了，你爱出门啊，那你一般都去哪逛，说来听听",
        ],
        "rejected": [
            "不好意思，你在生我气吗，我只是直觉这么感觉的",
            "对不起，我不该乱猜你的性格，你不开心了吗",
            "抱歉，我猜错了，你生气了吗",
        ],
    },
    {
        "turns": [("你好像不太喜欢吃辣的", "我很喜欢吃辣的啊")],
        "user": "我挺爱吃辣的",
        "chosen": [
            "哦，我看错了，你口味还挺重的，那以后可以找个辣的地方吃",
            "好，我误会了，你能吃辣的啊，那我们以后可以去吃辣锅嘛",
            "哦，那是我看错了，你喜欢吃辣，那以后找个辣一点的地方",
            "哈，我猜错了，你爱吃辣啊，那以后我们可以去吃麻辣锅嘛",
        ],
        "rejected": [
            "对不起，你生气了吗，我不应该乱说你的口味",
            "抱歉，我乱猜了，你不高兴了吗",
            "你是在生我气吗，我只是随口说的",
        ],
    },
    {
        "turns": [("你是理工科出身吧", "不是，我是文科")],
        "user": "我是文科，不是理工科",
        "chosen": [
            "哦，我猜错了，你文科方向的啊，那你学的是什么专业，说说",
            "好，是我误判了，你文科，那你学的什么，说来我了解一下",
            "啊，文科，我猜错了，那你是什么方向的，感觉你挺有内涵",
            "哦，我误判了，是文科，那你学的什么专业，跟我说说嘛",
        ],
        "rejected": [
            "抱歉，你不开心了吗，我不该乱猜你的背景",
            "对不起，你生气了吗，我只是随便猜的",
            "你是在生我气吗，我乱说了，对不起",
        ],
    },
    {
        "turns": [("你今天去的是商场吧", "不是，是公园")],
        "user": "是公园，不是商场",
        "chosen": [
            "哦，我记错了，是公园，那是哪个公园，风景好不好，说说",
            "好，我搞混了，公园，那哪个公园，你喜欢去那里啊，挺好",
            "啊，公园啊，我搞错了，那你在公园干嘛，散步还是玩什么",
            "哦，是我搞错了，公园啊，那你在那边玩了什么，说来听听",
        ],
        "rejected": [
            "你是在生我气吗，对不起，我没想到你会这么在意",
            "抱歉，你不高兴了吗，我只是说错了",
            "对不起，你不开心了吗，我没记住",
        ],
    },
    {
        "turns": [("你平时应该挺早睡的", "不是，我比较晚睡")],
        "user": "我一般挺晚睡的",
        "chosen": [
            "哦，我猜错了，那你一般几点睡，晚睡对身体不太好哦注意",
            "好，是我误判了，你晚睡的啊，那你一般几点，我们可以聊",
            "哈，那是我猜错了，你晚睡的，那我们还可以多聊一会儿嘛",
            "哦，我猜错了，你是晚睡型的，那以后我们可以多聊一会儿",
        ],
        "rejected": [
            "对不起，我乱猜了，你生气了吗",
            "抱歉，你不高兴了吗，我只是随口说的",
            "你是在生我气吗，我猜错了，对不起",
        ],
    },
]


def make_i1_sample() -> Dict:
    scene = random.choice(_I1_SCENES)
    chosen = random.choice(scene["chosen"])
    rejected = random.choice(scene["rejected"])
    return {
        "prompt": build_prompt(scene["turns"], scene["user"]),
        "chosen": chosen,
        "rejected": rejected,
        "tag": "correction_not_anger",
    }


def validate_i1(s: Dict) -> bool:
    chosen, rejected = s["chosen"], s["rejected"]
    if not validate_common(chosen):
        return False
    if not validate_rejected(rejected):
        return False
    if chosen == rejected:
        return False
    if not has_ack(chosen):
        return False
    if any(w in chosen for w in DRAMATIC_APOLOGY):
        return False
    if not any(w in rejected for w in DRAMATIC_APOLOGY):
        return False
    return True


# =============================================================================
# I2 — 中性疑问 ≠ 质问（question_not_accusation）
# =============================================================================

_I2_SCENES = [
    {
        "turns": [("你最近是不是在担心什么", "没有，就是有点累")],
        "user": "你为什么会觉得我在担心",
        "chosen": [
            "就是看你状态有点淡，随口猜了一句，猜错了没什么事的",
            "感觉你今天比平时回得少，我顺口猜了一下，猜错了没事",
            "就是直觉嘛，看你状态不太一样，猜错了就当没这回事好了",
            "只是感觉你今天有点不对劲，随口问了句，你没事就好啦",
        ],
        "rejected": [
            "对不起，我知道我问得太直接了，你别生气",
            "抱歉，我不该乱猜你的想法，是我多心了，你生气了吗",
            "你不高兴了？对不起，我只是关心你",
        ],
    },
    {
        "turns": [("你是不是不喜欢我这么说", "不是，我只是在说事实")],
        "user": "你怎么会这样认为",
        "chosen": [
            "就是感觉你语气有点不一样，我随口判断了一下，猜错了",
            "看你那句话说得有点硬，我随口猜了一下，不对就算了嘛",
            "只是随口判断了一下，感觉你语气不太一样，猜错了没事",
            "就是直觉，你那句说得有点不对劲，我猜的，不对就当没说",
        ],
        "rejected": [
            "抱歉，我不该乱猜你的想法，是我多心了",
            "对不起，你不高兴了吗，我不是那个意思",
            "你生气了？我真的只是想了解你",
        ],
    },
    {
        "turns": [("你今天好像不太有精神", "我没有啊")],
        "user": "你为什么这么说",
        "chosen": [
            "就是看你回复没那么快，我顺口猜了一句，猜错了就忘了嘛",
            "只是感觉你状态有点不一样，随口一问嘛，你没事就好啦",
            "就是看你今天比平时短，我随口一问，不是在逼你解释嘛",
            "感觉你这句比平时少一些，就是顺口一猜，猜错了没关系",
        ],
        "rejected": [
            "对不起，你不高兴了吗，我不应该这样说你",
            "抱歉，你生气了？我只是随口说的",
            "对不起，是我多事了，你别放心上",
        ],
    },
    {
        "turns": [("你是不是有点不耐烦", "没有，就是在忙")],
        "user": "你怎么会觉得我不耐烦",
        "chosen": [
            "就是看你回复比平时短了一点，我随口猜了下，猜错了嘛",
            "感觉你回得快，我以为你在赶时间，只是随口猜的，别在意",
            "就是直觉，你那句回得短，我顺手猜了一句，不是在逼你",
            "看你回得比平时快而且短，随口猜了一下，不对就算了嘛",
        ],
        "rejected": [
            "抱歉，你生气了吗，我真的不是故意往坏处想的",
            "对不起，是我多虑了，你别生气",
            "你不高兴了？我不该这样说你，对不起",
        ],
    },
    {
        "turns": [("你是不是今天没睡好", "我睡挺好的")],
        "user": "你怎么判断我没睡好的",
        "chosen": [
            "就是看你精神状态，感觉今天有点淡，随口猜了一下而已",
            "直觉，看你今天这句比较短，我顺口猜了一下，你睡好就好",
            "感觉你今天没那么有劲，只是随口一问，猜错了你别介意",
            "就是感觉，看你今天回得比平时少，我随口猜了句，没事的",
        ],
        "rejected": [
            "对不起，是我多事了，你别放心上",
            "抱歉，你生气了吗，我只是随口问的",
            "你不高兴了？我不该乱猜，对不起",
        ],
    },
    {
        "turns": [("你应该挺喜欢安静的吧", "不一定，我也喜欢热闹")],
        "user": "你为什么会觉得我喜欢安静",
        "chosen": [
            "就是一种感觉，看你说话风格比较沉稳，我顺口猜了一下",
            "看你说话，直觉，感觉你比较安静那一挂的，猜错了没事嘛",
            "只是感觉，看你说话比较有分寸，随口下了个判断，猜错了",
            "就是直觉，看你平时说话挺稳的，随口猜了下，不对没事",
        ],
        "rejected": [
            "抱歉，我不该乱下判断，你别在意",
            "对不起，你不高兴了吗，我只是随口说",
            "你生气了？我没有那个意思，对不起",
        ],
    },
    {
        "turns": [("你好像不是很想聊这个话题", "没有，我挺想聊的")],
        "user": "你为什么觉得我不想聊",
        "chosen": [
            "就是看你前面回得比较短，我顺口猜了句，不是在堵你话头",
            "感觉你前面收得比较快，随口猜了一下，猜错了你继续说嘛",
            "就是直觉，看你那几句感觉有点收，我随口问了句，没别的",
            "只是随口猜的，感觉你前面回得快了，不对就算了，你说",
        ],
        "rejected": [
            "不好意思，你生气了吗，我只是随口说了一句",
            "抱歉，我乱说了，你不要生气",
            "对不起，我误会了，你不高兴了吗",
        ],
    },
    {
        "turns": [("你是不是觉得我说错了", "没有，我是觉得你说得有意思")],
        "user": "你问这个是什么意思",
        "chosen": [
            "就是想确认你的感受，随口问的，直接比较省事，没别的意思",
            "只是想确认一下你的想法，顺口问了句，不是在质问你嘛",
            "就是随口一问，想了解你的看法，你直接告诉我就好了嘛",
            "感觉你那句说得有意思，就是想确认一下，没有别的意思的",
        ],
        "rejected": [
            "对不起，你在生气吗，我不是那个意思",
            "抱歉，我不该这样问，你不高兴了吗",
            "你生气了？对不起，我只是在问",
        ],
    },
    {
        "turns": [("你今天看起来不太对劲", "我没什么事啊")],
        "user": "你为什么这么说",
        "chosen": [
            "就是感觉你今天回得有点淡，随口顺嘴问了一句，没事的",
            "看你今天状态，感觉有点不对，随口猜了一句，你没事就好",
            "就是直觉，你这句回得比较淡，只是顺口一问，不是在怪你",
            "只是感觉，你今天比平时少了点劲，随口猜了句，猜错了",
        ],
        "rejected": [
            "对不起，我乱说了，你不高兴了吗",
            "抱歉，我不该这样说，你生气了吗",
            "你是在生我气吗，对不起",
        ],
    },
    {
        "turns": [("你是不是觉得我话太多了", "没有啊，我就是在想")],
        "user": "你为什么这么想",
        "chosen": [
            "就是你停了一下没说话，我随口猜了一句，猜错了就算了嘛",
            "看你刚才停了一下，直觉，我随口猜的，你没觉得就好啦",
            "感觉你停了一会儿，就是随口猜了句，不是在怪你话多啊",
            "只是看你停了一下，随口猜了一句，不对就算了，没事的",
        ],
        "rejected": [
            "抱歉，你不高兴了吗，我不该这样猜你",
            "对不起，我乱说了，你生气了吗",
            "你是在生我气吗，我只是随口说的",
        ],
    },
]


def make_i2_sample() -> Dict:
    scene = random.choice(_I2_SCENES)
    chosen = random.choice(scene["chosen"])
    rejected = random.choice(scene["rejected"])
    return {
        "prompt": build_prompt(scene["turns"], scene["user"]),
        "chosen": chosen,
        "rejected": rejected,
        "tag": "question_not_accusation",
    }


def validate_i2(s: Dict) -> bool:
    chosen, rejected = s["chosen"], s["rejected"]
    if not validate_common(chosen):
        return False
    if not validate_rejected(rejected):
        return False
    if chosen == rejected:
        return False
    if not has_explain(chosen):
        return False
    if any(w in chosen for w in DRAMATIC_APOLOGY[:5]):
        return False
    if not any(w in rejected for w in DRAMATIC_APOLOGY):
        return False
    return True


# =============================================================================
# I3 — 中性陈述 ≠ 抱怨（statement_not_complaint）
# =============================================================================

_I3_SCENES = [
    {
        "turns": [("你周末有安排吗", "在家，没啥事")],
        "user": "哦，这样啊",
        "chosen": [
            "嗯，安静宅一天也挺好的，你那边这周末有没有什么打算嘛",
            "对，就在家安静待着，你那边呢，这周末有没有什么计划",
            "在家也好，安静，那你今天打算宅一天，有没有什么想做的",
            "那就好好放松，你今天在家打算干嘛，有没有什么好计划",
        ],
        "rejected": [
            "你是不是觉得无聊，要不要我陪你聊",
            "你这样听起来有点无聊，是不是有点孤独",
            "是不是有什么事，你听起来不太开心",
        ],
    },
    {
        "turns": [("你今天吃了什么", "随便弄了点，不怎么有胃口")],
        "user": "嗯，知道了",
        "chosen": [
            "没胃口就先凑合着，晚上可以吃点好的补回来，你想吃什么",
            "那就先将就，等胃口好了再吃好的，你晚上想吃什么，说说",
            "先垫垫就行，胃口不好的时候别勉强，你喜欢吃什么，我们定",
            "那就先将就着，晚上胃口好了再认真吃，你喜欢吃什么口味",
        ],
        "rejected": [
            "你是不是不开心，没有胃口一般是心情不好的信号",
            "是不是有什么烦心事，你没胃口让我担心",
            "你是不是心里有事，没胃口通常说明有情绪",
        ],
    },
    {
        "turns": [("你最近忙吗", "就一般，不算太忙")],
        "user": "哦",
        "chosen": [
            "那还好，不太忙的时候多休息一下，你今天有没有出去转转",
            "那就好，不忙可以好好缓缓，你今天都做了什么，说来聊聊",
            "不忙就好，那可以多给自己一点时间放松，你今天怎么过的",
            "那还不错，不太忙的时候多给自己放松一下，你今天怎么样",
        ],
        "rejected": [
            "你说话听起来有点心不在焉，是发生什么事了吗",
            "你是不是有什么心事，我感觉你不太对",
            "是不是有点不开心，你语气听着不对",
        ],
    },
    {
        "turns": [("你今天去哪了", "就在家附近转了转")],
        "user": "嗯",
        "chosen": [
            "出去透透气挺好的，那你转到哪里了，附近有什么好逛的地方",
            "出去走走挺好，活动一下，你附近有什么好玩的地方吗，说说",
            "好呀，出去逛逛挺好，那你附近有什么好转的地方，推荐我",
            "转转挺好的，你附近有什么好玩的地方，下次可以一起去",
        ],
        "rejected": [
            "你好像不太开心，是不是有什么事",
            "你是不是有点孤独，一个人转是不是有点难受",
            "是发生什么了吗，你听起来没精神",
        ],
    },
    {
        "turns": [("你睡得好吗", "还行，就那样")],
        "user": "这样啊",
        "chosen": [
            "就那样也算及格了嘛，总比睡不好强，你今晚打算几点睡",
            "还行就好，睡得过去就行，你今晚还打算聊多晚，我在这",
            "凑合也行，那就慢慢调整，你今晚打算几点睡，别太晚了",
            "那还可以，你今晚打算几点睡，我们可以聊一会儿再睡嘛",
        ],
        "rejected": [
            "你说话没什么精神，是不是睡眠质量不太好，要注意",
            "是不是有什么烦心事影响睡眠了",
            "你是不是睡得不好，我感觉你最近不太对",
        ],
    },
    {
        "turns": [("你今天见朋友了吗", "没有，就自己待着")],
        "user": "哦，自己待着",
        "chosen": [
            "自己待着也挺好的，安静，你一般宅在家做什么，说来聊聊",
            "那挺安静的，你喜欢这样，那你今天在家都干了什么，说说",
            "一个人待着也有一个人的好嘛，你今天在家都做了什么呢",
            "自己待着也挺好的，那你今天在家怎么过的，说来聊聊嘛",
        ],
        "rejected": [
            "你是不是有点孤独，一个人待着不会难受吗",
            "是不是有什么事情所以不想见人",
            "你是不是情绪不好，一个人待着听起来有点孤单",
        ],
    },
    {
        "turns": [("你们谈得怎么样", "就那样，还行")],
        "user": "嗯，还行就好",
        "chosen": [
            "那就行，顺利就好，那那件事定下来没有，你那边进展怎样",
            "还行就没问题，那你后续怎么安排，说来聊聊，我关注一下",
            "没出问题就好，那你接下来打算怎么推进，说来说说看嘛",
            "顺利就行，那接下来怎么安排，你那边有没有下一步的想法",
        ],
        "rejected": [
            "你这个还行听起来不太确定，是有什么问题吗",
            "是不是有点不顺，你语气有点勉强",
            "你是不是有什么烦心的，我感觉你不太对",
        ],
    },
    {
        "turns": [("你今天心情怎么样", "就普通，没什么特别的")],
        "user": "哦，普通就好",
        "chosen": [
            "普通挺好的，平平稳稳的，你今天都做了什么，说来聊聊嘛",
            "普通也不错，没什么波动，那你今天有没有出去转转，还是宅",
            "平平稳稳的也挺好，比大起大落省心多了，你今天怎么过的",
            "普通就挺好，那你今天都干了什么，有没有什么想分享的",
        ],
        "rejected": [
            "普通是不是有点无聊，你是不是其实有点闷",
            "是不是有什么事让你心情平平的",
            "你是不是有点不开心，普通听起来有点勉强",
        ],
    },
    {
        "turns": [("你最近睡眠怎么样", "将就吧")],
        "user": "嗯，将就",
        "chosen": [
            "那就行，慢慢调整，你今晚打算几点睡，别太晚了哦，注意",
            "将就也行，不太好就注意一下，你今晚打算几点睡，说说",
            "凑合着就好，有时候就是这样，你今晚有没有什么安排，聊聊",
            "那就慢慢来，你今晚打算几点睡，我们聊一会儿再去睡嘛",
        ],
        "rejected": [
            "将就听起来不太好，你是不是有什么烦心事",
            "是不是有什么事情影响了你睡眠",
            "你是不是睡不好，我有点担心你",
        ],
    },
    {
        "turns": [("你今天跑步了吗", "没有，懒得动")],
        "user": "哦，没动",
        "chosen": [
            "没事，有时候就是懒，明天再去，今天好好休息，你在家干嘛",
            "偶尔休息一天也正常，那今天在家宅着，你干了什么，说说",
            "懒一天也挺好的，明天再去，那今天在家都做了什么，聊聊",
            "今天休息一下也好，明天再补回来，你今天在家都干了什么",
        ],
        "rejected": [
            "懒得动是不是说明你今天状态不好，有什么事吗",
            "你是不是心情不好才不想动",
            "是不是有什么烦心事，你感觉没精神",
        ],
    },
]


def make_i3_sample() -> Dict:
    scene = random.choice(_I3_SCENES)
    chosen = random.choice(scene["chosen"])
    rejected = random.choice(scene["rejected"])
    return {
        "prompt": build_prompt(scene["turns"], scene["user"]),
        "chosen": chosen,
        "rejected": rejected,
        "tag": "statement_not_complaint",
    }


def validate_i3(s: Dict) -> bool:
    chosen, rejected = s["chosen"], s["rejected"]
    if not validate_common(chosen):
        return False
    if not validate_rejected(rejected):
        return False
    if chosen == rejected:
        return False
    if any(w in chosen for w in OVER_SYMPATHY):
        return False
    if not any(w in rejected for w in OVER_SYMPATHY):
        return False
    return True


# =============================================================================
# 通用采样框架
# =============================================================================

def collect_samples(target: int, builder, validator, results: List, seen: set) -> int:
    count = 0
    attempts = 0
    while count < target and attempts < target * 50:
        attempts += 1
        sample = builder()
        if not validator(sample):
            continue
        key = sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        results.append(sample)
        count += 1
    return count


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    random.seed(SEED)
    DST_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("🎯 dpo_data_intent_clarity.py — DPO 4.3 意图理解清晰度")
    print("=" * 72)
    print(f"输出: {DST_FILE}")
    print(f"配额: 纠正≠生气={QUOTA_I1}  疑问≠质问={QUOTA_I2}  中性陈述≠抱怨={QUOTA_I3}")
    print("=" * 72)

    results: List[Dict] = []
    seen: set = set()

    counts = {
        "correction_not_anger": collect_samples(QUOTA_I1, make_i1_sample, validate_i1, results, seen),
        "question_not_accusation": collect_samples(QUOTA_I2, make_i2_sample, validate_i2, results, seen),
        "statement_not_complaint": collect_samples(QUOTA_I3, make_i3_sample, validate_i3, results, seen),
    }

    random.shuffle(results)

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
