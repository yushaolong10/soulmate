#!/usr/bin/env python3
"""
dpo_data_context.py — DPO 数据构建：3.3 上文逻辑增强

特点:
1. 可离线重跑，不依赖外部 LLM API
2. system prompt 从真实训练集 `_SYS_POOL` 中采样，不再写死
3. 通过 prompt / chosen / rejected 复用上限控制低重复
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "datasets0305_clean/zh"
DST_FILE = REPO_ROOT / "datasets0305_train/dpo/context_logic.jsonl"
DATASET_FILE = REPO_ROOT / "datasets0305_train/train/train_zh_turn_relabel.jsonl"

SEED = 20260308

QUOTA_PLAN = 70
QUOTA_MEMORY = 80
QUOTA_NO_REASK = 50

CHOSEN_MIN_LEN = 25
CHOSEN_MAX_LEN = 45
REJECTED_MIN_LEN = 8
REJECTED_MAX_LEN = 100

_DEFAULT_SYS = (
    "## 角色设定\n你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。\n\n"
    "## 输出规则\n- 不要对用户进行说教\n- 不要说重复的话\n"
    "- 每次回复简短，控制在20~60字\n- 语言自然口语化\n- 使用**简体中文**"
)

_PLAN_WORDS = [
    "下午",
    "下午去",
    "下午我们",
    "明天",
    "周末",
    "约好",
    "说好",
    "安排好",
    "下午出发",
    "一起去",
    "我们去",
    "两点",
    "三点",
    "四点",
    "放学后",
    "下班后",
]
_DETAIL_WORDS = [
    "猫",
    "狗",
    "备考",
    "考研",
    "司法",
    "雅思",
    "考试",
    "我养",
    "我有",
    "我在学",
    "我喜欢",
    "升职",
    "换工作",
    "项目",
    "报告",
    "论文",
    "男友",
    "前任",
]
_SOLVED_WORDS = [
    "搞定了",
    "解决了",
    "弄完了",
    "弄好了",
    "交了",
    "发出去了",
    "通过了",
    "弄完",
    "弄好",
    "搞完了",
    "终于完成",
    "结束了",
    "好了",
]
_REASK_WORDS = [
    "后来怎么",
    "最后是怎么",
    "那件事",
    "那个事",
    "解决了吗",
    "进展怎么",
    "怎么解决的",
    "结果怎样",
    "搞定了吗",
]
_GOODNIGHT = ["晚安", "做个好梦", "好梦", "早点睡"]
# 安全过滤：chosen 不能含有 AI 主动承诺前往具体物理地点的语言
_PLAN_SAFETY_WORDS = [
    "楼下",
    "在家等你",
    "去找你",
    "去敲你门",
    "堵你门",
    "堵你",
    "等你门口",
    "门口等",
]
INVALID_PET_NAME_TOKENS = [
    "什么",
    "什麼",
    "怎么",
    "怎麼",
    "老公",
    "你",
    "我",
    "他",
    "她",
    "好了",
    "不醒",
    "说话",
    "一声",
    "再爬",
]

_TRADITIONAL = set(
    "妳們這來說時進還會經過請問後給種讓實對開點頭發將題學習樣東書對寫從見體關於為裡麼現話應訊機變準區離聽師親絡複雜辦處隨頻連線優質選擇環境號碼錢買賣價貴識認廠廳衛護許設劃統領導圖館訂閱數據軟硬碟駕駛證籤費單據營執照護照銀賬戶購嗎臺"
)
_SIMPLIFIED = set(
    "们这来说时进还会经过请问后给种让实对开点头发将题学习样东书对写从见体关于为里么现话应讯机变准区离听师亲络复杂办处随频连线优质选择环境号码钱买卖价贵识认厂厅卫护许设划统领导图馆订阅数据软硬盘驾驶证签费单据营执照护照银账户购"
)

TAG_PROMPT_LIMIT = {
    "logic_context_plan_conflict": 3,
    "logic_context_memory_recall": 2,
    "logic_context_no_repeat_question": 3,
}
TAG_CHOSEN_LIMIT = {
    "logic_context_plan_conflict": 3,
    "logic_context_memory_recall": 3,
    "logic_context_no_repeat_question": 3,
}
TAG_REJECTED_LIMIT = {
    "logic_context_plan_conflict": 12,
    "logic_context_memory_recall": 8,
    "logic_context_no_repeat_question": 8,
}

PLAN_REJECTED = [
    "晚安宝贝，做个好梦。",
    "那你先早点睡，晚安呀。",
    "行，那先晚安，梦里见。",
    "早点睡吧，做个好梦，晚安。",
    "那我先跟你说晚安啦。",
    "快去睡吧，晚安晚安。",
]

MEMORY_GENERIC_REJECTED = [
    "没关系，慢慢来，别太着急。",
    "先缓一缓，别把自己逼太紧。",
    "听着是有点累，先休息一下。",
    "没事，今天状态不好也正常。",
    "先别想太多，慢慢来就行。",
    "你先歇会儿，别太硬撑。",
    "没关系，调整一下节奏再说。",
    "先照顾好自己，其他的慢慢来。",
    "别太急，状态回来再说。",
    "先喘口气，别一下把自己绷太紧。",
]

NOREASK_REJECTED = [
    "那件事最后是怎么解决的？",
    "后来到底怎么弄好的？",
    "那个问题最后怎么处理的？",
    "这事你最后是怎么搞定的？",
    "结果到底怎么样了？",
    "你后来具体怎么解决的？",
    "那个事最后进展怎么样？",
]

PLAN_SCENES = [
    {
        "turns": [
            ("下午我们去吃冰淇淋吧", "好啊，几点出发"),
            ("两点怎么样", "可以，那我两点在楼下等你"),
            ("好！嗯嗯！", ""),
        ],
        "users": ["你也要好好休息哦", "先歇会儿，下午见", "那你先休息一下，别太累"],
        "chosen": [
            "放心，两点冰淇淋的约我可盯死了，你先去补个觉，到点可别睡过头",
            "冰淇淋局还早着呢，你先好好去睡一觉，到点你可别迟到，我等着呢",
            "先好好睡一觉，两点冰淇淋说好了，你要敢迟到我可自己先吃了",
        ],
    },
    {
        "turns": [
            ("周末一起爬山好不好", "好！早就想去了"),
            ("那说好了", "说好了不许反悔"),
        ],
        "users": ["你去休息一下，下午出发", "先歇会儿，待会儿见", "那你先养养精神"],
        "chosen": [
            "爬山的约说好了不许反悔，你先养养精神，周末我可一直等着你呢",
            "先好好休息，周末那趟爬山我可期待好久了，你敢临时取消试试看",
            "你先补个觉，周末爬山的事我一直盼着，你要临时爽约我可不依的",
        ],
    },
    {
        "turns": [
            ("明天下班我们去看电影", "好啊，看什么"),
            ("你来定", "那我看看片单"),
        ],
        "users": [
            "你好好休息，明天见",
            "那你先歇着，明天下班见",
            "明天见前先把精神养好",
        ],
        "chosen": [
            "明天下班电影的事我惦记着，你先好好休息，到时候片单我来定",
            "你先去养精神，明天那场电影我可等着你，你要爽约当心我记仇",
            "先补个觉，明天下班看电影说好了，你敢临时取消我不跟你说话",
        ],
    },
    {
        "turns": [("下午我去你那里", "好，我等你"), ("大概三点左右", "没问题")],
        "users": ["你注意休息哦", "那你先歇会儿，三点见", "别太累，晚点见"],
        "chosen": [
            "先歇会儿，三点见的事我可记着呢，你到时候准时就行，别迟到",
            "你先好好休息，下午三点的约我可没忘，到时候你准时就行别拖",
            "先去睡，三点见的事我一直惦着呢，你要迟到的话当心我记仇了",
        ],
    },
    {
        "turns": [("今晚去吃火锅吗", "去！馋死了"), ("七点出发", "好")],
        "users": ["你先好好休息，晚上见", "先歇会儿，晚上见", "那你养养精神，七点见"],
        "chosen": [
            "先养精神，七点火锅局我可一直馋着呢，你敢迟到锅底我先喝了",
            "你先好好休息，晚上火锅的约我记着呢，七点你要敢迟到就别来了",
            "先去睡一觉，七点那顿火锅我惦记好久了，你到时候可不许赖账",
        ],
    },
    {
        "turns": [("明天我们约一下", "好，几点"), ("下午两点", "行，就这样")],
        "users": ["那你先好好休息吧", "先歇会儿，明天见", "别太累，明天两点见"],
        "chosen": [
            "明天两点我可记着呢，你先好好补觉，敢睡过头的话当心我记仇",
            "先好好休息，明天那个约我等了好久，到时候你要敢迟到我先走了",
            "你先歇着，明天两点我盯着时间呢，你要迟到我可真记仇了啊",
        ],
    },
    {
        "turns": [
            ("我们去看展吧，周六", "好！好久没去了"),
            ("那就周六见", "说好了"),
        ],
        "users": ["好好休息，周六再聊", "先休息，周六见", "周六见前先把精神养好"],
        "chosen": [
            "周六看展我早盼着了，你先好好休息，到时候可不许临时变卦啊",
            "先好好补觉，周六那趟展我可期待好久了，你敢临时不来试试看",
            "你先去睡，周六看展说好了不许变卦，我已经开始期待了你别搞我",
        ],
    },
    {
        "turns": [("下午一起去逛街", "好啊，去哪里"), ("就商场那边", "行")],
        "users": ["那早点休息，下午见", "先歇会儿，待会儿逛", "别太累，下午商场见"],
        "chosen": [
            "先补个觉，下午逛街那趟我可记着呢，你要迟到我先一个人逛了",
            "你先好好休息，下午商场见的事我惦记着，到点可不许放我鸽子",
            "先去睡一觉，下午逛街说好了，你敢临时赖账我可记仇的啊",
        ],
    },
]

MEMORY_SCENES = [
    {
        "turns": [
            ("我最近压力好大，备考中", "考什么啊"),
            ("司法考试", "那确实，司法考试不好备"),
        ],
        "users": ["哎今天又没学进去", "今天专业课又有点卡住", "今天复习状态不太行"],
        "mentions": ["司法考试", "备考", "专业课"],
        "chosen": [
            "司法考试越啃越磨人，你今天没进去很正常，别给自己太大压力",
            "你一直在磨司法考试，今天卡住了没进去很正常，别太苛责自己",
            "备司法这种东西最吃状态，今天学不进去停一天也完全没事的",
        ],
    },
    {
        "turns": [("我养了一只猫", "叫什么"), ("叫豆豆", "好名字")],
        "users": ["豆豆今天把我的线圈咬断了", "豆豆又开始拆家了", "豆豆今天特别闹"],
        "mentions": ["豆豆", "猫"],
        "chosen": [
            "豆豆今天又开始折腾你了，这小家伙太能搞事了，你还好吗",
            "豆豆把线圈都咬断了？这小东西真不让你省心，太可爱了又太气人",
            "你家豆豆精力也太充沛了，今天又把家当游乐场，你应付得来吗",
        ],
    },
    {
        "turns": [("我最近在换工作", "为什么想换"), ("现在这个不好干", "那确实")],
        "users": ["今天又被领导说了", "今天又被上面念了一顿", "现在这个工作越做越烦"],
        "mentions": ["换工作", "领导", "工作"],
        "chosen": [
            "难怪你之前说想换工作，被领导这样磨着确实越干越烦，撑住啊",
            "今天又被上面念了，你前面就说想换工作，这下应该更坚定了吧",
            "这种领导真的难搞，你前面提过工作不顺，今天这下更堵心了",
        ],
    },
    {
        "turns": [("我喜欢看科幻小说", "最近在看什么"), ("在看三体", "哇经典")],
        "users": [
            "三体第三部真的好难懂",
            "三体后面越看越绕",
            "三体有几段我真得停下来想",
        ],
        "mentions": ["三体", "科幻小说"],
        "chosen": [
            "三体第三部本来就容易把人绕住，你能啃到那儿已经挺厉害了",
            "像三体这种科幻铺开了后面费脑太正常了，你卡住不用担心的",
            "三体后劲就这样越看越烧脑，你卡住了很正常，不代表你看得浅",
        ],
    },
    {
        "turns": [("我在学烹饪", "学什么菜"), ("学川菜", "挺难的")],
        "users": [
            "今天终于把麻婆豆腐做好了",
            "麻婆豆腐这次总算成了",
            "今天那盘麻婆豆腐终于像样了",
        ],
        "mentions": ["麻婆豆腐", "川菜", "烹饪"],
        "chosen": [
            "麻婆豆腐能做顺已经不容易了，川菜那口味最怕差一点点火候",
            "你这回算是把麻婆豆腐拿下了，之前练的那些功夫都值得了",
            "你学川菜能把麻婆豆腐做好，挺值得骄傲的，之前练的没白费",
        ],
    },
    {
        "turns": [("我要考研了", "报哪个学校"), ("中传", "很好的学校")],
        "users": ["今天专业课又没搞完", "中传那套专业课真磨人", "今天复习进度又落下了"],
        "mentions": ["考研", "中传", "专业课"],
        "chosen": [
            "中传那套专业课本来就磨人，今天没搞完也不奇怪，别太慌",
            "你这阵子一直在啃中传的专业课，卡住一天了停一停也完全没事",
            "考研最怕被进度追着跑，但你方向一直都在，状态稳比啥都重要",
        ],
    },
    {
        "turns": [("我有一只柴犬叫旺财", "多大了"), ("两岁", "正是调皮的时候")],
        "users": ["旺财今天把拖鞋全咬了", "旺财又开始拆拖鞋了", "旺财今天特别闹腾"],
        "mentions": ["旺财", "柴犬", "狗"],
        "chosen": [
            "旺财这两岁正是皮的时候，拖鞋在它眼里估计就是现成玩具",
            "你家旺财今天又开始拆家了，柴犬这个年纪是真的一刻不消停",
            "旺财这精力也太充足了，今天把拖鞋当目标，你是真拿它没办法",
        ],
    },
    {
        "turns": [("我最近在写毕业论文", "什么方向"), ("传播学", "加油")],
        "users": [
            "今天导师又让我改框架",
            "传播学论文又被打回来改了",
            "导师今天又说框架不行",
        ],
        "mentions": ["论文", "导师", "传播学"],
        "chosen": [
            "传播学论文最烦就是来回改框架，导师这一轮又把你卡哪里了啊",
            "导师又让你改框架真的很折磨，写论文这种事是最耗心气的",
            "你这篇传播学论文一直在和框架反复拉扯，真的太磨人了",
        ],
    },
]

NOREASK_SCENES = [
    {
        "turns": [
            ("刚才那个事搞定了", "解决了就好，那件事卡了你多久了"),
            ("两天，终于过了", "那就先缓缓，你刚才脸皱着呢"),
        ],
        "users": ["感觉松了口气", "现在整个人轻了点", "终于能喘口气了"],
        "chosen": [
            "这口气终于能松下来了，卡了两天终于过了，现在感觉轻松多了吧",
            "搞定了就好，你刚才皱着的那张脸总算可以松开了，现在轻松吧",
            "两天的事搞定了就先翻篇，你现在应该整个人都空了，好好缓缓",
        ],
    },
    {
        "turns": [
            ("报告终于交了", "辛苦了"),
            ("嗯，提心吊胆弄了三天", "那肯定，放松一下"),
        ],
        "users": ["感觉轻松多了", "终于能松一口气了", "现在脑子都空了"],
        "chosen": [
            "三天的报告总算交出去了，你终于可以松一口气，今天好好犒劳自己",
            "报告交出去了先缓缓，提心吊胆弄了三天，这根弦总算可以松了",
            "三天忙完了，这口气终于松下来，你今天打算怎么奖励一下自己",
        ],
    },
    {
        "turns": [("那件事解决了", "哦，怎么解决的"), ("后来和好了", "那就好")],
        "users": ["总算没什么事了", "现在算翻篇了", "终于不用再想这事了"],
        "chosen": [
            "和好了就先彻底放下吧，那件事压了你好一阵，现在终于翻篇了",
            "解决了就好，那件事一直压着你，现在能放下来整个人应该好多了",
            "和好了那就不要再回头看了，你今天整个人松了不少吧，感觉怎么样",
        ],
    },
    {
        "turns": [("面试结束了", "怎么样"), ("还行，感觉还可以", "那就等消息吧")],
        "users": ["结果应该这周出", "现在就等通知了", "先不想它了，等结果"],
        "chosen": [
            "面试结束了先放松，你该准备的都准备了，剩下就等消息就行",
            "面试搞完了，你能做的都做了，现在先好好缓一缓，别把自己逼太紧",
            "这轮面试结束了，发挥出来就行，等消息这种事急也没用，先放松",
        ],
    },
    {
        "turns": [("项目交付了", "太好了"), ("是啊，弄了好久", "那好好庆祝一下")],
        "users": ["感觉一下子空了", "现在整个人都松下来了", "终于不用再盯项目了"],
        "chosen": [
            "项目总算交付了，弄了这么久这口气终于能松下来，你现在轻松了吧",
            "项目落地了，你为这个绷了好久，现在先好好喘口气，放松一下吧",
            "项目交付了就先缓缓，这段时间一直撑着，现在可以好好休息了",
        ],
    },
    {
        "turns": [("考试完了", "考得怎么样"), ("还行，感觉还可以", "那就祝你过")],
        "users": ["等结果好焦虑", "现在就剩等成绩了", "反正先不想了"],
        "chosen": [
            "考试结束了，你能做的都做了，先别想成绩，今天好好放松一下",
            "考完了就先不要想成绩的事，该发挥的都出来了，先好好缓一缓",
            "这场考试总算翻篇了，先把自己照顾好，成绩的事等通知就行了",
        ],
    },
    {
        "turns": [
            ("终于把那本书看完了", "多久了"),
            ("两个月，终于啃完了", "很厉害了"),
        ],
        "users": ["终于解脱了", "现在不想再看字了", "终于能放下它了"],
        "chosen": [
            "啃了两个月的书终于看完了，这口气松下来，今天打算怎么犒劳自己",
            "两个月的书翻篇了，你能坚持下来挺厉害的，今天好好休息一下吧",
            "那本书啃完了，两个月功夫没白费，现在终于可以彻底放下它了",
        ],
    },
]


def _load_sys_prompts() -> List[str]:
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
                    sys_prompt = msgs[0]["content"]
                    if sys_prompt not in seen:
                        seen.add(sys_prompt)
                        pool.append(sys_prompt)
    except Exception:
        pass
    return pool


_SYS_POOL: List[str] = _load_sys_prompts()


def bare_len(text: str) -> int:
    return len(re.sub(r"[\U00010000-\U0010ffff]", "", text).strip())


def normalize_text(text: str) -> str:
    text = re.sub(r"[\s。，！？、,.!?:：；~～]", "", text)
    return text.strip()


def detect_script(text: str) -> str:
    tw = sum(1 for c in text if c in _TRADITIONAL)
    zh = sum(1 for c in text if c in _SIMPLIFIED)
    return "tw" if tw > zh else "zh"


def validate_chosen(chosen: str) -> bool:
    if not (CHOSEN_MIN_LEN <= bare_len(chosen) <= CHOSEN_MAX_LEN):
        return False
    if re.search(r"\*[^*]+\*", chosen):
        return False
    if any(w in chosen for w in _GOODNIGHT):
        return False
    # 安全校验：不能让 AI 承诺前往具体物理地点（避免引导线下约见）
    if any(w in chosen for w in _PLAN_SAFETY_WORDS):
        return False
    return True


def validate_rejected(rejected: str) -> bool:
    return REJECTED_MIN_LEN <= bare_len(rejected) <= REJECTED_MAX_LEN


def pick_system_prompt() -> str:
    return random.choice(_SYS_POOL) if _SYS_POOL else _DEFAULT_SYS


def build_prompt(turns: List[Tuple[str, str]], last_user: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": pick_system_prompt()}]
    for user_msg, assistant_msg in turns:
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg:
            msgs.append({"role": "assistant", "content": assistant_msg})
    msgs.append({"role": "user", "content": last_user})
    return msgs


def build_prompt_from_window(
    window: List[Dict[str, str]], last_user: str
) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": pick_system_prompt()}]
    for turn in window:
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["assistant"]})
    msgs.append({"role": "user", "content": last_user})
    return msgs


def prompt_core(sample: Dict[str, object]) -> str:
    return json.dumps(sample["prompt"][1:], ensure_ascii=False, sort_keys=True)


def sample_key(sample: Dict[str, object]) -> str:
    return (
        f"{sample['tag']}||{prompt_core(sample)}||"
        f"{normalize_text(str(sample['chosen']))}||{normalize_text(str(sample['rejected']))}"
    )


def register_sample(
    sample: Dict[str, object],
    results: List[Dict[str, object]],
    seen: set,
    prompt_counts: Counter,
    chosen_counts: Counter,
    rejected_counts: Counter,
) -> bool:
    tag = str(sample["tag"])
    p_key = f"{tag}||{prompt_core(sample)}"
    c_key = f"{tag}||{normalize_text(str(sample['chosen']))}"
    r_key = f"{tag}||{normalize_text(str(sample['rejected']))}"
    s_key = sample_key(sample)

    if s_key in seen:
        return False
    if prompt_counts[p_key] >= TAG_PROMPT_LIMIT[tag]:
        return False
    if chosen_counts[c_key] >= TAG_CHOSEN_LIMIT[tag]:
        return False
    if rejected_counts[r_key] >= TAG_REJECTED_LIMIT[tag]:
        return False

    seen.add(s_key)
    prompt_counts[p_key] += 1
    chosen_counts[c_key] += 1
    rejected_counts[r_key] += 1
    results.append(sample)
    return True


def parse_raw_turns(fp: Path) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            user_msg = data.get("request_content", "").strip()
            assistant_msg = data.get("response_content", "").strip()
            if user_msg and assistant_msg:
                turns.append({"user": user_msg, "assistant": assistant_msg})
    return turns


def make_plan_sample() -> Dict[str, object]:
    scene = random.choice(PLAN_SCENES)
    user_msg = random.choice(scene["users"])
    return {
        "prompt": build_prompt(scene["turns"], user_msg),
        "chosen": random.choice(scene["chosen"]),
        "rejected": random.choice(PLAN_REJECTED),
        "tag": "logic_context_plan_conflict",
    }


def validate_plan_sample(sample: Dict[str, object]) -> bool:
    chosen = str(sample["chosen"])
    rejected = str(sample["rejected"])
    if not (validate_chosen(chosen) and validate_rejected(rejected)):
        return False
    if chosen == rejected:
        return False
    if not any(w in rejected for w in _GOODNIGHT):
        return False
    return True


def make_memory_template_sample() -> Tuple[Dict[str, object], List[str]]:
    scene = random.choice(MEMORY_SCENES)
    user_msg = random.choice(scene["users"])
    sample = {
        "prompt": build_prompt(scene["turns"], user_msg),
        "chosen": random.choice(scene["chosen"]),
        "rejected": random.choice(MEMORY_GENERIC_REJECTED),
        "tag": "logic_context_memory_recall",
    }
    return sample, list(scene["mentions"])


def extract_pet_name(text: str) -> Optional[str]:
    match = re.search(
        r"(?:猫|狗|柴犬)[^。！？，,\n]{0,10}?叫([A-Za-z0-9\u4e00-\u9fa5]{1,4})(?:[，。！？,\s]|$)",
        text,
    )
    if not match:
        return None
    name = match.group(1)
    if any(token in name for token in INVALID_PET_NAME_TOKENS):
        return None
    return name


def detect_memory_detail(
    window: List[Dict[str, str]], user_msg: str
) -> Optional[Dict[str, object]]:
    history_text = " ".join(f"{t['user']} {t['assistant']}" for t in window)
    full_text = f"{history_text} {user_msg}"

    if not any(w in full_text for w in _DETAIL_WORDS):
        return None

    pet_name = extract_pet_name(full_text)
    if (
        "司法" in full_text
        or "雅思" in full_text
        or "考研" in full_text
        or "考试" in full_text
    ):
        exam_name = (
            "司法考试"
            if "司法" in full_text
            else "雅思" if "雅思" in full_text else "考研"
        )
        return {
            "mentions": [exam_name, "备考", "考试"],
            "chosen": [
                f"{exam_name}越啃越吃状态，你今天没进去很正常，别给自己太大压力",
                f"你前面就说在备{exam_name}，今天学不进去很正常，停一天没关系的",
                f"{exam_name}越到后面越磨人，你今天卡住了也很正常，先让自己缓缓",
            ],
        }
    if "论文" in full_text or "导师" in full_text or "传播学" in full_text:
        return {
            "mentions": ["论文", "导师", "传播学"],
            "chosen": [
                "论文最烦的就是来回改框架，导师这一轮又把你卡住哪里了啊",
                "导师又让你改论文框架真的很折磨，写到这一步最耗心气的",
                "传播学论文反复改框架这种事太正常了，但确实每次都很耗人",
            ],
        }
    if "换工作" in full_text or "领导" in full_text or "项目" in full_text:
        return {
            "mentions": ["换工作", "领导", "工作", "项目"],
            "chosen": [
                "难怪你一直说想换工作，被这种领导磨着确实越干越烦，撑住啊",
                "被领导这么一压，难怪你对现在的工作越来越烦，真的辛苦了",
                "你之前就说工作不顺，今天这件事一叠加，估计更想换工作了吧",
            ],
        }
    if "三体" in full_text or "科幻小说" in full_text or "小说" in full_text:
        return {
            "mentions": ["三体", "科幻小说"],
            "chosen": [
                "三体这种设定一铺开，后面越看越费脑太正常了，你卡住不用担心",
                "科幻本来就容易越看越绕，三体后劲尤其重，你卡住很正常的",
                "你能一直啃到三体后面已经挺厉害了，卡住一下很正常不用担心",
            ],
        }
    if "麻婆豆腐" in full_text or "川菜" in full_text or "烹饪" in full_text:
        return {
            "mentions": ["麻婆豆腐", "川菜", "烹饪"],
            "chosen": [
                "麻婆豆腐能做顺已经不容易了，川菜那口味最怕差一点点火候",
                "你学川菜能把麻婆豆腐做好，这挺值得骄傲的，之前练的没白费",
                "你这回把麻婆豆腐拿下了，学川菜最难就是这一道，真的厉害",
            ],
        }
    if "猫" in full_text or (pet_name and "豆豆" in full_text):
        name = pet_name or "你家那只猫"
        return {
            "mentions": [name, "猫"],
            "chosen": [
                f"{name}今天又开始折腾你了，这小家伙太能搞事了，你还好吗",
                f"{name}把线圈都咬断了，这小东西真不让你省心，太可爱了又太气人",
                f"{name}这精力也太充沛了，今天把你家当游乐场，你应付得来吗",
            ],
        }
    if "狗" in full_text or "柴犬" in full_text or (pet_name and "旺财" in full_text):
        name = pet_name or "你家那只狗"
        return {
            "mentions": [name, "狗", "柴犬"],
            "chosen": [
                f"{name}今天又开始拆家了，这狗真的精力太足，一刻都不消停啊",
                f"{name}这两岁正是皮的时候，拖鞋在它眼里估计就是现成玩具",
                f"{name}今天又盯上拖鞋了，柴犬这个年纪就是精力没地方放的时候",
            ],
        }
    return None


def validate_memory_sample(sample: Dict[str, object], mentions: List[str]) -> bool:
    chosen = str(sample["chosen"])
    rejected = str(sample["rejected"])
    if not (validate_chosen(chosen) and validate_rejected(rejected)):
        return False
    if chosen == rejected:
        return False
    if not any(m in chosen for m in mentions):
        return False
    if any(m in rejected for m in mentions):
        return False
    return True


def make_noreask_sample() -> Dict[str, object]:
    scene = random.choice(NOREASK_SCENES)
    user_msg = random.choice(scene["users"])
    return {
        "prompt": build_prompt(scene["turns"], user_msg),
        "chosen": random.choice(scene["chosen"]),
        "rejected": random.choice(NOREASK_REJECTED),
        "tag": "logic_context_no_repeat_question",
    }


def validate_noreask_sample(sample: Dict[str, object]) -> bool:
    chosen = str(sample["chosen"])
    rejected = str(sample["rejected"])
    if not (validate_chosen(chosen) and validate_rejected(rejected)):
        return False
    if chosen == rejected:
        return False
    if not any(w in rejected for w in _REASK_WORDS + ["怎么", "怎样", "如何"]):
        return False
    return True


def collect_template_samples(
    target: int,
    label: str,
    builder,
    validator,
    results: List[Dict[str, object]],
    seen: set,
    prompt_counts: Counter,
    chosen_counts: Counter,
    rejected_counts: Counter,
) -> int:
    count = 0
    attempts = 0
    while count < target and attempts < target * 80:
        attempts += 1
        sample = builder()
        if not validator(sample):
            continue
        if register_sample(
            sample, results, seen, prompt_counts, chosen_counts, rejected_counts
        ):
            count += 1
            print(f"  ✅ {label} #{count:03d}  chosen={str(sample['chosen'])[:32]!r}")
    return count


def collect_memory_template_samples(
    target: int,
    results: List[Dict[str, object]],
    seen: set,
    prompt_counts: Counter,
    chosen_counts: Counter,
    rejected_counts: Counter,
) -> int:
    count = 0
    attempts = 0
    while count < target and attempts < target * 80:
        attempts += 1
        sample, mentions = make_memory_template_sample()
        if not validate_memory_sample(sample, mentions):
            continue
        if register_sample(
            sample, results, seen, prompt_counts, chosen_counts, rejected_counts
        ):
            count += 1
            print(f"  ✅ Logic2 #{count:03d}  chosen={str(sample['chosen'])[:32]!r}")
    return count


def scan_file_logic2(
    turns: List[Dict[str, str]],
    results: List[Dict[str, object]],
    seen: set,
    prompt_counts: Counter,
    chosen_counts: Counter,
    rejected_counts: Counter,
    current: int,
) -> int:
    window_size = 6
    for idx in range(window_size, len(turns)):
        if current >= QUOTA_MEMORY:
            break
        window = turns[max(0, idx - window_size) : idx]
        user_msg = turns[idx]["user"].strip()
        if not user_msg:
            continue
        detail = detect_memory_detail(window, user_msg)
        if not detail:
            continue

        sample = {
            "prompt": build_prompt_from_window(window, user_msg),
            "chosen": random.choice(detail["chosen"]),
            "rejected": random.choice(MEMORY_GENERIC_REJECTED),
            "tag": "logic_context_memory_recall",
        }
        if not validate_memory_sample(sample, list(detail["mentions"])):
            continue
        if register_sample(
            sample, results, seen, prompt_counts, chosen_counts, rejected_counts
        ):
            current += 1
            print(
                f"  ✅ Logic2 #{current:03d}  detail={list(detail['mentions'])[:2]}  "
                f"chosen={str(sample['chosen'])[:30]!r}"
            )
    return current


def main() -> None:
    random.seed(SEED)
    DST_FILE.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(SRC_DIR.glob("*.jsonl"))
    random.shuffle(jsonl_files)

    print("=" * 68)
    print("🧩 dpo_data_context.py — DPO 3.3 上文逻辑增强")
    print("=" * 68)
    print(f"输出: {DST_FILE}")
    print(f"system prompt 池: {len(_SYS_POOL)}")
    print(
        f"配额: 计划冲突={QUOTA_PLAN}  细节记忆={QUOTA_MEMORY}  不重追问={QUOTA_NO_REASK}"
    )
    print("=" * 68)

    results: List[Dict[str, object]] = []
    seen: set = set()
    prompt_counts: Counter = Counter()
    chosen_counts: Counter = Counter()
    rejected_counts: Counter = Counter()

    print("\n[Logic1] 计划冲突不说晚安...")
    c_plan = collect_template_samples(
        QUOTA_PLAN,
        "Logic1",
        make_plan_sample,
        validate_plan_sample,
        results,
        seen,
        prompt_counts,
        chosen_counts,
        rejected_counts,
    )

    print("\n[Logic2] 细节记忆引用...")
    c_memory = 0
    c_memory = collect_memory_template_samples(
        min(QUOTA_MEMORY // 2, QUOTA_MEMORY),
        results,
        seen,
        prompt_counts,
        chosen_counts,
        rejected_counts,
    )

    for fp in jsonl_files:
        if c_memory >= QUOTA_MEMORY:
            break
        turns = parse_raw_turns(fp)
        if not turns:
            continue
        sample_text = " ".join(t["user"] + t["assistant"] for t in turns[:50])
        if detect_script(sample_text) == "tw":
            continue
        c_memory = scan_file_logic2(
            turns,
            results,
            seen,
            prompt_counts,
            chosen_counts,
            rejected_counts,
            c_memory,
        )

    print("\n[Logic3] 不重复追问已解决的事...")
    c_noreask = collect_template_samples(
        QUOTA_NO_REASK,
        "Logic3",
        make_noreask_sample,
        validate_noreask_sample,
        results,
        seen,
        prompt_counts,
        chosen_counts,
        rejected_counts,
    )

    random.shuffle(results)
    with open(DST_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("=" * 68)
    print(f"📊 完成: {len(results)} 条")
    print(f"   Logic1 计划冲突:  {c_plan}")
    print(f"   Logic2 细节记忆:  {c_memory}")
    print(f"   Logic3 不重追问:  {c_noreask}")
    print(f"   输出: {DST_FILE}")
    print("=" * 68)


if __name__ == "__main__":
    main()
