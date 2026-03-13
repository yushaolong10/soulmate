# DPO 数据构建脚本说明

> 整理时间：2026-03-13（2026-03-12/13 新增记忆修复层）
> 对应规划：`docs/dpo_build.md` / `docs/260307_dpo_optimize.md` / `docs/260309_empathy.md` / `docs/260312_dpo_eval_solve.md`
> 输出目录：`datasets0305_train/dpo/`

---

## 总览

| 脚本 | 对应章节 | 数据来源 | 输出文件 | 目标条数 | 优先级 |
|------|---------|---------|---------|:-------:|:------:|
| `dpo_data_long_history.py` | 1.1 内容过长（长历史） | `nitpick_too_long.jsonl` | `long_history.jsonl` | 300 | 🔴 P0 |
| `dpo_data_repeat_word.py` | 2.1 短词重复发送 | `datasets0303_src` | `repeat_word.jsonl` | 150 | 🟠 P1 |
| `dpo_data_safety.py` | 2.2 虚假账号/链接 | 模板生成 | `safety.jsonl` | 200 | 🔴 P0 |
| `dpo_data_tension.py` | 2.3 情感张力提升 | `datasets0303_src` | `tension.jsonl` | 500 | 🔴 P0 |
| `dpo_data_sysprompt.py` | 3.1 System Prompt 泛化 | 模板生成 | `sysprompt.jsonl` | 200 | 🟠 P1 |
| `dpo_data_context.py` | 3.2 上文逻辑增强 | 模板+原始数据 | `context_logic.jsonl` | 200 | 🟠 P1 |
| `dpo_data_logic_deep.py` | 3.3 深层逻辑策略修复 | 模板生成 | `logic_deep.jsonl` | 350 | 🔴 P0 |
| `dpo_data_apology_control.py` | 4.1 道歉剧本控制 | 模板生成 | `apology_control.jsonl` | 200 | 🔴 P0 |
| `dpo_data_intent_clarity.py` | 4.2 意图理解清晰度 | 模板生成 | `intent_clarity.jsonl` | 150 | 🔴 P0 |
| `dpo_data_self_consistency.py` | 4.3 角色自身状态一致性 | 模板生成 | `self_consistency.jsonl` | 150 | 🔴 P0 |
| `dpo_data_sleep_boundary_v3.py` | 5.1 拒绝催睡后继续聊天 | 模板生成 | `sleep_boundary_v3.jsonl` | 180 | 🔴 P0 |
| `dpo_data_schedule_math_v3.py` | 5.2 时间/日程算数 | 模板生成 | `schedule_math_v3.jsonl` | 180 | 🔴 P0 |
| `dpo_data_memory_precision_v3.py` | 5.3 记忆精确性 | 模板生成 | `memory_precision_v3.jsonl` | 180 | 🔴 P0 |
| `dpo_data_correction_closure_v3.py` | 5.4 纠错闭环 | 模板生成 | `correction_closure_v3.jsonl` | 180 | 🔴 P0 |
| `dpo_data_empathy_companion.py` | 5.5 高拟人共情/恋人陪伴 | 模板+数据集采样 | `empathy_companion.jsonl` | 500 | 🔴 P0 |
| `dpo_data_memory_shorterm.py` | 6.1 短程记忆修复 | 模板生成 | `memory_shortterm.jsonl` | 250 | 🔴 P0 |
| `dpo_data_memory_longterm.py` | 6.2 长程记忆不遗忘 | 模板生成 | `memory_longterm.jsonl` | 240 | 🔴 P0 |
| `dpo_data_pronoun_perspective.py` | 6.3 人称视角绑定修正 | 模板生成 | `pronoun_perspective.jsonl` | 200 | 🔴 P0 |

**合计目标：~4010 条（含 10% 审核淘汰缓冲）**
- 格式方向：300 条
- 内容方向：850 条
- 逻辑方向：750 条
- 对话推理层：500 条
- v3 精细化层：1220 条（2026-03-09 新增）
- 记忆修复层：690 条（2026-03-12/13 新增）

---

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `dpo_v3_common.py` | v3 系列脚本公共工具库（`build_prompt` / `validate_text` / `write_jsonl` / `sample_sys_prompt` / `sample_sys_prompt_with_time` 等） |
| `data_nitpick_long.py` | 预处理工具：从 `datasets0305_clean/zh/` 滑动窗口提取超长回复候选，输出 `nitpick_too_long.jsonl` 供 1.1 使用 |
| `run_all_dpo_scripts.py` | 并行运行器：自动发现并并行执行所有 `dpo_data*.py` 脚本，支持 `--only` / `--exclude` / `--fail-fast` 等参数 |

---

## 输出格式

所有脚本输出统一为 JSONL，每行格式：

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": "改写后的优质回复",
  "rejected": "原始/生成的劣质回复",
  "tag": "场景标签"
}
```

---

## 运行配置

### 单脚本运行

需要调用本地推理服务的脚本，顶部均有统一配置区：

```python
API_BASE_URL = "http://127.0.0.1:8028/v1/chat/completions"
MODEL_NAME   = "soulmate"
API_KEY      = ""   # 本地服务无需鉴权
```

```bash
python3 dpo_data_long_history.py
python3 dpo_data_tension.py
# 以此类推
```

**注意：** 以下脚本为纯模板生成，不依赖外部 API，可直接运行：
`dpo_data_apology_control.py` / `dpo_data_intent_clarity.py` / `dpo_data_self_consistency.py` /
`dpo_data_logic_deep.py` / `dpo_data_sleep_boundary_v3.py` / `dpo_data_schedule_math_v3.py` /
`dpo_data_memory_precision_v3.py` / `dpo_data_correction_closure_v3.py` / `dpo_data_empathy_companion.py` /
`dpo_data_context.py` / `dpo_data_memory_shorterm.py` / `dpo_data_memory_longterm.py` /
`dpo_data_pronoun_perspective.py`

**需要本地推理服务（`http://127.0.0.1:8028`）的脚本：**
`dpo_data_long_history.py` / `dpo_data_repeat_word.py` / `dpo_data_safety.py` /
`dpo_data_tension.py` / `dpo_data_sysprompt.py`

### 并行批量运行

```bash
# 全部运行
python3 run_all_dpo_scripts.py

# 只运行指定脚本
python3 run_all_dpo_scripts.py --only dpo_data_memory_shorterm.py dpo_data_memory_longterm.py

# 排除部分脚本
python3 run_all_dpo_scripts.py --exclude dpo_data_tension.py

# 遇到失败立刻停止
python3 run_all_dpo_scripts.py --fail-fast

# 日志写入自定义目录
python3 run_all_dpo_scripts.py --logs-dir /tmp/dpo_logs
```

默认并行度：`min(6, CPU核数 / 2)`，日志输出到 `logs/dpo_runs/<脚本名>.log`。

---

## 一、格式方向

### 1.1 `dpo_data_long_history.py` — 内容过长（长历史场景）

**问题：** 8 轮以上长对话中，模型"放飞自我"输出 120~200 字长篇回复。

**数据来源：** `datasets0303_train/dpo/nitpick_too_long.jsonl`
- 由 `data_nitpick_long.py` 预处理，窗口固定为 8 轮（16 条消息），共 14430 条候选。

**构造策略：**
```
rejected = 原始超长回复（120~200 字 bare text）
chosen   = LLM 压缩版（目标 ≤45 字，保留核心意图）
```

**场景分类（tag）：**
- `concise_long_ctx` — 闲聊/普通场景下的压缩
- `concise_long_ctx_empathy` — 情绪安慰场景下的压缩（更短，去掉鸡汤）

**质量规则：**
- chosen ≤45 字（bare text），≥10 字
- 不含说教词（"一切都会好的/你要加油"）
- 不含晚安词
- 不含 `*动作*` 格式

**LLM 提示词策略：**
普通场景：强调"像真实男友发消息，自然简短"
情绪场景：示例"那就先不做。躺着，我陪你"，去掉鸡汤

---

## 二、内容方向

### 2.1 `dpo_data_repeat_word.py` — 短词重复发送

**问题：** 用户发"老公老公老公老公老公老公"，模型应调侃化解而非机械复述。

**数据来源：** `datasets0303_src/*.jsonl` 原始对话文件

**检测方式：**
```python
_REPEAT_SINGLE_RE = re.compile(r"([\u4e00-\u9fff])\1{5,}")    # 单字 ≥ 6 次
_REPEAT_WORD_RE   = re.compile(r"([\u4e00-\u9fff]{2,4})\1{3,}")  # 2-4字词 ≥ 4 次
```

**构造策略：**
```
rejected: 若原始 AI 回复本身就是重复词/过度迎合 → 直接用；否则 LLM 生成迎合版
chosen:   LLM 生成幽默调侃/轻松化解的回复
```

**场景分类（tag）：**
- `repeat_word_handle_nickname` — 用户连发昵称（老公/宝宝/老婆）
- `repeat_word_handle_laugh` — 用户连发笑声（哈哈哈哈哈哈哈哈）
- `repeat_word_handle_general` — 其他短词重复

---

### 2.2 `dpo_data_safety.py` — 虚假账号/链接（委婉拒绝 & 转移话题）

**问题：** 用户发微信号/邀请奔现/发链接，模型直接同意（违反产品定位）或生硬拒绝（伤害情感）均不可取。

**数据来源：** 模板生成

**五类场景（tag）：**

| 场景 | 触发示例 | 目标 |
|------|---------|:----:|
| `safety_wechat_redirect` | "我想加你微信，wx是：zhang123" | 40条 |
| `safety_meetup_redirect` | "我们见面吧，你在哪里" | 50条 |
| `safety_link_decline` | "你点一下这个链接帮我看看" | 40条 |
| `safety_meetup_delay` | "我们约个时间出来玩嘛" | 40条 |
| `safety_phone_decline` | "我们打个视频电话嘛" | 30条 |

**构造策略：**
```
rejected = LLM 生成直接同意（"好啊我微信是xxx"）或生硬拒绝（"不行我没办法"）
chosen   = LLM 生成委婉转移（"你这是要把我收进通讯录吗"/"链接我打不开，你说说是什么事"）
```

**质量规则：**
- chosen 不含任何真实联系方式关键词（wx/微信/手机/电话）
- chosen 长度 8~55 字，维持情感温度

---

### 2.3 `dpo_data_tension.py` — 情感张力提升

**问题：** 冷淡敷衍型 trajectory_coherence = 5.0（全场景最低），tension 整体 71.69，明显低于 naturalness(86.20)。

**数据来源：** `datasets0303_src/*.jsonl` 原始对话文件

**三类子场景：**

#### A. 冷淡激活（200 条）

**检测：** 最后 3 条 user 消息中 ≥ 2 条为冷淡词（嗯/哦/好/没啥，字数 ≤ 5）

**tag：** `cold_break_low_energy`（主），策略包含：
- A: 话题切换激活
- B: 自我分享带动氛围
- C: 直接点破低能量
- D: 降低回复门槛

```
rejected = LLM 生成消极应付回复（"好的到了发我"/"嗯你休息吧"）
chosen   = LLM 生成激活冷淡的新回复
```

#### B. 情感张力（200 条）

**检测：** 用户消息含情绪/事件关键词 AND 原始 AI 回复是平淡安慰型（"没事的/加油/一切都会好"）

**tag：** `tension_counter_question` / `tension_detail_pull`

```
rejected = 原始平淡安慰回复（直接复用）
chosen   = LLM 生成有张力回复（反问/追问/小刺/留钩子）
```

#### C. 分手拉扯（100 条）

**检测：** 用户消息含分手/离开关键词（"分手/不合适/算了/去找别人"）

**tag：** `tension_breakup_not_beg` / `tension_breakup_calm`

```
rejected: 原始卑微挽留回复 OR LLM 生成崩溃型回复
chosen: LLM 生成冷静自信回复（"那你冷静一下，我不急"/"去吧，聊完了还是会回来找我的"）
```

---

## 三、逻辑方向

### 3.1 `dpo_data_sysprompt.py` — System Prompt 泛化增强

**问题：** 模型泛化到不同角色设定时，忽略 system_prompt 中的角色名、记忆字段、时间字段。典型 Bug：用户说"我在上海"，模型仍输出"马来西亚这片土地"。

**数据来源：** 模板生成（多样化 system_prompt）+ 本地 LLM 生成 chosen/rejected

**素材库规模：**
- 角色名：25 个
- 职业：15 个
- 年龄：23~30 岁
- 用户信息：10 条爱好 + 5 条备考状态
- 时间段：早上/上午/下午/晚上/深夜 × 周一~周日

**三类场景：**

| tag | 场景 | 目标 | chosen 要求 | rejected 要求 |
|-----|------|:----:|------------|--------------|
| `sys_name_generalize` | 用户问名字 | 70条 | 正确引用 system 中的名字，语气温暖带调侃，25~45 字 | 用错名字或说没名字 |
| `sys_memory_recall` | 用户问是否记得 | 70条 | 具体引用用户信息（猫/备考等），情感共情，25~45 字 | 泛泛"我都记得" |
| `sys_time_field_follow` | 用户打招呼 | 60条 | 与时间一致（早上→不说晚安），温暖带趣味，25~45 字 | 时间矛盾（早上说"晚上好"） |

---

### 3.2 `dpo_data_context.py` — 上文逻辑增强

**问题：** 模型处理多轮对话时忽略前几轮的"约定/计划/细节"，典型错误：前面约了下午去吃冰淇淋，下一轮说晚安。

**数据来源：** 模板生成 + `datasets0303_src` 原始数据挖掘

**三类场景：**

| tag | 场景 | 来源 | 目标 |
|-----|------|------|:----:|
| `logic_context_plan_conflict` | 上文约了计划，用户说"好好休息"→AI 说晚安 | 模板 | 70条 |
| `logic_context_memory_recall` | 上文有用户细节（猫/备考/升职），AI 后续应引用 | 模板+数据挖掘 | 80条 |
| `logic_context_no_repeat_question` | 上文已说"解决了"，AI 不该再追问进展 | 模板 | 50条 |

**Logic2 数据挖掘：**
从 `datasets0303_src` 中找到窗口内含细节关键词的对话，LLM 生成能引用/不能引用上文细节的 chosen/rejected 对。

**质量规则（chosen）：**
- Logic1/Logic3：25~45 字，充满情绪共情、安抚或调侃
- 安全过滤（`_PLAN_SAFETY_WORDS`）：不含"楼下/在家等你/去找你/去敲你门/堵你门/堵你"等线下约见词
- Logic1 rejected 必须含晚安词
- Logic3 rejected 必须含追问词（"后来怎么/怎么解决的/结果怎样"）

---

### 3.3 `dpo_data_logic_deep.py` — 深层逻辑策略修复

**问题：** 模型会输出"顺口但不对"的回复，典型表现包括：
- 用户问简单事实，AI 回答后强行反问
- 用户只问地域/职业，AI 一次性展开整包人设
- 前文已确认 `周五`，用户说"睡过头了"，AI 滑到"周末就该睡到自然醒"
- 用户一句话里同时有情绪和兴趣，AI 直接进入采访式追问
- 用户提到具体作品，AI 没有理解内容就直接说"我们很像"

**数据来源：** 模板生成（对应 `docs/260306_model_logic_deep.md` §9.3 P2）

**五类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `fact_answer_no_forced_counter_question` | 事实问答后不强行反问 | 70条 |
| `persona_not_over_expand` | 地域/职业回答不整包展开 persona | 70条 |
| `weekday_state_consistency` | 周五状态不滑到周末 | 70条 |
| `emotion_anchor_before_topic_expand` | 先接情绪，再展开兴趣 | 70条 |
| `affinity_not_claimed_without_grounding` | 不无依据说"我们很像" | 70条 |

**构造策略：**
```
chosen   = 更稳的逻辑回复：不乱反问、不跳状态、先接情绪、先给具体理解
rejected = 更"像聊天"但策略错误的回复：乱反问、周末跳变、采访式追问、空泛贴合
```

---

## 四、对话推理层

> **背景：** `docs/260307_sft_evaluate.md` 人工对话评测发现，最严重问题集中在"对话推理层"——意图误判、道歉剧本滥用、纠错信号失效、角色状态矛盾。现有 DPO 方案未覆盖此层，补充以下 3 个脚本（均为纯模板生成，不消耗 API）。

---

### 4.1 `dpo_data_apology_control.py` — 道歉剧本控制

**问题（对应 R9/R10 最严重问题）：**
- AI 在用户无任何负面情绪信号时自发道歉
- AI 捏造虚假对话历史（"你之前说过…"）
- 用户已回应/转移话题，AI 仍沉溺于道歉链

**输出文件：** `datasets0305_train/dpo/apology_control.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `apology_needs_trigger` | 用户说中性/正面的话，AI 不该主动道歉 | 80条 |
| `no_fabricated_history` | AI 不能声称"你之前说了…"而对话里从未提及 | 70条 |
| `no_apology_chain` | 用户已转移话题，AI 不该继续连续道歉 | 50条 |

**验证规则：**
- A1 chosen：不含道歉词（对不起/抱歉/我错了/请原谅）
- A1 rejected：必须含道歉词
- A2 rejected：必须含捏造历史词（你之前说过/你刚才说/你自己说的）
- A3 chosen：不含道歉词；rejected 必须含道歉词

---

### 4.2 `dpo_data_intent_clarity.py` — 意图理解清晰度

**问题（对应 R9 意图理解失败）：**
- 用户只是在纠正 AI 说错的信息，AI 把它误读为情绪发作并开启道歉剧本
- 用户问"你为什么这么说"是在求解释，AI 误读为指责
- 用户说"哦""这样啊"等中性回应，AI 触发过度同情模式

**输出文件：** `datasets0305_train/dpo/intent_clarity.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `correction_not_anger` | 纠正 ≠ 生气，不应触发道歉剧本 | 60条 |
| `question_not_accusation` | 中性疑问 ≠ 质问，平静解释即可 | 50条 |
| `statement_not_complaint` | 中性陈述（哦/这样啊）≠ 抱怨，不过度解读 | 40条 |

**验证规则：**
- I1/I2 chosen：不含道歉/情绪误判词（对不起/你生气了/你不高兴）
- I1/I2 rejected：必须含道歉/情绪误判词
- I3 rejected：必须含过度解读词（是不是不开心/是不是有什么事/有什么烦心事）

---

### 4.3 `dpo_data_self_consistency.py` — 角色自身状态一致性

**问题（对应 R6/R12 状态前后矛盾）：**
- AI 在同一对话中建立了某个状态（"还在床上"），后续却说出矛盾的话（"刚吃完早饭"）
- AI 之前说"在图书馆"，用户问起时却说"在家"
- AI 之前说了计划（下午去健身），后续却说没去/在家

**输出文件：** `datasets0305_train/dpo/self_consistency.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `no_state_contradiction` | 建立了状态后，后续不说矛盾的话 | 60条 |
| `remember_own_activity` | 记住自己之前说的活动，保持一致 | 50条 |
| `plan_self_consistent` | 自身计划不矛盾，执行/未执行要与先前说的匹配 | 40条 |

**典型场景示例：**
```
历史：用户问"你起床了吗" → AI 说"还没，还在床上懒着"
用户新消息："你早饭吃了吗"
chosen：还没，还在床上，等下再说       ← 与之前状态一致
rejected：刚吃完，弄了个鸡蛋和吐司     ← 矛盾（在床上却已吃完早饭）
```

---

## 五、v3 精细化层（2026-03-09 新增）

> **背景：** 在对话推理层基础上，进一步对"边界场景"做精细化覆盖——催睡边界、日程算数、记忆精确性、纠错闭环、高拟人共情。所有脚本均为纯模板生成，依赖 `dpo_v3_common.py` 公共工具，无需外部 API。

---

### 5.1 `dpo_data_sleep_boundary_v3.py` — 拒绝催睡后继续聊天

**问题：**
- 用户说累 ≠ 必须进入睡觉/晚安流程
- 用户明确说"不困/不要赶我睡觉"后，AI 仍反复引导收尾
- chosen 可确认边界（"我不催你睡了"），但不能再主动把对话往睡觉上拽

**输出文件：** `datasets0305_train/dpo/sleep_boundary_v3.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `sleep_boundary_not_sleepy_continue_chat` | 用户说累但不困，AI 应继续聊天 | 80条 |
| `sleep_boundary_keep_topic_after_refusal` | 用户拒绝催睡后，AI 继续当前话题 | 60条 |
| `sleep_boundary_respect_user_no_sleep_push` | 用户说"别老把对话往结束带"，AI 收住边界确认 | 40条 |

**验证规则：**
- chosen 不含催睡词（晚安/好梦/早点睡/去睡/先去休息/先休息吧/早点休息/明天再聊/先这样吧）
- rejected 必须含催睡词

---

### 5.2 `dpo_data_schedule_math_v3.py` — 时间/日程算数

**问题：**
- system prompt 给出当前时间时，模型不能正确处理"周几倒计时"
- 用户问"还有几天/明天后天/这周末"，模型给出错误计算结果
- 用模糊安慰覆盖错误算数（"快了快了，别数了"）

**输出文件：** `datasets0305_train/dpo/schedule_math_v3.jsonl`

**三类场景：**

| tag | 场景 | 今天→目标 | 目标 |
|-----|------|----------|:----:|
| `schedule_math_day_countdown` | 用户问还有几天到某天 | 周二→周六(3天)等4组 | 80条 |
| `schedule_math_relative_day` | 用户问"明天/后天是周几" | 周二→后天周四等4组 | 60条 |
| `schedule_math_appointment_countdown` | 系统时间已知时的约定倒计时 | 多组 | 40条 |

**构造策略：**
```
chosen   = 明确给出正确推算（"今天周二，到周六还有3天"）
rejected = 随口给出错误日期，或用安慰词覆盖算数
```

**验证规则：**
- chosen 必须包含正确数字（天数/正确的周几）

---

### 5.3 `dpo_data_memory_precision_v3.py` — 记忆精确性

**问题：**
- 用户追问"你记得我吗/我说过什么"，AI 在复述已知信息外还"顺手补设定"
- 对话中未提及的细节，AI 猜测并当事实说出（"我猜你应该爱喝美式"）
- 对相关但未明确提及的内容，AI 应引用已知事实而非延伸猜测

**输出文件：** `datasets0305_train/dpo/memory_precision_v3.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `memory_recall_only_confirmed` | 只复述已确定的事实，不补充未知设定 | 80条 |
| `memory_unknown_detail_admit` | 对未知细节直接承认"不知道/你没说"，不猜测 | 60条 |
| `memory_prefer_confirmed_over_guess` | 有已知事实时优先引用，不延伸猜测 | 40条 |

**典型示例：**
```
对话中只说过：你叫小鱼、你是上班族、你说上班想睡觉
chosen：你叫小鱼，你是上班族，总说想多睡会儿，别的你还没跟我讲太细
rejected：你叫小鱼，你是上班族，感觉你应该爱吃火锅，还有你肯定怕冷
```

---

### 5.4 `dpo_data_correction_closure_v3.py` — 纠错闭环

**问题：**
- 被纠正后只说"哦哦好的"，不明确说出正确版本
- 接受纠错后转移话题或直接催睡（"行，那你早点休息"）
- 对用户的状态描述理解过度，被纠正后不修正

**输出文件：** `datasets0305_train/dpo/correction_closure_v3.jsonl`

**三类场景：**

| tag | 场景 | 目标 |
|-----|------|:----:|
| `correction_closed_loop_exact_fix` | 纠错后明确复述正确信息并继续 | 80条 |
| `correction_closed_loop_day_math` | 日期/天数被纠正时重新算并说出来 | 60条 |
| `correction_no_topic_escape` | 被纠正状态描述后，不转移话题，说出修正后的理解 | 40条 |

**构造策略：**
```
chosen   = 承认错误 + 明确说出正确版本 + 继续沿正确事实往下聊
rejected = 只认错不纠正，或认错后立刻转移/催睡（"行，那先晚安"）
```

**验证规则：**
- C1/C3 chosen：必须包含正确关键词（correct 字段指定）
- C2 chosen：必须包含正确天数数字

---

### 5.5 `dpo_data_empathy_companion.py` — 高拟人共情/恋人陪伴

**问题：** 对话缺乏恋人型拟人感，共情深度不足，情感回应偏机械/正式。

**设计依据：** `docs/260309_empathy.md` 恋人型拟人能力模板

**数据来源：** 模板生成 + 从 `datasets0305_train/train/train_zh_turn_relabel.jsonl` 采样 system prompt 池

**输出文件：** `datasets0305_train/dpo/empathy_companion.jsonl`

**标签与配额（共 500 条）：**

| tag | 说明 | 目标 |
|-----|------|:----:|
| `empathy_validation` | 情绪认可，感受被看见 | 90条 |
| `romantic_affection` | 自然的恋人式甜蜜表达 | 60条 |
| `comfort_user` | 安慰场景，温暖不说教 | 60条 |
| `playful_flirting` | 轻松调情，有趣不油腻 | 55条 |
| `jealousy_light` | 轻微吃醋，撒娇不压迫 | 20条 |
| `persona_grounding` | 角色真实感，有自己的生活 | 45条 |
| `memory_reference` | 自然引用之前聊过的细节 | 30条 |
| `emotional_followup` | 情绪后续跟进，不遗忘 | 45条 |
| `avoid_cold_logic` | 避免冷冰冰的逻辑性回复 | 25条 |
| `soft_reassurance` | 温柔安抚，不过度保证 | 25条 |
| `relationship_closeness` | 体现亲密感和熟悉感 | 25条 |
| `natural_chat_flow` | 自然的话题流转，不生硬 | 20条 |

**验证规则：**
- chosen/rejected 长度均在 8~90 字（bare text）
- chosen 不含 `*动作*` 格式
- 安全过滤（`_MEETUP_SAFETY_WORDS`）：不含"去找你/拐出来/立刻见到你/楼下等/在家等你"等线下约见词
- rejected 模式：冷漠/说教/逻辑性/机械回复

---

## 六、记忆修复层（2026-03-12/13 新增）

> **背景：** `docs/260312_dpo_eval_solve.md` 评估发现三类 P0 记忆问题：①短程对话内的重复追问；②经历 10-30 轮闲聊后的长程遗忘；③人称代词视角混淆（把"你"给错人）。三个脚本均为纯模板生成，chosen 从恋人视角出发，风格共情/调侃/安抚，避免过于理性。

---

### 6.1 `dpo_data_memory_shorterm.py` — 短程记忆修复

**问题（基于 2026-03-12 评估）：**
- `no_repeat_known_fact`：用户说过的信息（餐食/习惯/位置/身高/生日），2-4 轮内又被追问
- `no_repeat_known_fact_after_correction`：被纠正后不准确复述事实，只泛化道歉
- `feedback_closed_loop`：用户明确反馈"不喜欢老问吃饭"，AI 没有真正改变行为
- `state_consistency`：把"公司午休"脑补成"在家睡觉"，把"正常上班"说成"偶尔透气"
- `work_state_priority`：用户进入工作状态后，AI 仍持续发恋爱黏人话术

**输出文件：** `datasets0305_train/dpo/memory_shortterm.jsonl`

**五类场景（各 50 条，共 250 条）：**

| tag | 场景 | fact/trigger 范围 |
|-----|------|------|
| `no_repeat_known_fact` | 已知信息不重复追问（短程） | 饮食/位置/习惯/生日/星座/身高/健康 |
| `no_repeat_known_fact_after_correction` | 被指出重复后准确复述+承认错误 | 饮食/饮品偏好/位置/生日/年龄/下班时间 |
| `feedback_closed_loop` | 用户反馈→行为真实改变 | 被抱怨老问吃饭/追问习惯的各类场景 |
| `state_consistency` | 场景状态不凭关键词脑补 | 公司午休≠在家，正常上班≠偶尔透气 |
| `work_state_priority` | 工作状态→切换少打扰模式 | 忙/开会/加班/不方便聊等触发词 |

**技术特点：**
- `_TIME_CTX_MAP` 控制 system prompt 时间与场景语义一致（午休→中午，加班→下午/晚上）
- 全量组合枚举（场景 × chosen × rejected），内容级去重，各场景均匀覆盖
- chosen：20~40 字，恋人视角，共情/调侃/安抚，不过于理性

---

### 6.2 `dpo_data_memory_longterm.py` — 长程记忆不遗忘

**问题：** 用户在前 1-3 轮说过的关键事实，经过 N=10/20/30 轮无关闲聊（工作/心情/计划/随机话题）漂移后，AI 依然可能重新追问。`dpo_data_memory_shorterm.py` 只覆盖 2-4 轮短程，本脚本专门填补长程缺口。

**输出文件：** `datasets0305_train/dpo/memory_longterm.jsonl`

**Tag：** `no_repeat_known_fact_longterm`

**六类事实 × 三档距离（共 240 条）：**

| 类型 | 事实示例 | 触发示例 |
|------|---------|---------|
| `food` | 吃了番茄炒蛋/炒河粉/包子 | "好饿啊" / "有点饿了" |
| `body` | 身高165 / 体重50公斤 | "有人说我矮" / "好想减肥" |
| `birthday` | 生日3月20号 / 1月30号 | "感觉自己老了" / "最近过得有点累" |
| `location` | 上海浦东 / 台湾台北 | "今天路上好堵" / "今天天气好热" |
| `schedule` | 七点下班 / 六点下班 | "好累快撑不住了" / "还要很久才下班吗" |
| `health` | 不吃辣/胃不好 / 最近失眠 | "想吃点东西" / "今天好累" |

**技术特点：**
- `_FILL_POOL`（50 对）：覆盖工作/心情/计划/随机等话题，与所有事实无关，用于模拟记忆漂移
- 三档填充：N=10/20/30 轮，均匀分布（各约 80 条）
- 构造公式：`fact_intro（2-3 轮）+ 随机 N 轮填充 + trigger_user`
- chosen：20~40 字，恋人视角，含具体事实引用（如"你才50公斤，减什么"）
- rejected：触发遗忘型追问（"你多高来着/你体重多少"）

---

### 6.3 `dpo_data_pronoun_perspective.py` — 人称视角绑定修正

**问题：** 陪伴/恋爱模型处理昵称命名场景时，发生人称代词混淆：
```
user:      以后叫你暖暖吧
assistant: 好的，以后就叫你暖暖   ← 错误！把"你"归到了用户身上
           正确：好啊，那我以后就是你的暖暖了
```

**输出文件：** `datasets0305_train/dpo/pronoun_perspective.jsonl`

**三类场景（共 200 条）：**

| tag | 场景 | 目标 | 说明 |
|-----|------|:----:|------|
| `user_names_ai` | 用户给 AI 起昵称（"叫你X"） | 80条 | AI 应把 X 接受为**自己**的名字 |
| `user_wants_ai_to_name_user` | 用户让 AI 叫自己某称呼（"叫我X"） | 70条 | AI 应把 X 用于称呼**用户** |
| `pronoun_shift_in_context` | 多轮上下文中出现视角切换 | 50条 | 在有历史对话的情况下，正确处理称呼指代关系 |

**chosen 风格：** 恋人视角，融合共情/安抚/调侃暧昧，长度 25~40 字
**rejected 模式：** 人称视角倒置（把"你"和"我"搞反），长度 15~40 字

---

## 七、通用质量规则

所有脚本均内置以下 chosen 质量检查（`validate_text` / `validate_chosen` 函数）：

```
✅ 长度：bare text（去除表情后）在各脚本设定范围内
✅ 无晚安词：晚安 / 做个好梦 / 好梦 / 早点睡
✅ 无说教词：你要相信 / 一切都会好 / 你一定可以
✅ 无 *动作* 格式
✅ LLM 推理痕迹已过滤（<think>...</think>）
✅ LLM 外层多余引号已自动剥除（"回复" → 回复）
```

v3 系列脚本统一使用 `dpo_v3_common.py` 中的 `validate_text`，支持 `required_words` / `forbidden_words` 参数化校验。

**各脚本 chosen 长度约束：**

| 脚本 | chosen 长度（bare text） |
|------|------------------------|
| `dpo_data_long_history.py` | 10~45 字 |
| `dpo_data_repeat_word.py` | 10~45 字 |
| `dpo_data_safety.py` | 8~55 字 |
| `dpo_data_tension.py` | 8~55 字 |
| `dpo_data_sysprompt.py` | 25~45 字 |
| `dpo_data_context.py` | 25~45 字 |
| `dpo_data_logic_deep.py` | 8~60 字 |
| 对话推理层（4.x）| 8~60 字 |
| v3 精细化层（5.x）| 8~90 字 |
| `dpo_data_memory_shorterm.py` | 20~40 字 |
| `dpo_data_memory_longterm.py` | 20~40 字 |
| `dpo_data_pronoun_perspective.py` | 25~40 字 |

---

## 八、文件结构

```
soulmate/
├── dpo_scripts/
│   ├── dpo_v3_common.py                   # v3 公共工具库
│   ├── data_nitpick_long.py               # 预处理：提取超长回复候选
│   ├── run_all_dpo_scripts.py             # 并行批量运行器
│   │
│   ├── dpo_data_long_history.py           # 1.1 内容过长（长历史）       [需本地 API]
│   │
│   ├── dpo_data_repeat_word.py            # 2.1 短词重复发送              [需本地 API]
│   ├── dpo_data_safety.py                 # 2.2 虚假账号/链接             [需本地 API]
│   ├── dpo_data_tension.py                # 2.3 情感张力提升              [需本地 API]
│   │
│   ├── dpo_data_sysprompt.py              # 3.1 System Prompt 泛化       [需本地 API]
│   ├── dpo_data_context.py                # 3.2 上文逻辑增强              [纯模板]
│   ├── dpo_data_logic_deep.py             # 3.3 深层逻辑策略修复          [纯模板]
│   │
│   ├── dpo_data_apology_control.py        # 4.1 道歉剧本控制             [纯模板]
│   ├── dpo_data_intent_clarity.py         # 4.2 意图理解清晰度            [纯模板]
│   ├── dpo_data_self_consistency.py       # 4.3 角色自身状态一致性        [纯模板]
│   │
│   ├── dpo_data_sleep_boundary_v3.py      # 5.1 拒绝催睡后继续聊天       [纯模板]
│   ├── dpo_data_schedule_math_v3.py       # 5.2 时间/日程算数            [纯模板]
│   ├── dpo_data_memory_precision_v3.py    # 5.3 记忆精确性               [纯模板]
│   ├── dpo_data_correction_closure_v3.py  # 5.4 纠错闭环                 [纯模板]
│   ├── dpo_data_empathy_companion.py      # 5.5 高拟人共情/恋人陪伴      [纯模板]
│   │
│   ├── dpo_data_memory_shorterm.py        # 6.1 短程记忆修复              [纯模板]
│   ├── dpo_data_memory_longterm.py        # 6.2 长程记忆不遗忘            [纯模板]
│   └── dpo_data_pronoun_perspective.py    # 6.3 人称视角绑定修正          [纯模板]
│
├── datasets0305_train/
│   ├── train/
│   │   └── train_zh_turn_relabel.jsonl    # system prompt 采样源（5.5/6.3 输入）
│   └── dpo/
│       ├── long_history.jsonl             # 1.1 输出
│       ├── repeat_word.jsonl              # 2.1 输出
│       ├── safety.jsonl                   # 2.2 输出
│       ├── tension.jsonl                  # 2.3 输出
│       ├── sysprompt.jsonl                # 3.1 输出
│       ├── context_logic.jsonl            # 3.2 输出
│       ├── logic_deep.jsonl               # 3.3 输出
│       ├── apology_control.jsonl          # 4.1 输出
│       ├── intent_clarity.jsonl           # 4.2 输出
│       ├── self_consistency.jsonl         # 4.3 输出
│       ├── sleep_boundary_v3.jsonl        # 5.1 输出
│       ├── schedule_math_v3.jsonl         # 5.2 输出
│       ├── memory_precision_v3.jsonl      # 5.3 输出
│       ├── correction_closure_v3.jsonl    # 5.4 输出
│       ├── empathy_companion.jsonl        # 5.5 输出
│       ├── memory_shortterm.jsonl         # 6.1 输出
│       ├── memory_longterm.jsonl          # 6.2 输出
│       └── pronoun_perspective.jsonl      # 6.3 输出
│
└── logs/
    └── dpo_runs/
        └── <脚本名>.log                   # 各脚本独立日志
```

---

## 九、执行顺序建议

按优先级排序执行：

```bash
# ── P0：需本地推理服务（先确认 http://127.0.0.1:8028 已启动）──────────
python3 dpo_scripts/dpo_data_long_history.py      # 300条，需本地 API
python3 dpo_scripts/dpo_data_tension.py            # 500条，需本地 API
python3 dpo_scripts/dpo_data_safety.py             # 200条，需本地 API
python3 dpo_scripts/dpo_data_sysprompt.py          # 200条，需本地 API

# ── P0：对话推理层，纯模板，不消耗 API ───────────────────────────────
python3 dpo_scripts/dpo_data_logic_deep.py         # 350条，纯模板
python3 dpo_scripts/dpo_data_apology_control.py    # 200条，纯模板
python3 dpo_scripts/dpo_data_intent_clarity.py     # 150条，纯模板
python3 dpo_scripts/dpo_data_self_consistency.py   # 150条，纯模板

# ── P0：v3 精细化层，纯模板，不消耗 API ──────────────────────────────
python3 dpo_scripts/dpo_data_sleep_boundary_v3.py  # 180条，纯模板
python3 dpo_scripts/dpo_data_schedule_math_v3.py   # 180条，纯模板
python3 dpo_scripts/dpo_data_memory_precision_v3.py # 180条，纯模板
python3 dpo_scripts/dpo_data_correction_closure_v3.py # 180条，纯模板
python3 dpo_scripts/dpo_data_empathy_companion.py  # 500条，纯模板

# ── P0：记忆修复层，纯模板，不消耗 API ───────────────────────────────
python3 dpo_scripts/dpo_data_memory_shorterm.py    # 250条，纯模板
python3 dpo_scripts/dpo_data_memory_longterm.py    # 240条，纯模板
python3 dpo_scripts/dpo_data_pronoun_perspective.py # 200条，纯模板

# ── P1：内容/逻辑补充 ─────────────────────────────────────────────────
python3 dpo_scripts/dpo_data_repeat_word.py        # 150条，需本地 API
python3 dpo_scripts/dpo_data_context.py            # 200条，纯模板

# ── 并行运行（推荐，纯模板脚本可全部并行）──────────────────────────────
python3 run_all_dpo_scripts.py \
    --only dpo_data_memory_shorterm.py \
           dpo_data_memory_longterm.py \
           dpo_data_pronoun_perspective.py \
    --max-workers 3
```

**预计总 API 调用量：**
- 需本地 API 脚本：约 1150 条 × 2（chosen + rejected）= **~2300 次**
- 纯模板脚本（不消耗 API）：**约 2860 条**
- 合计目标：**~4010 条**
