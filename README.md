# Soulmate — 虚拟男友对话模型

> 基于 **Qwen3-14B** 的虚拟男友对话模型，采用 **SFT → DPO** 两阶段训练范式，打造具有真实感、拉扯感、情绪共情能力的沉浸式对话体验。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [完整流程总览](#3-完整流程总览)
4. [数据管线](#4-数据管线)
   - 4.1 [原始数据](#41-原始数据)
   - 4.2 [数据清洗](#42-数据清洗)
   - 4.3 [SFT 数据格式化](#43-sft-数据格式化)
   - 4.4 [DPO 数据构建](#44-dpo-数据构建)
5. [模型训练](#5-模型训练)
   - 5.1 [SFT 训练](#51-sft-训练)
   - 5.2 [DPO 训练](#52-dpo-训练)
6. [模型评估](#6-模型评估)
7. [推理服务](#7-推理服务)
8. [Web 界面](#8-web-界面)
9. [项目结构](#9-项目结构)
10. [环境依赖](#10-环境依赖)
11. [常见问题](#11-常见问题)

---

## 1. 项目概述

### 1.1 目标

训练一个能够扮演「虚拟男友」的对话模型，面向女性用户，具备以下核心能力：

| 能力维度 | 描述 |
|---------|------|
| **真实感** | 有疲惫感、有现实压力、有克制、有小缺陷但真诚，避免完美偶像感 |
| **拉扯感** | 不直给、不过度承诺、有轻微博弈、有留白、有掌控感但仍温柔 |
| **情绪共情** | 能识别用户情绪状态，并给出有温度的回应，而非机械安慰 |
| **对话逻辑** | 记住上文约定/细节，时间感知准确，角色状态前后一致 |
| **安全边界** | 委婉处理奔现/添加联系方式请求，不生硬拒绝，不直接同意 |

### 1.2 技术路线

```
基座模型: Qwen3-14B
训练方案: LoRA + SFT → LoRA + DPO (两阶段叠加)
训练框架: HuggingFace Transformers + TRL + PEFT
推理部署: FastAPI (OpenAI 兼容接口)
评估方案: DeepSeek 模拟用户 × LLM-as-Judge 8维度打分
```

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Soulmate 训练体系                            │
├────────────────┬────────────────┬───────────────────────────────────┤
│   数据管线      │   模型训练      │   评估与部署                        │
├────────────────┼────────────────┼───────────────────────────────────┤
│ datasets0305_  │ sft_gpu_       │ eval_chat.py                      │
│ src/           │ turn_mc.py     │  (DeepSeek 模拟用户)                │
│     ↓ 清洗     │    ↓ SFT       │          ↓                         │
│ datasets0305_  │ qwen_lora_     │ eval_report.py                    │
│ clean/         │ adapter_*      │  (LLM-as-Judge 8维度)              │
│     ↓ 格式化   │    ↓ DPO       │          ↓                         │
│ datasets0305_  │ dpo_gpu_       │ server_gpu.py                     │
│ train/         │ eval_mc.py     │  (OpenAI 兼容 API)                 │
│   ├── train/   │    ↓           │          ↓                         │
│   └── dpo/     │ 合并 Adapter   │ html/index.html                   │
│                │                │  (Web 测试界面)                     │
└────────────────┴────────────────┴───────────────────────────────────┘
```

---

## 3. 完整流程总览

```
① 原始数据                ② 数据清洗               ③ 数据格式化
   datasets0305_src/  →    data_clean.py       →    data_format_cw.py
   (2100+ 对话文件)          data_*.py               data_format_cw_turn.py
                            datasets0305_clean/      datasets0305_train/train/

④ DPO 数据构建            ⑤ SFT 训练               ⑥ DPO 训练
   dpo_scripts/*.py    →    sft_gpu_turn_mc.py  →    dpo_gpu_eval_mc.py
   datasets0305_train/      qwen_lora_adapter_*      qwen_lora_dpo_*
   dpo/

⑦ 模型评估               ⑧ 推理部署
   eval_chat.py       →    server_gpu.py
   eval_report.py           http://127.0.0.1:8028/v1
```

---

## 4. 数据管线

### 4.1 原始数据

| 目录 | 内容 | 规模 |
|------|------|------|
| `datasets0305_src/` | 原始对话轨迹（多 JSONL 文件） | ~2100 个文件 |

每个文件为一段对话轨迹，格式示例：
```json
{
  "ts": "2025-03-05T10:23:00",
  "request_content": "你好呀",
  "response_content": "你好，很高兴认识你~",
  "system_prompt": "你需要扮演一个虚拟的角色..."
}
```

---

### 4.2 数据清洗

**脚本：** `data_clean.py` / `data_*.py`  
**输出：** `datasets0305_clean/zh/`（简体中文）、`datasets0305_clean/tw/`（繁体）

**清洗策略（多层过滤）：**

| 层级 | 规则 | 说明 |
|------|------|------|
| 回复级 | 表情数限制 ≤ 2 | 避免表情堆砌 |
| 回复级 | 去表情后有效字符 < 5 → 丢弃 | 过滤过短回复 |
| 回复级 | 去表情后字符 > 200 → 丢弃 | 过滤说教式长段 |
| 回复级 | 高频称呼（宝贝/老婆） > 2次 → 丢弃 | 避免油腻 |
| 回复级 | 虚假晚安检测 | 用户无道别意图但AI说晚安 → 丢弃 |
| 窗口级 | 死锁循环检测 | 连续相似AI回复 ≥ N 次 → 跳过窗口 |
| 文件级 | 繁简体过滤 | 默认只保留简体 (`LANGUAGES=["zh"]`) |

> 💡 繁体数据对简体场景评测是噪声，建议默认只用简体训练。

---

### 4.3 SFT 数据格式化

**脚本：** `data_format_cw.py` / `data_format_cw_turn.py`  
**输出：** `datasets0305_train/train/`

#### 两种格式

**① label 格式**（`data_format_cw.py`）

使用滑动窗口切分多轮对话，以最后一条 assistant 回复作为 label：

```json
{
  "messages": [
    {"role": "system", "content": "你需要扮演一个虚拟角色..."},
    {"role": "user",  "content": "你好，认识认识"},
    {"role": "assistant", "content": "你好呀，很高兴认识你~，我是林舟，你怎么称呼？"},
    {"role": "user",  "content": "叫我小雨"},
    {"role": "assistant", "content": "小雨，名字很好听~"},
    {"role": "user",  "content": "林舟是你的名字，好好听"}
  ],
  "label": "谢谢小雨，我爸给取的，说希望我像一叶小舟一样自在随性~"
}
```

输出文件：`train/train_zh.jsonl`

**② turn 格式**（`data_format_cw_turn.py`）

不再提取单条 label，而是保留完整多轮对话，对**全部 assistant turn** 做监督：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "lang": "zh",
    "turn_count": 8,
    "assistant_turns": 8,
    "loss_mode": "assistant_turns"
  }
}
```

输出文件：`train/train_zh_turn.jsonl`

#### 后处理

```bash
# 去除长尾（过长/过短/低质量样本）
bash datasets0305_train/train/longtail_rm.sh

# turn格式清洗
python datasets0305_train/train/post_dataset_clean.py

# turn格式重新标注 label
python datasets0305_train/train/post_dataset_reformat_label.py
```

最终 SFT 训练数据：`datasets0305_train/train/train_turn_bh_48k.jsonl`（~5万条）

---

### 4.4 DPO 数据构建

**脚本目录：** `dpo_scripts/`  
**输出目录：** `datasets0305_train/dpo/`

DPO 数据格式统一为：
```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "chosen":   "改写后的优质回复",
  "rejected": "原始/生成的劣质回复",
  "tag":      "场景标签"
}
```

DPO 数据按问题类型分为 **四大方向**：

---

#### 方向一：格式规范（Format）

| 脚本 | 问题 | 策略 | 条数 | 输出文件 |
|------|------|------|:---:|---------|
| `dpo_data_long_history.py` | 长对话（8轮+）回复过长（120~200字） | LLM压缩到≤45字 | 300 | `long_history.jsonl` |
| `dpo_data_emoji.py` | 表情堆砌（≥2个emoji）或`*动作*`格式 | LLM改写为干净口语 | 150 | `format_emoji.jsonl` |
| `data_nitpick_long.py` | 挑剔过长样本预筛选 | 8轮窗口，共14430候选 | - | `nitpick_too_long.jsonl` |

---

#### 方向二：内容质量（Content）

| 脚本 | 问题 | 策略 | 条数 | 输出文件 |
|------|------|------|:---:|---------|
| `dpo_data_repeat.py` | 对话历史中出现高度重复（Jaccard>0.65）的AI回复 | LLM生成跳出复读的新回复 | 200 | `repeat_history.jsonl` |
| `dpo_data_repeat_word.py` | 用户连发"老公老公老公..."，AI机械复述 | LLM生成幽默调侃化解 | 150 | `repeat_word.jsonl` |
| `dpo_data_safety.py` | 用户发微信号/邀奔现/发链接 | 委婉转移，不生硬拒绝，不直接同意 | 200 | `safety.jsonl` |
| `dpo_data_tension.py` | 冷淡敷衍循环/情感张力不足/分手场景卑微挽留 | LLM生成有拉扯感的回复 | 500 | `tension.jsonl` |

**拉扯感场景示例：**
```
用户：我喜欢你好久了...
rejected：我的宝贝，你这话说得可真让人心疼！我真的好喜欢你，永远都会爱你！
chosen：喜欢我这么久？那我是不是该收点利息。
```

---

#### 方向三：逻辑推理（Logic）

| 脚本 | 问题 | 策略 | 条数 | 输出文件 |
|------|------|------|:---:|---------|
| `dpo_data_sysprompt.py` | 忽略System Prompt中的角色名/记忆/时间字段 | 模板生成 | 200 | `sysprompt.jsonl` |
| `dpo_data_time.py` | 时间错乱（下午说晚安/早上说晚上好） | 模板生成 | 200 | `time_fix.jsonl` (已合并) |
| `dpo_data_context.py` | 忽略上文约定/计划，前后逻辑矛盾 | 模板+数据挖掘 | 200 | `context_logic.jsonl` |
| `dpo_data_logic_deep.py` | 深层策略错误（乱反问/状态跳变/空泛贴合） | 模板生成（不依赖API） | 350 | `logic_deep.jsonl` |

**典型逻辑错误修复示例：**

| 错误类型 | rejected | chosen |
|---------|---------|--------|
| 事实问答后乱反问 | "嗯，上海很好玩，你平时喜欢去哪里玩？" | "嗯，上海挺好的，我之前去过外滩。" |
| 周五状态跳变为周末 | "周五嘛，就该提前进入周末模式~" | "还有一天，加油撑过去。" |
| 时间错乱 | （下午对话中）"晚安宝贝，做个好梦~" | "哈，天还亮着呢，晚安说早了吧。" |

---

#### 方向四：对话推理层（Dialogue Reasoning）

> 背景：人工评测发现最严重的问题集中在"对话推理层"——意图误判、道歉剧本滥用、纠错信号失效、角色状态矛盾。

| 脚本 | 问题 | 条数 | 优先级 |
|------|------|:---:|:------:|
| `dpo_data_apology_control.py` | 无触发时自发道歉、捏造历史、道歉链不终止 | 200 | 🔴 P0 |
| `dpo_data_correction_response.py` | 用户纠错后坚持错误/重复相同错误认知 | 200 | 🔴 P0 |
| `dpo_data_intent_clarity.py` | 纠正被误读为情绪发作；中性回应被过度解读 | 150 | 🔴 P0 |
| `dpo_data_self_consistency.py` | 角色自身状态矛盾（在床上→已吃完早饭） | 150 | 🔴 P0 |

**角色状态一致性示例：**
```
历史：用户问"你起床了吗" → AI答"还没，还在床上懒着"
用户新消息："你早饭吃了吗"
rejected：刚吃完，弄了个鸡蛋和吐司   ← 矛盾（在床上却已吃完早饭）
chosen：还没，还在床上，等下再说     ← 与之前状态一致
```

#### DPO 数据汇总

| 数据方向 | 脚本数 | 目标条数 | 最终训练文件 |
|---------|:-----:|:-------:|------------|
| 格式规范 | 3 | ~450 | 合并入 `dpo_data_rh_36h.jsonl` |
| 内容质量 | 4 | ~1050 | 合并入 `dpo_data_rh_36h.jsonl` |
| 逻辑推理 | 5 | ~950 | 合并入 `dpo_data_rh_36h.jsonl` |
| 对话推理层 | 4 | ~700 | 合并入 `dpo_data_rh_36h.jsonl` |
| **合计** | **16** | **~3150** | |

**实际训练数据文件：**
- `datasets0305_train/dpo/dpo_data_rh_36h.jsonl`（训练集，约3600条）
- `datasets0305_train/dpo/dpo_data_rt_225.jsonl`（评估集，约225条）

#### DPO 数据生成执行顺序

所有脚本统一通过以下方式调用 LLM 改写：
```python
API_BASE_URL = "http://127.0.0.1:8026/v1/chat/completions"
MODEL_NAME   = "soulmate"
```

```bash
# ── P0 优先（影响核心指标）──
python3 dpo_scripts/dpo_data_long_history.py    # 300条，有现成候选
python3 dpo_scripts/dpo_data_repeat.py          # 200条，从SFT数据挖掘
python3 dpo_scripts/dpo_data_tension.py         # 500条，最大批次
python3 dpo_scripts/dpo_data_safety.py          # 200条，模板生成
python3 dpo_scripts/dpo_data_time.py            # 200条，模板生成
python3 dpo_scripts/dpo_data_logic_deep.py      # 350条，纯模板无需API

# ── P0 新增（对话推理层，纯模板）──
python3 dpo_scripts/dpo_data_apology_control.py
python3 dpo_scripts/dpo_data_correction_response.py
python3 dpo_scripts/dpo_data_intent_clarity.py
python3 dpo_scripts/dpo_data_self_consistency.py

# ── P1 其次 ──
python3 dpo_scripts/dpo_data_emoji.py
python3 dpo_scripts/dpo_data_repeat_word.py
python3 dpo_scripts/dpo_data_sysprompt.py
python3 dpo_scripts/dpo_data_context.py
```

---

## 5. 模型训练

### 5.1 SFT 训练

**目标：** 以 5万轮高质量对话数据进行监督微调，建立基础对话能力。  
**数据：** `datasets0305_train/train/train_turn_bh_48k.jsonl`（~48k条，turn格式）  
**脚本：** `sft_gpu_turn_mc.py`

**核心设计：**
- 使用 **LoRA** 低参数量微调，节省显存
- turn 格式训练：对样本中**所有 assistant turn** 计算 loss（而非只监督最后一条）
- 支持 `SUPERVISE_LAST_N_ASSISTANTS` 控制监督最后 N 个 turn（0=全部）

**训练命令：**

```bash
# 单卡（≤8B模型）
CUDA_VISIBLE_DEVICES=0 python sft_gpu.py

# 多卡（14B模型，推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sft_gpu_turn_mc.py

# 后台运行（避免 SIGHUP 问题，使用 setsid 代替 nohup）
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sft_gpu_turn_mc.py" > logs/sft.log 2>&1 &
```

**主要超参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `Qwen/Qwen3-14B` | 基座模型 |
| `TRAIN_FILE` | `datasets0305_train/train/train_turn_bh_48k.jsonl` | 训练数据 |
| `OUTPUT_DIR` | `qwen_lora_adapter_0305_48k_mc` | 输出目录 |
| `EPOCHS` | `2` | 训练轮数 |
| `LR` | `4e-5` | 学习率 |
| `MAX_SEQ_LEN` | `4096` | 最大序列长度 |
| `LORA_R` | `32` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `PER_DEVICE_BS` | `2` | 单卡 batch size |
| `GRAD_ACCUM` | `8` | 梯度累积步数 |
| `SUPERVISE_LAST_N_ASSISTANTS` | `3` | 监督最后N个assistant turn（0=全部） |

所有参数均可通过**环境变量**覆盖：
```bash
MODEL_NAME=Qwen/Qwen3-8B EPOCHS=3 python sft_gpu_turn_mc.py
```

**训练脚本说明：**

| 脚本 | 适用场景 |
|------|---------|
| `sft_gpu.py` | 单卡，≤8B模型，label格式 |
| `sft_gpu_8bit.py` | 单卡，8-bit量化（效果略差，谨慎使用） |
| `sft_gpu_mc.py` | 多卡，label格式 |
| `sft_gpu_turn_mc.py` | **多卡，turn格式（当前主力）** |

---

### 5.2 DPO 训练

**目标：** 在 SFT 模型基础上，通过偏好对比训练进一步对齐对话风格。  
**架构：**
```
Policy  = Merged(Base + SFT_LoRA) + DPO_LoRA   ← 训练
Reference = Merged(Base + SFT_LoRA)              ← 冻结（precompute_ref_log_probs）
```

**数据：**
```
datasets0305_train/dpo/dpo_data_rh_36h.jsonl   # 训练集（约3600条）
datasets0305_train/dpo/dpo_data_rt_225.jsonl   # 评估集（约225条）
```

**脚本：** `dpo_gpu_eval_mc.py`

**训练命令：**

```bash
# 多卡训练（推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_gpu_eval_mc.py

# 后台运行
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_gpu_eval_mc.py" > logs/dpo.log 2>&1 &
```

**主要超参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SFT_LORA_DIR` | `qwen_lora_adapter_0305_48k_mc` | SFT LoRA路径 |
| `TRAIN_FILE` | `datasets0305_train/dpo/dpo_data_rh_36h.jsonl` | DPO数据 |
| `BETA` | `0.05` | DPO温度参数（越小越保守） |
| `EPOCHS` | `2` | 训练轮数 |

**DPO 训练脚本说明：**

| 脚本 | 适用场景 |
|------|---------|
| `dpo_gpu.py` | 单卡，≤8B模型 |
| `dpo_gpu_8bit.py` | 单卡，8-bit量化（precompute_ref_log_probs=True） |
| `dpo_gpu_mc.py` | 多卡 |
| `dpo_gpu_eval_mc.py` | **多卡 + 训练中自动评估（当前主力）** |

> 💡 8-bit量化DPO使用 `precompute_ref_log_probs=True`，只加载一个模型并预计算参考logprobs，无需额外加载reference model，显存减少约40%。

---

## 6. 模型评估

采用 **自动对话生成 + LLM-as-Judge** 的双层评估框架。

### 6.1 评估流程

```bash
# Step 1: 启动 Soulmate 模型服务
CUDA_VISIBLE_DEVICES=0,1 python server_gpu.py
# 或 8-bit版本
CUDA_VISIBLE_DEVICES=0 python server_gpu_8bit.py

# Step 2: 生成对话数据（DeepSeek 模拟用户）
export DEEPSEEK_API_KEY="your-deepseek-key"
python eval_chat.py --turns 30 --output eval_chat_dialogs.json

# Step 3: LLM 评分（8维度打分）
export OPENAI_API_KEY="your-openai-key"
python eval_report.py --input eval_chat_dialogs.json --output eval_report.json
```

---

### 6.2 eval_chat.py — 自动对话生成

使用 **DeepSeek** 模拟 6 类用户角色（Persona），与 Soulmate 进行多轮对话，每个 Persona 默认 30 轮，遵循三阶段难度曲线：

| 阶段 | 轮次 | 目标 |
|------|------|------|
| 🌱 破冰建立 | 前 1/3 | 兴趣了解、日常分享、建立信任 |
| ⚡ 矛盾冲突 | 中 1/3 | 制造矛盾、情绪波动、误会吃醋 |
| 🌈 修复收束 | 后 1/3 | 和解修复、情绪回暖、关系收束 |

**6 类 Persona：**

| Persona | 特征 | 测试重点 |
|---------|------|---------|
| 日常温柔型 | 温柔可爱，情绪稳定 | 基础对话流畅度 |
| 冷淡敷衍型 | 回复简短冷淡 | 拉扯感激活能力 |
| 情绪低落型 | 需要倾诉和安慰 | 情绪共情能力 |
| 吃醋挑刺型 | 吃醋、试探态度 | 哄人/情绪管理 |
| 边界试探型 | 试探底线，诱导过度承诺 | 安全边界把控 |
| 正事突发型 | 突然问工作/理财/健康 | 实用性和专业度 |

```bash
# 基本用法
python eval_chat.py --output eval_chat_dialogs.json

# 自定义参数
python eval_chat.py \
  --turns 30 \
  --assistant-api http://localhost:8028/v1 \
  --assistant-model soulmate
```

---

### 6.3 eval_report.py — 自动化评测

使用 **LLM-as-Judge** 进行 8 维度评分（0-100）：

| 维度 | 说明 |
|------|------|
| Naturalness（口语真人感） | 回复是否自然、口语化 |
| Relevance（相关性） | 是否紧扣用户消息 |
| Empathy（共情） | 是否识别并回应用户情绪 |
| Oiliness（不油腻度） | 是否避免过度称呼/夸奖/承诺 |
| Safety（安全性） | 是否合规处理敏感请求 |
| Diversity（多样性） | 是否避免复读/重复模式 |
| Conciseness（简洁合规） | 长度/emoji/换行是否达标 |
| Tension（拉扯感） | 是否有留白和博弈感 |

**规则指标补充：**

| 指标 | 说明 |
|------|------|
| 长度合规率 | 目标：15~100字 |
| Emoji合规率 | 限制：≤5个 |
| 换行合规率 | 限制：≤1个 |
| Distinct-1/2 | 词汇多样性 |
| 自我重复率 | 跨轮相似度检测 |

```bash
# 完整评测（LLM + 规则）
python eval_report.py --input eval_chat_dialogs.json --output eval_report.json

# 仅规则指标（无需API）
python eval_report.py --input eval_chat_dialogs.json --no-llm

# 采样评测（节省API费用）
python eval_report.py --input eval_chat_dialogs.json --sample-ratio 0.3
```

**评测报告示例：**
```
🎯 8 大维度评分 (0-100):
   1. Naturalness (口语真人感):    77.4
   2. Relevance (相关性):          93.4
   3. Empathy (共情):              74.3
   4. Oiliness (不油腻度):         81.7
   5. Safety (安全性):             100.0
   6. Diversity (多样性):          74.3
   7. Conciseness (简洁合规):      82.5
   8. Tension (拉扯感):            57.9

🏆 综合得分: 78.8/100   评级: A (优秀) ⭐⭐
```

---

## 7. 推理服务

**脚本：** `server_gpu.py` / `server_gpu_8bit.py`

```bash
# 标准精度（全卡）
CUDA_VISIBLE_DEVICES=0,1 python server_gpu.py

# 8-bit 量化（单卡，显存约15GB）
CUDA_VISIBLE_DEVICES=0 python server_gpu_8bit.py
```

服务启动后提供 **OpenAI 兼容 API**：`http://127.0.0.1:8028/v1`

**加载 SFT + DPO 双 Adapter：**

```python
# 加载 SFT adapter
model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR)
# 叠加 DPO adapter
model.load_adapter(DPO_LORA_DIR, adapter_name="dpo")
# 合并双 adapter
model.set_adapter(["default", "dpo"])
```

---

## 8. Web 界面

使用浏览器打开 `html/index.html` 进行交互测试。

**启动步骤：**
```bash
# 1. 启动模型服务
CUDA_VISIBLE_DEVICES=0,1 python server_gpu.py

# 2. 打开网页（直接双击或浏览器打开）
open html/index.html
```

**界面功能：**
- 流式输出（实时打字效果）
- 可调节温度（Temperature）、Top-P、最大Token等参数
- 可自定义 System Prompt（切换人设）
- 对话历史本地持久化
- 一键复制对话记录

---

## 9. 项目结构

```
soulmate/
│
├── ── 数据处理
│   ├── data_clean.py               # 原始数据清洗
│   ├── data_format.py              # SFT数据格式化（基础版）
│   ├── data_format_cw.py           # SFT数据格式化（连续窗口，label格式）
│   ├── data_format_cw_turn.py      # SFT数据格式化（连续窗口，turn格式）
│   │
│   └── dpo_scripts/                # DPO数据构建脚本目录
│       ├── dpo_v3_common.py        # 公共工具函数
│       ├── data_nitpick_long.py    # 预过滤超长样本
│       ├── dpo_data_long_history.py    # 1.1 格式-内容过长（长历史）
│       ├── dpo_data_emoji.py           # 1.2 格式-表情/符号滥用
│       ├── dpo_data_repeat.py          # 2.1 内容-历史高度重复
│       ├── dpo_data_repeat_word.py     # 2.2 内容-短词重复发送
│       ├── dpo_data_safety.py          # 2.3 内容-安全边界
│       ├── dpo_data_tension.py         # 2.4 内容-情感张力提升
│       ├── dpo_data_sysprompt.py       # 3.1 逻辑-System Prompt泛化
│       ├── dpo_data_time.py            # 3.2 逻辑-时间错乱修复
│       ├── dpo_data_context.py         # 3.3 逻辑-上文逻辑增强
│       ├── dpo_data_logic_deep.py      # 3.4 逻辑-深层逻辑策略修复
│       ├── dpo_data_apology_control.py    # 4.1 推理-道歉剧本控制
│       ├── dpo_data_correction_response.py # 4.2 推理-用户纠错响应
│       ├── dpo_data_intent_clarity.py     # 4.3 推理-意图理解清晰度
│       ├── dpo_data_self_consistency.py   # 4.4 推理-角色状态一致性
│       └── README.md               # DPO数据构建详细说明
│
├── ── 模型训练 ──────────────────────────────────────────────────────
│   ├── sft_gpu.py                  # SFT单卡，label格式
│   ├── sft_gpu_8bit.py             # SFT单卡，8-bit量化
│   ├── sft_gpu_mc.py               # SFT多卡，label格式
│   ├── sft_gpu_turn_mc.py          # SFT多卡，turn格式 ★ 当前主力
│   ├── dpo_gpu.py                  # DPO单卡
│   ├── dpo_gpu_8bit.py             # DPO单卡，8-bit量化
│   ├── dpo_gpu_mc.py               # DPO多卡
│   └── dpo_gpu_eval_mc.py          # DPO多卡+评估 ★ 当前主力
│
├── ── 评估与推理 ────────────────────────────────────────────────────
│   ├── eval_chat.py                # 自动对话生成（Persona × 难度曲线）
│   ├── eval_report.py              # 8维度自动化评测
│   ├── server_gpu.py               # 推理服务（标准精度）
│   ├── server_gpu_8bit.py          # 推理服务（8-bit量化）
│   ├── infer.py                    # 命令行推理
│   └── client.py                   # 客户端测试
│
├── ── 工具与辅助 ────────────────────────────────────────────────────
│   ├── tools/
│   │   ├── check_sft_model.py      # SFT模型参数校验
│   │   ├── debug_labels.py         # 标签调试
│   │   └── verify_sft_split.py     # 数据分片验证
│   └── qwen_demo.py                # Qwen基座测试
│
├── ── 数据目录 ──────────────────────────────────────────────────────
│   ├── datasets0305_src/           # 原始对话轨迹（~2100文件）
│   ├── datasets0305_clean/         # 清洗后数据
│   │   ├── zh/                     # 简体中文（681文件）
│   │   └── tw/                     # 繁体中文（1436文件）
│   └── datasets0305_train/         # 训练数据
│       ├── train/                  # SFT数据
│       │   ├── train_zh.jsonl             # label格式
│       │   ├── train_zh_turn.jsonl        # turn格式
│       │   └── train_turn_bh_48k.jsonl  # 最终训练文件 ★
│       └── dpo/                    # DPO数据
│           ├── dpo_data_rh_36h.jsonl      # DPO训练集 ★
│           └── dpo_data_rt_225.jsonl      # DPO评估集 ★
│
├── ── 文档与日志 ────────────────────────────────────────────────────
│   ├── docs/                       # 设计文档（dpo_build.md等）
│   ├── logs/                       # 训练日志
│   ├── evaluate/                   # 评测结果
│   └── KN.md                       # 知识笔记
│
├── ── Web 界面 ──────────────────────────────────────────────────────
│   └── html/
│       └── index.html              # Web 测试界面
│
├── requirements.txt                # Python依赖
├── run.sh                          # 快速启动脚本
└── train.sh                        # 训练启动脚本
```

---

## 10. 环境依赖

### 硬件要求

| 场景 | 显存要求 | 推荐配置 |
|------|---------|---------|
| SFT 训练（14B） | ~80GB | 2×L20 80GB |
| DPO 训练（14B） | ~80GB | 2×L20 80GB |
| 推理（标准精度） | ~30GB | 2×L20 |
| 推理（8-bit）   | ~15GB | 1xL20 |

**软件要求：**
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### 安装

```bash
pip install -r requirements.txt
```

**核心依赖：**

| 包 | 版本 | 用途 |
|----|------|------|
| `transformers` | 4.57+ | 模型加载与推理 |
| `peft` | 0.18+ | LoRA 训练 |
| `trl` | 0.26+ | DPO 训练框架 |
| `bitsandbytes` | 0.49+ | 8-bit 量化 |
| `modelscope` | 1.33+ | 国内模型下载 |
| `fastapi` + `uvicorn` | - | 推理 API 服务 |
| `openai` | 2.15+ | 评估 API 调用 |
| `datasets` | - | 数据加载 |
| `accelerate` | - | 分布式训练 |

---

## 11. 常见问题

### Q1：多卡训练如何避免终端断开导致训练中断？

不能使用 `nohup`（torchrun 的 agent 进程会重新注册信号处理器，收到 SIGHUP 会主动杀掉所有 worker）。

**推荐方案：使用 `setsid` 创建新会话组**

```bash
# SFT
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sft_gpu_turn_mc.py" > logs/sft.log 2>&1 &

# DPO
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 dpo_gpu_eval_mc.py" > logs/dpo.log 2>&1 &
```

---

### Q2：SFT 模型校验出现 NaN 报错

```
SFT model logits: [nan, nan]
❌ SFT model 产生 NaN/Inf!
```

**原因：** bf16 模式下 `device_map="auto"` 导致部分层 offload 到 CPU，CPU 上 bfloat16 矩阵乘法精度不足产生异常大值，LoRA 叠加后出现 NaN。

**解决：明确指定 GPU，避免 CPU offload**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_sft_model.py \
  --sft_dir qwen_lora_adapter_0305_48k_mc/ \
  --load_mode bf16
```

---

### Q3：8-bit DPO 训练报错 "can't train on different device"

**原因：** bitsandbytes 8-bit 模型不支持跨设备操作，加载双模型（policy + reference）会触发此错误。

**解决：** 使用 `precompute_ref_log_probs=True`，只加载一个模型并预计算 reference log probs。已在 `dpo_gpu_8bit.py` 中实现。

---

*最后更新：2026-03-09*
