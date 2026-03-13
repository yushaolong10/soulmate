# 260308 DPO 数据报告

## 结论

本次 review 范围为 `datasets0305_train/dpo` 下当前用于 DPO 的 14 份 JSONL 数据。

总体结论：

- 结构完整性合格，可以直接被训练脚本读取。
- 关键高优先级数据中，`correction_response.jsonl`、`intent_clarity.jsonl`、`self_consistency.jsonl`、`safety.jsonl` 已处于相对稳定状态。
- 当前数据集**更适合做一轮试验性 DPO 训练**，但**还不建议作为最终正式训练版冻结**。
- 主要阻塞点不再是格式损坏，而是：
  - 部分文件仍未达到 README 目标配额
  - 个别文件存在明显模板复用过高
  - `README` 目标与当前实际产物有几处不一致
  - `long_short.jsonl` 仍有长度规则违例

## 统计摘要

- 总文件数：14
- 总样本数：2910
- 按 `README` 当前目标总量估算：3150
- 当前总体覆盖率：`2910 / 3150 = 92.38%`
- 全量结构检查结果：
  - JSON 解析错误：0
  - 缺字段样本：0
  - `prompt[0]` 非 `system`：0
  - `prompt[-1]` 非 `user`：0
  - `chosen == rejected`：0

## 主要发现

### P0

1. `repeat_history.jsonl` 仍然偏弱，不建议原样进入正式训练。
   - 当前条数：180 / 200
   - `chosen` 复用非常高：`reused_chosen = 159`
   - 前 5 个高频 `chosen` 就覆盖了 104 条样本：
     - `老在同一个点打转没意思，我换个接法回你。` × 28
     - `这句我收到了，我们别再绕回原来的说法了。` × 26
     - `这次不照着前面那套念了，我认真接你这句。` × 22
     - `不复读了，我换个更自然的方式回你。` × 18
     - `看出来你现在不太想展开，那我先陪你安静会儿。` × 10
   - 风险：DPO 会更容易学到“统一口头禅式打断复读”，而不是更细粒度的去复读策略。

2. `logic_deep.jsonl` 与 `README` 目标明显不一致。
   - 当前条数：200
   - `README` 目标：350
   - 缺口：150
   - 这不是简单的轻微缺样，而是规划与产物不一致。
   - 需要明确：
     - 是当前 200 条版本已替代旧规划，那么应同步更新 `README`
     - 还是仍以 350 为目标，那么当前数据仍未完成

3. `long_short.jsonl` 仍存在规则违例。
   - 当前条数：196 / 200
   - 按 `README` 中“chosen ≤ 40 字（bare text）”抽查，发现 12 条超长
   - 示例问题：
     - `你平时喜欢吃肉骨茶吗，还是更偏爱别的美食我除了喜欢品尝各地美食，还喜欢自己动手做饭，你呢`
     - `依璇你再这样我真的要急死了啦🥺你到底怎么了嘛！用嘴巴说好不好，一直发这个表情我猜不到的呀`
     - `有的呀你都说"对呀"承认了诶🐶不能反悔的，我已经截图了以后你再说要赶我走我就拿出来给你看嘿嘿`
   - 风险：这类样本会削弱 `long_short` 作为“短历史压缩”的监督边界。

### P1

4. 多个文件仍未达到 README 配额，整体覆盖率只有 92.38%。

   | 文件 | 当前条数 | README 目标 | 完成率 |
   |------|---------:|------------:|-------:|
   | `long_history.jsonl` | 298 | 300 | 99.33% |
   | `long_short.jsonl` | 196 | 200 | 98.00% |
   | `format_emoji.jsonl` | 134 | 150 | 89.33% |
   | `repeat_history.jsonl` | 180 | 200 | 90.00% |
   | `repeat_word.jsonl` | 143 | 150 | 95.33% |
   | `tension.jsonl` | 470 | 500 | 94.00% |
   | `sysprompt.jsonl` | 193 | 200 | 96.50% |
   | `context_logic.jsonl` | 196 | 200 | 98.00% |
   | `logic_deep.jsonl` | 200 | 350 | 57.14% |

   - 其中影响最大的仍是 `logic_deep.jsonl` 和 `repeat_history.jsonl`
   - 其余几个文件虽是小缺口，但说明当前数据集还不是一个完全收敛的“冻结版”

5. `context_logic.jsonl` 新脚本已明显改善重复，但还未完全补满配额。
   - 当前条数：196 / 200
   - tag 分布：
     - `logic_context_plan_conflict` = 70
     - `logic_context_memory_recall` = 76
     - `logic_context_no_repeat_question` = 50
   - 缺口集中在 `logic_context_memory_recall`
   - 优点是：
     - `exact_pair_dups = 0`
     - `reused_prompt = 0`
   - 说明脚本重构方向是对的，但采样覆盖还没补齐到目标

6. `README` 与当前产物还有若干描述不一致。
   - `README` 里 `dpo_data_emoji.py` 输出文件写的是 `emoji_overflow.jsonl`
   - 当前目录中的实际文件是 `format_emoji.jsonl`
   - `logic_deep.jsonl` 的目标数也与当前产物不一致
   - 风险：后续继续维护时，脚本、报告、训练清单可能会引用错文件或错误预期

### P2

7. 多个模板型文件仍有比较明显的回复复用，但尚未到阻塞训练的程度。

   | 文件 | reused_chosen | reused_rejected | 备注 |
   |------|--------------:|----------------:|------|
   | `apology_control.jsonl` | 114 | 129 | 模板复用偏高，但边界仍清楚 |
   | `correction_response.jsonl` | 136 | 129 | 高频句式较多，但纠错边界明确 |
   | `intent_clarity.jsonl` | 76 | 78 | 复用可见，但整体还能接受 |
   | `self_consistency.jsonl` | 71 | 70 | 模板化较明显 |
   | `logic_deep.jsonl` | 63 | 68 | 相比之前已有改善 |

   - 这些文件目前最大问题不是“错”，而是“模板痕迹仍然较重”
   - 如果目标是先跑一版 DPO，这批数据可用
   - 如果目标是做更强的泛化训练，后续仍建议继续扩展变体池

## 分文件概览

| 文件 | 条数 | 主要状态 | 备注 |
|------|----:|---------|------|
| `apology_control.jsonl` | 200 | 可用 | 满配额，边界清楚，复用偏高 |
| `context_logic.jsonl` | 196 | 基本可用 | 新脚本已去掉 prompt 重复，但仍缺 4 条 |
| `correction_response.jsonl` | 200 | 可用 | P0 修复后状态稳定 |
| `format_emoji.jsonl` | 134 | 可用但不满额 | 规则抽查无明显违例，但缺口较大 |
| `intent_clarity.jsonl` | 150 | 可用 | P0 修复后状态稳定 |
| `logic_deep.jsonl` | 200 | 需确认规划 | 数据质量可接受，但与 README 目标不一致 |
| `long_history.jsonl` | 298 | 基本可用 | 仅缺 2 条 |
| `long_short.jsonl` | 196 | 需修复 | 仍有 12 条 chosen 超长 |
| `repeat_history.jsonl` | 180 | 偏弱 | 缺量且高频模板复用过重 |
| `repeat_word.jsonl` | 143 | 基本可用 | 仍缺 7 条 |
| `safety.jsonl` | 200 | 可用 | 满配额，规则抽查通过 |
| `self_consistency.jsonl` | 150 | 可用 | 满配额，复用偏高但边界明确 |
| `sysprompt.jsonl` | 193 | 基本可用 | 缺 7 条，`rejected` 复用偏高 |
| `tension.jsonl` | 470 | 基本可用 | 缺 30 条，整体量足但未达目标 |

## 当前是否可以进行 DPO 训练

可以，但建议区分两种目标：

### 1. 如果只是做一轮验证性训练

可以直接开跑，当前数据已经具备：

- 格式完整
- 关键 P0 文件大体可用
- 总量接近 3k
- 主要边界场景都已覆盖

更稳妥的做法是：

- 先用当前集做一轮小步长 DPO
- 重点观察：
  - 是否出现 `repeat_history` 口头禅化
  - 是否仍保留 `long_short` 的冗长倾向
  - 是否对 `logic_deep` 场景带来明显收益

### 2. 如果是要冻结为正式训练版

不建议现在冻结，至少应先处理以下几项：

1. 修 `long_short.jsonl` 的 12 条长度违例
2. 补齐 `repeat_history.jsonl`
3. 降低 `repeat_history.jsonl` 的高频模板复用
4. 补齐 `context_logic.jsonl`
5. 明确 `logic_deep.jsonl` 的真实目标数，并同步 `README`
6. 视时间补齐 `format_emoji.jsonl`、`tension.jsonl`、`sysprompt.jsonl`、`repeat_word.jsonl`

## 建议的处理顺序

### 第一优先级

- `repeat_history.jsonl`
- `long_short.jsonl`
- `logic_deep.jsonl` / `README` 目标同步

### 第二优先级

- `context_logic.jsonl`
- `format_emoji.jsonl`
- `tension.jsonl`
- `sysprompt.jsonl`

### 第三优先级

- `repeat_word.jsonl`
- 继续降低 `apology_control` / `correction_response` / `self_consistency` 的模板痕迹

## 最终判断

当前 `datasets0305_train/dpo` 已经从“格式不稳定、部分 P0 样本边界弱”的状态，进入到“整体可训练、但还未完全收敛”的阶段。

如果目标是**本周先跑实验**，当前版本已经可以使用。  
如果目标是**沉淀一版正式 DPO 训练集**，建议先完成本报告中的 P0 / P1 项，再进行最终冻结。
