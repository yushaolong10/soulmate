# 训练知识总结 (Knowledge Notes)

## 1. 8-bit 量化 vs bfloat16

### 1.1 为什么使用 8-bit 量化

| 特性 | bfloat16 | 8-bit 量化 |
|------|----------|-----------|
| 显存占用 | ~30GB (14B 模型) | ~15GB (14B 模型) |
| 数值稳定性 | 可能出现 NaN/Inf | 更稳定 |
| 训练速度 | 较快 | 稍慢 |
| CPU offload | `device_map="auto"` 可能触发 | 通常不需要 |

### 1.2 bfloat16 出现 NaN 的原因

1. **Qwen3 模型的 logits 范围很大**：`[-129, 14976]`
2. **LoRA scaling 放大效应**：`scaling = alpha / r`，如果 `alpha=32, r=16`，则 `scaling=2.0`
3. **bfloat16 精度有限**：大数值 + LoRA scaling 容易溢出

### 1.3 解决方案

```python
# 推荐配置：scaling = 1.0
LORA_R = 16
LORA_ALPHA = 16  # alpha = r，scaling = 1.0
```

---

## 2. device_map 配置

### 2.1 单卡 vs 多卡

```python
# 单卡（推荐用于 8-bit）
device_map = "cuda:0"

# 多卡自动分配（可能触发 CPU offload）
device_map = "auto"
```

### 2.2 8-bit 量化的多卡限制

**重要**：8-bit/4-bit 量化模型与 accelerator 的多卡调度有冲突！

```
ValueError: You can't train a model that has been loaded in 8-bit or 4-bit precision 
on a different device than the one you're training on.
```

**解决方案**：
- 方案 A：两个模型放在同一个 GPU 上
- 方案 B：使用 `precompute_ref_log_probs=True` 避免加载 ref model
- 方案 C：不使用量化，改用 bfloat16（需要更多显存）

---

## 3. DPO 训练架构

### 3.1 标准架构（需要两个模型）

```
Policy Model: Base + SFT LoRA (冻结) + DPO LoRA (训练)
Ref Model:    Base + SFT LoRA (冻结)
```

### 3.2 precompute_ref_log_probs 架构（只需一个模型）

```python
dpo_config = DPOConfig(
    precompute_ref_log_probs=True,  # 预计算 ref log probs
    ...
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 不需要单独加载
    ...
)
```

**工作流程**：
1. 训练开始前，用当前模型计算所有样本的 reference log probs
2. 训练时只更新 policy model，使用预计算的 ref log probs

**优点**：显存减半
**缺点**：需要额外的预处理时间

---

## 4. PEFT 多 Adapter 使用

### 4.1 添加新 Adapter

```python
from peft import PeftModel, LoraConfig

# 加载 SFT LoRA（冻结）
model = PeftModel.from_pretrained(base_model, sft_lora_dir, is_trainable=False)

# 定义新的 DPO LoRA 配置
dpo_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# 添加新 adapter
model.add_adapter("dpo", dpo_lora_config)

# 设置活动 adapter
model.set_adapter("dpo")
```

### 4.2 推理时加载多个 Adapter

```python
# 先加载 SFT
model = PeftModel.from_pretrained(base_model, sft_lora_dir)

# 再加载 DPO
model.load_adapter(dpo_lora_dir, adapter_name="dpo")

# 激活 DPO adapter
model.set_adapter("dpo")
```

---

## 5. 常见问题排查

### 5.1 训练时出现 NaN

**检查项**：
1. LoRA scaling 是否过大（建议 <= 1.0）
2. 学习率是否过高（建议 1e-5 ~ 3e-5）
3. 数据中是否有异常样本（过短、重复等）
4. 是否启用了梯度裁剪（`max_grad_norm=1.0`）

**诊断工具**：
```python
# 添加 NaN 检测回调
class NaNDetectorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and math.isnan(logs.get("loss", 0)):
            control.should_training_stop = True
        return control
```

### 5.2 推理时出现 NaN

**可能原因**：
1. 训练环境和推理环境不一致
2. 8-bit 训练的模型用 bfloat16 加载

**解决方案**：
```python
# 训练时使用 8-bit，推理时也使用 8-bit
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False,
)
```

### 5.3 显存不足 (OOM)

**解决方案**：
1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 启用 `gradient_checkpointing`
4. 使用 8-bit 或 4-bit 量化
5. 减小 `max_seq_len`

---

## 6. 训练流程总结

### 6.1 SFT 训练

```bash
# 使用 8-bit 量化
CUDA_VISIBLE_DEVICES=0 python sft_gpu_8bit.py
```

**输出**：`qwen_lora_adapter_xxx/`

### 6.2 DPO 训练

```bash
# 单卡（8-bit）
CUDA_VISIBLE_DEVICES=0 \
SFT_LORA_DIR=qwen_lora_adapter_xxx \
python dpo_gpu_8bit.py
```

**输出**：`qwen_lora_dpo_xxx/`

### 6.3 推理部署

```bash
# 8-bit 推理服务
CUDA_VISIBLE_DEVICES=0 \
SFT_LORA_DIR=qwen_lora_adapter_xxx \
DPO_LORA_DIR=qwen_lora_dpo_xxx \
python server_gpu_8bit.py
```

---

## 7. 脚本对照表

| 功能 | bfloat16 版本 | 8-bit 版本 |
|------|---------------|-----------|
| SFT 训练 | `sft_gpu.py` | `sft_gpu_8bit.py` |
| DPO 训练 | `dpo_gpu.py` | `dpo_gpu_8bit.py` |
| 推理服务 | `server_gpu.py` | `server_gpu_8bit.py` |
| 模型检查 | - | `tools/check_sft_model.py` |

**注意**：8-bit 训练的模型应该用 8-bit 加载，bfloat16 训练的模型用 bfloat16 加载，不要混用！

---

## 8. 关键参数参考

### 8.1 SFT 参数

```python
LORA_R = 16
LORA_ALPHA = 16      # scaling = 1.0
LR = 3e-5
PER_DEVICE_BS = 3
GRAD_ACCUM = 16
EPOCHS = 3
```

### 8.2 DPO 参数

```python
DPO_LORA_R = 8
DPO_LORA_ALPHA = 16  # scaling = 2.0
LR = 2e-5
PER_DEVICE_BS = 1
GRAD_ACCUM = 8
EPOCHS = 2
BETA = 0.05          # DPO 温度参数
```

### 8.3 推理参数

```python
temperature = 0.7
top_p = 0.7
repetition_penalty = 1.1
max_tokens = 1024
```
