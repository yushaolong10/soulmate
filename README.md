# 🤖 Soulmate - 情感聊天助手

基于 **Qwen3-1.7B/14B** 模型，使用 **LoRA (Low-Rank Adaptation)** 技术微调的中文聊天助手。该模型专门针对情感陪伴场景进行训练，能够以自然、口语化的方式进行对话交流。

---

## ✨ 特性

- 🎯 **轻量级微调**：采用 LoRA 技术，仅训练少量参数，节省计算资源
- 💬 **自然对话**：生成简短、流畅、口语化的回复
- 🚀 **OpenAI 兼容 API**：提供标准的 `/v1/chat/completions` 接口
- 🍎 **Apple Silicon 支持**：原生支持 MPS 加速 (Mac M系列芯片)
- 📦 **Response-Only 训练**：仅对助手回复部分计算损失，训练更精准

---

## 📁 项目结构

```
soulmate/
├── finetune.py          # LoRA 微调脚本
├── server.py            # FastAPI 服务端 (OpenAI 兼容 API)
├── client.py            # 客户端测试脚本
├── infer.py             # 本地推理脚本
├── run.sh               # 快速启动服务脚本
├── requirements.txt     # Python 依赖
├── datasets/            # 训练数据目录
│   ├── train_0119_s.jsonl   # 小数据集 (1400 样本)
│   ├── train_0119_m.jsonl   # 中数据集 (4000 样本)
│   └── train_0119_l.jsonl   # 大数据集 (8000 样本)
├── qwen_lora_adapter_0119_sw/  # 小数据集训练的 LoRA 适配器
└── qwen_lora_adapter_0119_mw/  # 中等数据集训练的 LoRA 适配器
└── qwen_lora_adapter_0119_lw/  # 大数据集训练的 LoRA 适配器
```

---

## 🛠️ 安装

### 1. 克隆项目并创建虚拟环境

```bash
git clone <your-repo-url>
cd soulmate

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖列表：**
- `transformers >= 4.57`
- `torch >= 2.9`
- `peft >= 0.18`
- `trl >= 0.26`
- `modelscope >= 1.33`
- `fastapi`
- `uvicorn`
- `openai`

---

## 📊 数据格式

训练数据使用 **JSONL** 格式，每行一个对话样本：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个活泼开朗的男生，和想要进一步追求的女生进行对话..."},
    {"role": "user", "content": "我今天工作好累，刚下班"},
    {"role": "assistant", "content": "辛苦啦！下班路上注意安全，想吃点什么好吃的犒劳自己？"}
  ]
}
```

---

## 🚀 快速开始

### 1. 启动 API 服务

```bash
# 使用 run.sh 快速启动
bash run.sh

# 或手动配置启动
BASE_MODEL=Qwen/Qwen3-1.7B \
LORA_DIR=./qwen_lora_adapter_0115_x \
DEVICE=mps \
DTYPE=float16 \
SERVED_MODEL_NAME=soulmate \
uvicorn server:app --host 0.0.0.0 --port 8026
```

### 2. 调用 API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8026/v1", api_key="dummy")

resp = client.chat.completions.create(
    model="soulmate",
    messages=[
        {"role": "system", "content": "你是一个贴心的男生，能够及时回复女生消息"},
        {"role": "user", "content": "你住哪？"}
    ],
    temperature=0.7,
    top_p=0.7,
    max_tokens=1024,
)

print(resp.choices[0].message.content)
```

### 3. 本地推理 (无需启动服务)

```bash
python infer.py
```

---

## 🔌 API 接口

### 获取模型列表

```bash
GET /v1/models
```

### 聊天补全

```bash
POST /v1/chat/completions
```

**请求体：**

```json
{
  "model": "soulmate",
  "messages": [
    {"role": "system", "content": "系统提示词"},
    {"role": "user", "content": "用户消息"}
  ],
  "temperature": 0.7,
  "top_p": 0.7,
  "max_tokens": 1024
}
```

**响应：**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "soulmate",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "助手回复内容"
      },
      "finish_reason": "stop"
    }
  ]
}
```