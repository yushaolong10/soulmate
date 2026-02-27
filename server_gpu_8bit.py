# server_gpu_8bit.py
# 8-bit 量化 GPU 推理服务器
#
# 特性：
#   - 8-bit 量化加载，显存占用更少 (~15GB)
#   - 与 sft_gpu_8bit.py / dpo_gpu_8bit.py 训练的模型兼容
#   - CUDA GPU 加速推理
#   - KV Cache 优化
#   - 真正的流式生成（逐 token 输出）
#   - 支持 SFT LoRA + DPO LoRA 双层 adapter
#
# 运行：
#   # 仅 SFT LoRA
#   CUDA_VISIBLE_DEVICES=0 SFT_LORA_DIR=./qwen_lora_adapter_0226_1h python server_gpu_8bit.py
#
#   # SFT LoRA + DPO LoRA
#   CUDA_VISIBLE_DEVICES=0 SFT_LORA_DIR=./qwen_lora_adapter_0226_1h DPO_LORA_DIR=./qwen_lora_dpo_0226_8bit python server_gpu_8bit.py
#
#   # 指定端口
#   CUDA_VISIBLE_DEVICES=0 uvicorn server_gpu_8bit:app --host 0.0.0.0 --port 8000

import os
import re
import time
import uuid
import json
import asyncio
from typing import Any, Dict, List, Optional, Iterator
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from contextlib import asynccontextmanager

# -----------------------
# Config
# -----------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-14B")

# LoRA adapter 配置
# SFT_LORA_DIR: SFT 微调后的 LoRA adapter (必需)
# DPO_LORA_DIR: DPO 训练后的 LoRA adapter (可选，在 SFT 基础上加载)
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "./qwen_lora_adapter_0226_1w_8bit")
DPO_LORA_DIR = os.environ.get("DPO_LORA_DIR", "./qwen_lora_dpo_0226_1700_8bit")  # 为空则不加载 DPO adapter

# GPU 配置
DEVICE = os.environ.get("DEVICE", "cuda")  # cuda / cpu

# 推理优化
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "2"))

SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "soulmate")

# 默认系统提示词
DEFAULT_SYSTEM = os.environ.get(
    "DEFAULT_SYSTEM",
    "你是一个活泼开朗的男生，和想要进一步追求的女生进行对话。对话需要简短且自然流畅口语化。请务必确保使用简体中文进行回复",
)


# -----------------------
# OpenAI-like schemas
# -----------------------
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=SERVED_MODEL_NAME)
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.1  # 重复惩罚，>1 减少重复，1.0 无惩罚

    class Config:
        extra = "ignore"


# -----------------------
# Global model references
# -----------------------
tokenizer = None
model = None


def _load_model_8bit(model_name: str, device_map: str = "auto"):
    """
    使用 8-bit 量化加载模型
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    return model


def _load_model():
    """加载模型到 GPU (8-bit 量化)"""
    global tokenizer, model

    print(f"🔹 Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔹 Loading base model (8-bit) from {BASE_MODEL}...")
    print(f"   Device: {DEVICE}")
    print(f"   Quantization: 8-bit")

    if DEVICE == "cuda" and torch.cuda.is_available():
        device_map = "auto"
    else:
        print("⚠️  CUDA not available, 8-bit quantization requires CUDA!")
        raise RuntimeError("8-bit quantization requires CUDA GPU")

    base = _load_model_8bit(BASE_MODEL, device_map=device_map)

    # 加载 SFT LoRA adapter（如果存在）
    if SFT_LORA_DIR and os.path.exists(SFT_LORA_DIR):
        print(f"🔹 Loading SFT LoRA adapter from {SFT_LORA_DIR}...")
        model = PeftModel.from_pretrained(base, SFT_LORA_DIR)
        print(f"   ✅ SFT LoRA loaded")

        # 加载 DPO LoRA adapter（如果存在，在 SFT 基础上叠加）
        if DPO_LORA_DIR and os.path.exists(DPO_LORA_DIR):
            print(f"🔹 Loading DPO LoRA adapter from {DPO_LORA_DIR}...")
            # 加载 DPO adapter 并设置为活动 adapter
            model.load_adapter(DPO_LORA_DIR, adapter_name="dpo")
            model.set_adapter("dpo")
            print(f"   ✅ DPO LoRA loaded (active adapter: dpo)")
        elif DPO_LORA_DIR:
            print(f"⚠️  DPO LoRA directory not found: {DPO_LORA_DIR}, using SFT only")
    else:
        print(f"⚠️  SFT LoRA directory not found: {SFT_LORA_DIR}, using base model")
        model = base

    model.eval()

    # 验证模型
    print(f"🔹 Verifying model...")
    with torch.no_grad():
        test_input = tokenizer("你好", return_tensors="pt").to(model.device)
        test_output = model(**test_input)
        logits = test_output.logits
        print(
            f"   Model logits: [{logits.min().item():.2f}, {logits.max().item():.2f}]"
        )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("❌ Model produces NaN/Inf!")
        print(f"   ✅ Model verification passed")

    # 打印 GPU 显存使用情况
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"   GPU {i}: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 Starting 8-bit GPU Inference Server")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   SFT LoRA: {SFT_LORA_DIR or 'None'}")
    print(f"   DPO LoRA: {DPO_LORA_DIR or 'None'}")
    print(f"   Quantization: 8-bit")
    print("=" * 60)
    _load_model()
    print("✅ Model loaded successfully")
    print("=" * 60)
    yield
    print("🔻 Shutting down...")


def _ensure_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """确保消息列表包含系统提示"""
    if not messages:
        return [{"role": "system", "content": DEFAULT_SYSTEM}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": DEFAULT_SYSTEM}] + messages
    return messages


@torch.inference_mode()
def _generate_chat(
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float = 1.1,
) -> str:
    """非流式生成"""
    assert tokenizer is not None and model is not None

    messages = _ensure_system(messages)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # 移动到 GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0

    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # 启用 KV Cache
    )

    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 过滤掉 <think>...</think> 标签
    text = _filter_think_tags(text)

    return text.strip()


def _generate_chat_stream(
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float = 1.1,
) -> Iterator[str]:
    """
    真正的流式生成 - 逐 token 输出
    使用 TextIteratorStreamer 实现异步流式生成
    """
    assert tokenizer is not None and model is not None

    messages = _ensure_system(messages)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    # 移动到 GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0

    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "streamer": streamer,
    }

    # 在后台线程中运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐个 token 返回
    for text in streamer:
        if text:
            yield text

    thread.join()


# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(
    title="Soulmate 8-bit GPU Inference Server",
    description="High-performance 8-bit quantized GPU inference server with OpenAI-compatible API",
    lifespan=lifespan,
)

# 添加 CORS 中间件，支持跨域请求（允许前端页面调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
def health_check():
    """健康检查"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": SERVED_MODEL_NAME, "quantization": "8-bit"}


def _create_chat_completion_response(content: str, model_name: str) -> dict:
    """创建标准聊天完成响应"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _create_stream_chunk(
    content: str, model_name: str, chunk_id: str, finish_reason: Optional[str] = None
) -> str:
    """创建 SSE 流式响应的单个 chunk"""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _filter_think_tags(text: str) -> str:
    """
    过滤掉 <think>...</think> 标签及其内容
    支持多行匹配和嵌套空白
    """
    # 匹配 <think> 和 </think> 之间的所有内容（包括换行）
    pattern = r"<think>[\s\S]*?</think>"
    filtered = re.sub(pattern, "", text)
    # 清理多余的空白行
    filtered = re.sub(r"\n\s*\n", "\n", filtered)
    return filtered.strip()


async def _stream_response_real(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float = 1.1,
):
    """
    真正的流式响应 - 逐 token 发送
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 使用真正的流式生成器
    for token_text in _generate_chat_stream(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    ):
        yield _create_stream_chunk(token_text, model_name, chunk_id)
        # 短暂让出控制权，确保响应能及时发送
        await asyncio.sleep(0)

    # 发送结束标记
    yield _create_stream_chunk("", model_name, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI 兼容的聊天完成接口"""
    messages = [m.model_dump() for m in req.messages]

    # 流式响应
    if req.stream:
        return StreamingResponse(
            _stream_response_real(
                messages=messages,
                model_name=req.model,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.7,
                max_tokens=req.max_tokens or 1024,
                repetition_penalty=req.repetition_penalty or 1.1,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            },
        )

    # 非流式响应
    content = _generate_chat(
        messages=messages,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.7,
        max_tokens=req.max_tokens or 1024,
        repetition_penalty=req.repetition_penalty or 1.1,
    )

    return _create_chat_completion_response(content, req.model)


# -----------------------
# Main entry point
# -----------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
