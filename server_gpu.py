# server_gpu.py
# 高效 GPU 推理服务器 - 针对 Ubuntu CUDA 环境优化
#
# 特性：
#   - CUDA GPU 加速推理
#   - 支持多 GPU 自动分配
#   - BFloat16/Float16 混合精度
#   - KV Cache 优化
#   - 真正的流式生成（逐 token 输出）
#   - 支持 vLLM 风格的批量推理（可选）
#
# 运行：
#   # 使用 GPU 0
#   CUDA_VISIBLE_DEVICES=0 python server_gpu.py
#
#   # 使用 GPU 1
#   CUDA_VISIBLE_DEVICES=1 python server_gpu.py
#
#   # 指定端口
#   CUDA_VISIBLE_DEVICES=0 uvicorn server_gpu:app --host 0.0.0.0 --port 8000

import os
import re
import time
import uuid
import json
import asyncio
from typing import Any, Dict, List, Optional, Iterator
from threading import Thread
from queue import Queue

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from contextlib import asynccontextmanager
from transformers import TextIteratorStreamer

# -----------------------
# Config
# -----------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
LORA_DIR = os.environ.get("LORA_DIR", "./qwen_lora_adapter_0115_s3")

# GPU 配置
DEVICE = os.environ.get("DEVICE", "cuda")  # cuda / cpu
DTYPE = os.environ.get("DTYPE", "bfloat16")  # bfloat16 / float16 / float32

# 推理优化
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "false").lower() == "true"
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1"))

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

    class Config:
        extra = "ignore"


# -----------------------
# Global model references
# -----------------------
tokenizer = None
model = None


def _get_dtype():
    """获取 torch 数据类型"""
    if DTYPE == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("⚠️  BFloat16 not supported, falling back to Float16")
        return torch.float16
    elif DTYPE == "float16":
        return torch.float16
    else:
        return torch.float32


def _load_model():
    """加载模型到 GPU"""
    global tokenizer, model

    print(f"🔹 Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔹 Loading base model from {BASE_MODEL}...")
    print(f"   Device: {DEVICE}")
    print(f"   Dtype: {DTYPE}")

    # 模型加载配置
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _get_dtype(),
    }

    # GPU 加载
    if DEVICE == "cuda" and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"  # 自动分配到可用 GPU
        if USE_FLASH_ATTN:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("   Flash Attention 2: Enabled")
    else:
        print("⚠️  CUDA not available, using CPU")

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

    # 加载 LoRA adapter（如果存在）
    if LORA_DIR and os.path.exists(LORA_DIR):
        print(f"🔹 Loading LoRA adapter from {LORA_DIR}...")
        model = PeftModel.from_pretrained(base, LORA_DIR)
    else:
        print(f"⚠️  LoRA directory not found: {LORA_DIR}, using base model")
        model = base

    model.eval()

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
    print("🚀 Starting GPU Inference Server")
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
) -> str:
    """非流式生成"""
    assert tokenizer is not None and model is not None

    messages = _ensure_system(messages)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # 移动到 GPU
    if DEVICE == "cuda" and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0

    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
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
) -> Iterator[str]:
    """
    真正的流式生成 - 逐 token 输出
    使用 TextIteratorStreamer 实现异步流式生成
    """
    assert tokenizer is not None and model is not None

    messages = _ensure_system(messages)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    if DEVICE == "cuda" and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

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
    title="Soulmate GPU Inference Server",
    description="High-performance GPU inference server with OpenAI-compatible API",
    lifespan=lifespan,
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
    )

    return _create_chat_completion_response(content, req.model)
