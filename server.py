import os
import time
import uuid
import json
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from contextlib import asynccontextmanager

# -----------------------
# Config
# -----------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")
LORA_DIR = os.environ.get("LORA_DIR", "./qwen_lora_adapter")
DEVICE = os.environ.get("DEVICE", "cpu")  # mps / cpu
DTYPE = os.environ.get("DTYPE", "float16")  # float16 / float32

SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "soulmate")

# Optional: force a default system prompt
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
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

    # 允许额外字段，兼容更多客户端
    class Config:
        extra = "ignore"


tokenizer = None
model = None


def _get_dtype():
    return torch.float16 if DTYPE == "float16" else torch.float32


def _load_model():
    global tokenizer, model
    print("🔹 Loading tokenizer...", BASE_MODEL, LORA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # pad token for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=_get_dtype(),
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    if DEVICE == "mps":
        model.to("mps")
    else:
        model.to("cpu")

    return


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔹 Loading model...")
    _load_model()
    print("✅ Model loaded")
    yield
    print("🔻 Shutting down...")


def _ensure_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages:
        return [{"role": "system", "content": DEFAULT_SYSTEM}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": DEFAULT_SYSTEM}] + messages
    return messages


@torch.inference_mode()
def _generate_chat(
    messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int
) -> str:
    assert tokenizer is not None and model is not None

    # Ensure system prompt
    messages = _ensure_system(messages)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]  # 记录输入长度

    if DEVICE == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    # Safer defaults: if temperature <= 0, do greedy
    do_sample = temperature is not None and temperature > 0

    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 只解码新生成的 token（跳过输入部分）
    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text.strip()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
def list_models():
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
    """创建标准的聊天完成响应"""
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
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


async def _stream_response(content: str, model_name: str):
    """生成流式响应"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 逐字符发送（模拟流式效果）
    for char in content:
        yield _create_stream_chunk(char, model_name, chunk_id)

    # 发送结束标记
    yield _create_stream_chunk("", model_name, chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]

    content = _generate_chat(
        messages=messages,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.7,
        max_tokens=req.max_tokens or 1024,
    )

    # 流式响应
    if req.stream:
        return StreamingResponse(
            _stream_response(content, req.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # 非流式响应
    return _create_chat_completion_response(content, req.model)
