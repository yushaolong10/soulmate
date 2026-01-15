import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen3-1.7B"
adapter = "qwen_lora_adapter_0115_x"

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter)

msgs = [
    {
        "role": "system",
        "content": "你需要扮演一个虚拟男生角色，和想要进一步追求的女生进行对话。对话需要简短且自然流畅口语化。请务必确保使用简体中文进行回复",
    },
    {"role": "user", "content": "我今天工作好累，刚下班，你想我没？"},
]
prompt = tok.apply_chat_template(
    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.7
)
print(tok.decode(out[0], skip_special_tokens=True))
