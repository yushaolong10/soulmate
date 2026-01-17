---
library_name: peft
model_name: qwen_lora_adapter_0116_l
tags:
- base_model:adapter:/home/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-14B
- lora
- sft
- transformers
- trl
licence: license
base_model: /home/yushaolong/.cache/modelscope/hub/models/Qwen/Qwen3-14B
pipeline_tag: text-generation
---

# Model Card for qwen_lora_adapter_0116_l

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 0.26.2
- Transformers: 4.57.5
- Pytorch: 2.9.1
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```