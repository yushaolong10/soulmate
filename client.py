from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8026/v1", api_key="dummy")

resp = client.chat.completions.create(
    model="soulmate",
    messages=[{"role": "system", "content": "你是一个贴心的男生，能够及时回复女生消息，并且能够提供帮助。"},{"role": "user", "content": "你住哪？"}],
    temperature=0.7,
    top_p=0.7,
    max_tokens=1024,
)

print(resp.choices[0].message.content)
