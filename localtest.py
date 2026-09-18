from openai import OpenAI
c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
r = c.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "reply with one word: working"}])
print(r.choices[0].message.content)
