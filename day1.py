import json, subprocess
import sounddevice as sd, soundfile as sf, mlx_whisper
from openai import OpenAI

ASR_MODEL = "mlx-community/whisper-small-mlx"
LLM_MODEL = "qwen3:8b"
RATE, SECONDS = 16000, 5

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM = "You control a Mac. Use the tools when the user asks for an action. Reply in one short sentence. /no_think"

TOOLS = [
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Open a macOS application by name.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "get_calendar_today",
        "description": "Return today's calendar events.",
        "parameters": {"type": "object", "properties": {}}}},
]


def act(name, args):
    if name == "open_app":
        subprocess.run(["osascript", "-e",
                        f'tell application "{args["name"]}" to activate'])
        return f"opened {args['name']}"
    if name == "get_calendar_today":
        return "10:00 standup, 15:00 design review"
    return f"unknown tool: {name}"


def listen():
    input("press enter, then speak: ")
    audio = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write("turn.wav", audio, RATE)
    return mlx_whisper.transcribe("turn.wav",
                                  path_or_hf_repo=ASR_MODEL,
                                  language="en")["text"].strip()


def loop(user_text):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text}]

    for turn in range(5):
        r = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, max_tokens=200)
        m = r.choices[0].message

        assistant_msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        messages.append(assistant_msg)

        if not m.tool_calls:
            print("agent:", m.content)
            return

        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = act(tc.function.name, args)
            except json.JSONDecodeError:
                result = f"error: malformed arguments {tc.function.arguments!r}"
            print(f"  tool {tc.function.name} -> {result}")
            messages.append({"role": "tool",
                             "tool_call_id": tc.id,
                             "content": result})

    print("agent: gave up after 5 turns")


if __name__ == "__main__":
    said = listen()
    print("heard:", said)
    loop(said)
