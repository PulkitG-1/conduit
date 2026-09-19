import json, os, subprocess, time
from datetime import datetime
import sounddevice as sd, soundfile as sf, mlx_whisper
from openai import OpenAI

ASR_MODEL = "mlx-community/whisper-small-mlx"
LLM_MODEL = "qwen2.5:7b"
RATE, SECONDS = 16000, 5

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM = ("You control a Mac. Call a tool for any action the user asks for. "
          "Reply in one short sentence. State only what the tool returned.")

TOOLS = [
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Open or launch an application window.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "spotify_control",
        "description": "Control music playback.",
        "parameters": {"type": "object",
                       "properties": {"action": {"type": "string",
                                      "enum": ["play", "pause", "next", "previous"]}},
                       "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "now_playing",
        "description": "Name of the track currently playing.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_calendar_today",
        "description": "Today's calendar events.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_clipboard",
        "description": "Current clipboard contents.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in one of the user's folders.",
        "parameters": {"type": "object",
                       "properties": {"folder": {"type": "string",
                                      "enum": ["Downloads", "Documents", "Desktop", "Home"]}},
                       "required": ["folder"]}}},
]


def osa(script):
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def act(name, args):
    if name == "open_app":
        p = subprocess.run(["open", "-a", args["name"]], capture_output=True, text=True)
        return (f"opened {args['name']}" if p.returncode == 0
                else f"failed: {p.stderr.strip() or 'not found'}")

    if name == "spotify_control":
        verbs = {"play": "play", "pause": "pause",
                 "next": "next track", "previous": "previous track"}
        verb = verbs.get(args.get("action"))
        if not verb:
            return f"unknown action {args.get('action')!r}"
        code, _, err = osa(f'tell application "Spotify" to {verb}')
        return f"spotify {args['action']}" if code == 0 else f"spotify error: {err}"

    if name == "now_playing":
        code, out, err = osa(
            'tell application "Spotify" to (name of current track) '
            '& " by " & (artist of current track)')
        return out or "nothing playing" if code == 0 else f"spotify error: {err}"

    if name == "get_calendar_today":
        return "10:00 standup, 15:00 design review"

    if name == "get_clipboard":
        p = subprocess.run(["pbpaste"], capture_output=True, text=True)
        return p.stdout.strip()[:500] or "clipboard is empty"

    if name == "list_files":
        folder = args.get("folder", "Home")
        path = os.path.expanduser("~" if folder == "Home" else f"~/{folder}")
        try:
            return ", ".join(sorted(os.listdir(path))[:20]) or "empty"
        except OSError as e:
            return f"could not read {path}: {e}"

    return f"unknown tool: {name}"


def speak(text):
    if text:
        subprocess.run(["say", text])


def listen():
    input("press enter, then speak: ")
    audio = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    t0 = time.perf_counter()
    sf.write("turn.wav", audio, RATE)
    text = mlx_whisper.transcribe("turn.wav", path_or_hf_repo=ASR_MODEL,
                                  language="en")["text"].strip()
    return text, t0, (time.perf_counter() - t0) * 1000


def record(said, asr_ms, llm_ms, tool_ms, total_ms):
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "said": said,
           "model": LLM_MODEL, "asr": ASR_MODEL, "temp": 0,
           "asr_ms": round(asr_ms), "llm_ms": [round(x) for x in llm_ms],
           "tool_ms": [round(x) for x in tool_ms], "calls": len(llm_ms),
           "to_speech_ms": round(total_ms)}
    with open("turns.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  asr {row['asr_ms']}ms | llm {sum(row['llm_ms'])}ms "
          f"({row['calls']} calls) | tool {sum(row['tool_ms'])}ms "
          f"| TOTAL {row['to_speech_ms']}ms")


def loop(user_text, t0, asr_ms):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text}]
    llm_ms, tool_ms = [], []

    for _ in range(5):
        t = time.perf_counter()
        r = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, max_tokens=400, temperature=0)
        llm_ms.append((time.perf_counter() - t) * 1000)
        m = r.choices[0].message

        if not m.tool_calls and not (m.content or "").strip():
            print("  [empty, retrying]")
            continue

        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        messages.append(msg)

        if not m.tool_calls:
            print("agent:", m.content)
            record(user_text, asr_ms, llm_ms, tool_ms, (time.perf_counter() - t0) * 1000)
            speak(m.content)
            return

        for tc in m.tool_calls:
            t = time.perf_counter()
            try:
                result = act(tc.function.name, json.loads(tc.function.arguments))
            except json.JSONDecodeError:
                result = f"error: bad arguments {tc.function.arguments!r}"
            tool_ms.append((time.perf_counter() - t) * 1000)
            print(f"  tool {tc.function.name} -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("agent: gave up")


if __name__ == "__main__":
    print("say 'quit' to stop")
    while True:
        said, t0, asr_ms = listen()
        print("heard:", said)
        if not said or said.lower().strip(".!? ") in ("quit", "exit", "stop"):
            break
        loop(said, t0, asr_ms)
