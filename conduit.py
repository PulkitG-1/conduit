import collections, json, os, queue, subprocess, threading, time
from datetime import datetime
import numpy as np
import sounddevice as sd, soundfile as sf, mlx_whisper
from openai import OpenAI
from pynput import keyboard

ASR_MODEL = "mlx-community/whisper-small-mlx"
ASR_PROMPT = ("Voice commands for a Mac: open Spotify, play the music, pause, next song, "
              "what's playing, what's on my calendar, what's in my clipboard, "
              "what files are in my downloads, open Chrome.")
LLM_MODEL = "qwen2.5:7b"
TEMP = 0
RATE = 16000
HOTKEY = keyboard.Key.alt_r      # hold RIGHT Option to talk
MIN_SECONDS = 0.3                # ignore accidental taps
os.makedirs("recordings", exist_ok=True)

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
        code, out, err = osa('tell application "Spotify" to (name of current track) '
                             '& " by " & (artist of current track)')
        return (out or "nothing playing") if code == 0 else f"spotify error: {err}"

    if name == "get_calendar_today":
        return "10:00 standup, 15:00 design review"

    if name == "get_clipboard":
        p = subprocess.run(["pbpaste"], capture_output=True, text=True)
        return p.stdout.strip()[:500] or "clipboard is empty"

    if name == "list_files":
        folder = args.get("folder", "Home")
        path = os.path.expanduser("~" if folder == "Home" else f"~/{folder}")
        try:
            entries = [e for e in os.scandir(path) if not e.name.startswith(".")]
        except OSError as e:
            return f"could not read {path}: {e}"
        if not entries:
            return f"{folder} is empty"
        recent = sorted(entries, key=lambda e: e.stat().st_mtime, reverse=True)[:5]
        return f"{len(entries)} items in {folder}. Most recent: " + ", ".join(e.name for e in recent)

    return f"unknown tool: {name}"


def speak(text):
    if text:
        subprocess.run(["say", text])


class Recorder:
    """Mic stays open; ~320ms of pre-roll so the first syllable isn't clipped."""
    def __init__(self):
        self.recording = False
        self.frames = []
        self.preroll = collections.deque(maxlen=10)   # 10 x 32ms blocks
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                     blocksize=512, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        (self.frames if self.recording else self.preroll).append(indata.copy())

    def start(self):
        if not self.recording:
            self.frames = list(self.preroll)
            self.recording = True

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        return np.concatenate(self.frames).flatten() if self.frames else None


def warmup():
    t = time.perf_counter()
    subprocess.run(["say", "-o", "recordings/_warmup.aiff", "warming up"])
    mlx_whisper.transcribe("recordings/_warmup.aiff", path_or_hf_repo=ASR_MODEL, language="en")
    client.chat.completions.create(model=LLM_MODEL,
                                   messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    print(f"warm in {(time.perf_counter() - t) * 1000:.0f}ms")


rec, jobs, busy = Recorder(), queue.Queue(), threading.Event()


def on_press(key):
    if key == HOTKEY and not busy.is_set():
        rec.start()


def on_release(key):
    if key == HOTKEY:
        audio = rec.stop()
        if audio is not None:
            jobs.put((audio, time.perf_counter()))   # clock starts when you let go


def transcribe(audio, t0):
    path = f"recordings/{datetime.now():%Y%m%d-%H%M%S}.wav"
    sf.write(path, audio, RATE)
    text = mlx_whisper.transcribe(path, path_or_hf_repo=ASR_MODEL,
                                  language="en", initial_prompt=ASR_PROMPT)["text"].strip()
    return text, (time.perf_counter() - t0) * 1000


def record(said, asr_ms, llm_ms, tool_ms, total_ms):
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "said": said,
           "model": LLM_MODEL, "asr": ASR_MODEL, "temp": TEMP, "input": "hotkey", "asr_prompt": True,
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
        r = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                           tools=TOOLS, max_tokens=400,
                                           temperature=TEMP)
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
    warmup()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("hold RIGHT OPTION to talk, release to send. say 'quit' or Ctrl+C to stop.")
    try:
        while True:
            audio, t0 = jobs.get()
            if len(audio) < RATE * MIN_SECONDS:
                continue
            busy.set()
            try:
                said, asr_ms = transcribe(audio, t0)
                print("heard:", said)
                if said.lower().strip(".!? ") in ("quit", "exit"):
                    break
                if said:
                    loop(said, t0, asr_ms)
            finally:
                busy.clear()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
