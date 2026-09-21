import collections, json, os, queue, re, urllib.request, subprocess, threading, time
from datetime import datetime
import numpy as np
import sounddevice as sd, soundfile as sf, mlx_whisper
from openai import OpenAI
from pynput import keyboard

ASR_MODEL = "mlx-community/whisper-small-mlx"
ASR_PROMPT = ("Voice commands for a Mac: open Spotify, play the music, pause, next song, "
              "what's playing, what's on my calendar, what's in my clipboard, "
              "what files are in my downloads, open Chrome.")
LLM_MODEL = "qwen2.5:3b"
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
        "name": "playback",
        "description": "Play, pause, skip or go back a song.",
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


def fail(error_type, message, try_instead=None):
    """A failure the model can act on, instead of a bare string."""
    out = {"ok": False, "error_type": error_type, "message": message}
    if try_instead:
        out["try_instead"] = try_instead
    return json.dumps(out)


def act(name, args):   # args already passed validate(), so enums are safe to index
    if name == "open_app":
        p = subprocess.run(["open", "-a", args["name"]], capture_output=True, text=True)
        if p.returncode != 0:
            return fail("app_not_found", f"No application named {args['name']!r} is installed.",
                        "Tell the user it is not installed. Do not retry the same name.")
        return f"opened {args['name']}"

    if name == "playback":
        verbs = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
        code, _, err = osa(f'tell application "Spotify" to {verbs[args["action"]]}')
        if code != 0:
            return fail("spotify_error", err or "Spotify did not respond.",
                        "Open Spotify with open_app, then try again.")
        return f"spotify {args['action']}"

    if name == "now_playing":
        code, out, err = osa('tell application "Spotify" to (name of current track) '
                             '& " by " & (artist of current track)')
        if code != 0:
            return fail("spotify_error", err or "Spotify did not respond.", "Spotify may not be open.")
        return out or "nothing playing"

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
            return fail("cannot_read_folder", str(e))
        if not entries:
            return f"{folder} is empty"
        recent = sorted(entries, key=lambda e: e.stat().st_mtime, reverse=True)[:5]
        return f"{len(entries)} items in {folder}. Most recent: " + ", ".join(e.name for e in recent)

    return fail("unknown_tool", f"There is no tool called {name!r}.")


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


def pin_model():
    """Ask Ollama to keep the model loaded indefinitely (its default unloads after 5 idle minutes)."""
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": LLM_MODEL, "keep_alive": -1}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def warmup():
    t = time.perf_counter()
    pin_model()
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


KNOWN_HALLUCINATIONS = {"you", "thank you", "thanks for watching", "bye"}


def looks_hallucinated(text, seconds):
    """Catch Whisper output that can't be real speech, whatever the model or prompt."""
    t = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    if len(t) < 2 or t in KNOWN_HALLUCINATIONS:
        return True
    if len(t) / max(seconds, 0.1) > 30:          # faster than any human speaks
        return True
    words = t.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.3:   # "google play google play ..."
        return True
    return False


def transcribe(audio, t0):
    path = f"recordings/{datetime.now():%Y%m%d-%H%M%S}.wav"
    sf.write(path, audio, RATE)
    text = mlx_whisper.transcribe(path, path_or_hf_repo=ASR_MODEL,
                                  language="en", initial_prompt=ASR_PROMPT)["text"].strip()
    return text, (time.perf_counter() - t0) * 1000


def record(said, asr_ms, llm_ms, tool_ms, total_ms, path="model"):
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "said": said,
           "model": LLM_MODEL, "asr": ASR_MODEL, "temp": TEMP, "input": "hotkey", "asr_prompt": True,
           "asr_ms": round(asr_ms), "llm_ms": [round(x) for x in llm_ms],
           "tool_ms": [round(x) for x in tool_ms], "calls": len(llm_ms),
           "to_speech_ms": round(total_ms), "path": path}
    with open("turns.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  asr {row['asr_ms']}ms | llm {sum(row['llm_ms'])}ms "
          f"({row['calls']} calls) | tool {sum(row['tool_ms'])}ms "
          f"| TOTAL {row['to_speech_ms']}ms")


SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def validate(name, args):
    """The schema is only a hint to the model. This is what actually enforces it."""
    schema = SCHEMAS.get(name)
    if schema is None:
        return f"unknown tool {name}"
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing argument '{key}'; expected: {list(props)}"
    for key, val in list(args.items()):
        if key not in props:
            return f"unexpected argument '{key}'; allowed: {list(props)}"
        allowed = props[key].get("enum")
        if allowed:
            match = next((a for a in allowed if str(a).lower() == str(val).lower()), None)
            if match is None:
                return f"'{val}' is not a valid {key}; allowed: {allowed}"
            args[key] = match
    return None


ACTIONS = {"open_app", "playback"}


def ok(result):
    return not result.startswith('{"ok": false')


def confirm(name, args):
    if name == "open_app":
        return f"Opened {args.get('name', 'it')}."
    return {"play": "Playing.", "pause": "Paused.", "next": "Next track.",
            "previous": "Previous track."}.get(args.get("action"), "Done.")


CLAIMS = [
    (re.compile(r"\b(opened|launched)\b", re.I), {"open_app"}),
    (re.compile(r"\b(paused|skipped|resumed|now playing|is playing)\b", re.I),
     {"playback", "now_playing"}),
]


def unbacked_claim(text, succeeded):
    """True if the reply claims an action that no successful tool call this turn backs up."""
    return any(pat.search(text or "") and not (tools & succeeded) for pat, tools in CLAIMS)


def finish(user_text, t0, asr_ms, llm_ms, tool_ms, reply, path):
    print("agent:", reply, "" if path == "model" else f" [{path}]")
    record(user_text, asr_ms, llm_ms, tool_ms, (time.perf_counter() - t0) * 1000, path=path)
    speak(reply)


def loop(user_text, t0, asr_ms):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text}]
    llm_ms, tool_ms = [], []
    succeeded, failures = set(), {}

    for _ in range(5):
        t = time.perf_counter()
        r = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                           tools=TOOLS, max_tokens=400, temperature=TEMP)
        llm_ms.append((time.perf_counter() - t) * 1000)
        m = r.choices[0].message

        if not m.tool_calls and not (m.content or "").strip():
            break   # at temperature 0 the same request returns the same nothing

        if not m.tool_calls:
            if unbacked_claim(m.content, succeeded):
                return finish(user_text, t0, asr_ms, llm_ms, tool_ms,
                              "I didn't manage to do that.", "caught_claim")
            return finish(user_text, t0, asr_ms, llm_ms, tool_ms, m.content, "model")

        messages.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [tc.model_dump() for tc in m.tool_calls]})

        done, stuck = [], None
        for tc in m.tool_calls:
            t = time.perf_counter()
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
                err = validate(name, args)
                result = (fail("invalid_arguments", err, "Call the tool again using only the allowed arguments.")
                          if err else act(name, args))
            except json.JSONDecodeError:
                args = {}
                result = fail("malformed_json", "Arguments were not valid JSON.",
                              "Send the arguments as a JSON object.")
            tool_ms.append((time.perf_counter() - t) * 1000)
            print(f"  tool {name} -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            done.append((name, args, result))

            if ok(result):
                succeeded.add(name)
            else:
                key = (name, json.dumps(args, sort_keys=True))
                failures[key] = failures.get(key, 0) + 1
                if failures[key] >= 2:
                    stuck = json.loads(result).get("message", "")

        if stuck is not None:   # same call, same failure, twice: retrying won't help
            return finish(user_text, t0, asr_ms, llm_ms, tool_ms,
                          f"Sorry, I couldn't do that. {stuck}", "stuck")

        if all(n in ACTIONS and ok(res) for n, _, res in done):
            return finish(user_text, t0, asr_ms, llm_ms, tool_ms,
                          " ".join(confirm(n, a) for n, a, _ in done), "fast")

    finish(user_text, t0, asr_ms, llm_ms, tool_ms, "Sorry, I can't do that yet.", "failed")


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
                if looks_hallucinated(said, len(audio) / RATE):
                    print("  [dropped: looks like an ASR hallucination]")
                    continue
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
