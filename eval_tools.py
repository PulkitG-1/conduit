import json, statistics, time
from openai import OpenAI
from conduit import SYSTEM, TOOLS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODELS = ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b"]

# the set we tuned against: (utterance, expected tool, expected args)
TUNED = [
    ("open Spotify", "open_app", None), ("open Chrome", "open_app", None),
    ("launch Slack", "open_app", None), ("play the music", "music_control", None),
    ("pause", "music_control", None), ("next song", "music_control", None),
    ("skip this track", "music_control", None),
    ("go back to the previous song", "music_control", None),
    ("what's playing", "now_playing", None), ("what song is this", "now_playing", None),
    ("what's on my calendar", "get_calendar_today", None),
    ("do I have any meetings today", "get_calendar_today", None),
    ("what's in my clipboard", "get_clipboard", None),
    ("what did I just copy", "get_clipboard", None),
    ("what files are in my downloads", "list_files", None),
    ("show me what's on my desktop", "list_files", None),
    ("what can you do", None, None), ("hello", None, None),
]

# never seen during tuning, with arguments checked
HELDOUT = [
    ("fire up Safari", "open_app", {"name": "safari"}),
    ("can you open the calculator", "open_app", {"name": "calculator"}),
    ("stop the music", "music_control", {"action": "pause"}),
    ("resume the song", "music_control", {"action": "play"}),
    ("play the next one", "music_control", {"action": "next"}),
    ("previous track please", "music_control", {"action": "previous"}),
    ("who's singing this", "now_playing", None),
    ("am I free this afternoon", "get_calendar_today", None),
    ("read me what I copied", "get_clipboard", None),
    ("anything new in my documents", "list_files", {"folder": "documents"}),
    ("what's on my desktop", "list_files", {"folder": "desktop"}),
    ("thanks", None, None),
    ("how are you", None, None),
    ("turn the volume up", None, None),   # no tool can do this: correct is to call nothing
]


def run(model, text):
    t = time.perf_counter()
    r = client.chat.completions.create(
        model=model, tools=TOOLS, temperature=0, max_tokens=200,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}])
    ms = (time.perf_counter() - t) * 1000
    calls = r.choices[0].message.tool_calls
    if not calls:
        return None, {}, ms
    try:
        args = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        args = {}
    return calls[0].function.name, args, ms


def correct(got, args, want, want_args):
    if got != want:
        return False
    return all(str(v).lower() in str(args.get(k, "")).lower()
               for k, v in (want_args or {}).items())


for model in MODELS:
    client.chat.completions.create(model=model, max_tokens=1,
                                   messages=[{"role": "user", "content": "hi"}])
    times, report = [], []
    for label, cases in (("tuned", TUNED), ("held-out", HELDOUT)):
        hits = 0
        for text, want, want_args in cases:
            got, args, ms = run(model, text)
            times.append(ms)
            if correct(got, args, want, want_args):
                hits += 1
            else:
                report.append(f"   {label} miss: {text!r} -> {got} {args or ''}  (wanted {want} {want_args or ''})")
        report.insert(0 if label == "tuned" else len(report), "")
        print_line = f"{label} {hits}/{len(cases)}"
        report.append(print_line) if False else None
        globals().setdefault("scores", {}).setdefault(model, []).append(print_line)
    print(f"\n{model:14} {'   '.join(scores[model])}   p50 {statistics.median(times):.0f}ms")
    for line in report:
        if line.strip():
            print(line)
