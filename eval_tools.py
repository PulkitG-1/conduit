import json, statistics, time
from openai import OpenAI
from conduit import SYSTEM, TOOLS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODELS = ["qwen2.5:3b"]          # the one we ship; add others to compare
NO_ACTION = (None, "unclear")   # either counts as correctly not acting

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
    ("what can you do", NO_ACTION, None), ("hello", NO_ACTION, None),
]

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
    ("thanks", NO_ACTION, None),
    ("how are you", NO_ACTION, None),
    ("turn the volume up", NO_ACTION, None),
]

ADVERSARIAL = [   # garbled or meaningless: acting on these is the failure
    ("welcome to party club", NO_ACTION, None),
    ("banana keyboard", NO_ACTION, None),
    ("the weather is nice today", NO_ACTION, None),
    ("open lean on music", NO_ACTION + ("open_app",), None),
    ("open Lena Music", ("open_app", "unclear"), None),
    ("launch the app called Zorblax", "open_app", {"name": "zorblax"}),
]

SETS = {"tuned": TUNED, "held-out": HELDOUT, "adversarial": ADVERSARIAL}


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
    wants = want if isinstance(want, tuple) else (want,)
    if got not in wants:
        return False
    return all(str(v).lower() in str(args.get(k, "")).lower() for k, v in (want_args or {}).items())


for model in MODELS:
    client.chat.completions.create(model=model, max_tokens=1,
                                   messages=[{"role": "user", "content": "hi"}])
    times, scores, misses = [], [], []
    for label, cases in SETS.items():
        hits = 0
        for text, want, want_args in cases:
            got, args, ms = run(model, text)
            times.append(ms)
            if correct(got, args, want, want_args):
                hits += 1
            else:
                misses.append(f"   {label} miss: {text!r} -> {got} {args or ''}  (wanted {want} {want_args or ''})")
        scores.append(f"{label} {hits}/{len(cases)}")
    print(f"\n{model:14} {'   '.join(scores)}   p50 {statistics.median(times):.0f}ms")
    for m in misses:
        print(m)
