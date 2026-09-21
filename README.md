# Conduit

A voice agent for macOS. Hold a key, say what you want, your machine does it.
Runs entirely on-device. No cloud API, no network call, no bill.

**Status:** week 1. Actions answer in under a second.

## Why

Voice loses to a keyboard on short, precise tasks. It wins on compound errands,
like "move my 3pm to tomorrow and tell Priya why", which cost 30 to 120 seconds
of navigating and context-switching, and four seconds spoken.

## Latency

Key released to spoken reply. qwen2.5:3b at temperature 0 via Ollama,
whisper-small on MLX, M5 MacBook Air 16GB.

| turn type | what happens | n | p50 |
|---|---|---|---|
| action (open app, play, pause) | 1 model call, templated reply | 3 | 974 ms |
| query (what's playing, calendar) | 2 model calls | 1 | 1,258 ms |
| can't do it | 1 model call, honest fallback | 1 | 696 ms |

Sample sizes on the current configuration are still small and will firm up with use.
Started at 10,432 ms.

## What moved the numbers

| change | effect |
|---|---|
| reasoning model (qwen3) to non-reasoning (qwen2.5) | 10.4 s to about 1.9 s |
| push-to-talk instead of a fixed 5 s recording window | no more waiting out the window |
| list_files returns a count plus the 5 newest, not 20 names | 11.4 s to 3.9 s on that query |
| ASR primed with the command list | first-word errors 4/9 to 1/9 on saved clips |
| successful actions skip the second model call | about 1.9 s to 1.3 s |
| qwen2.5:7b to 3b, chosen by the eval below | actions from about 1.3 s to under 1 s |

## Tool-choice eval

32 utterances with a known correct tool, and correct arguments for the held-out set.
The held-out cases were never looked at while tuning.

| model | tuned (18) | held-out (14) | p50 |
|---|---|---|---|
| qwen2.5:7b | 17 | 12 | 868 ms |
| qwen2.5:3b | 17 | 14 | 441 ms |
| qwen2.5:1.5b | 18 | 10 | 300 ms |

1.5b was perfect on the cases I tuned against and 10/14 on unseen ones, so it overfit.
3b generalised, so it's the default.

## Speech recognition eval

23 of my own recordings, labelled by hand. Exact = the whole command transcribed correctly.

| setup | size | exact | WER | p50 |
|---|---|---|---|---|
| whisper-small, no prompt | 481 MB | 15/23 | 27.9% | 117 ms |
| **whisper-small, command prompt** | **481 MB** | **21/23** | **7.2%** | **114 ms** |
| large-v3-turbo, no prompt | 1.6 GB | 17/23 | 14.1% | 369 ms |
| large-v3-turbo, command prompt | 1.6 GB | 18/23 | 9.8% | 371 ms |

The model three times the size lost on accuracy and speed. The prompt was worth six
clips to small and one to turbo. Small with the prompt also got every command that
isn't in the prompt; turbo heard "notes" as "nodes" both times. Remaining open problem:
single-word commands ("pause" heard as "voice").

## Things I got wrong

- **Enums don't make wrong answers unrepresentable.** 1.5b sent `{"app": "Safari"}` instead of `{"name": ...}` and invented an action called `volume_up`. The schema is only a hint. Arguments are now validated in code before any tool runs, which also fixed a KeyError that would have crashed the agent.
- **A tool's name is part of its prompt.** Small models sent "open Spotify" to `spotify_control` until it was renamed `music_control`.
- **Retrying at temperature 0 is pointless.** The same request gets the same answer.
- **A benchmark without its configuration is contaminated data.** Every turn now logs the model, ASR settings and which path it took.

## Known issues

- Single-word commands are the weakest link: "pause" was heard as "voice". Next up is an ASR eval on saved recordings.
- Ollama unloads an idle model after about 5 minutes, so the first turn after a break is slow.

## Tools

open_app, music_control, now_playing, get_calendar_today, get_clipboard, list_files

## Running

    uv venv && source .venv/bin/activate
    uv pip install sounddevice soundfile mlx-whisper openai pynput
    ollama pull qwen2.5:3b
    python conduit.py

Hold right Option to talk. macOS will ask for Microphone, Accessibility and
Input Monitoring permission for your terminal.

    python eval_tools.py    # tool-choice eval
    python stats.py         # latency by model and turn type
