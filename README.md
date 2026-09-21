# Conduit

A voice agent for macOS. Press a key, say what you want, your machine does it.
Runs entirely on-device — no cloud API, no network call, no bill.

**Status:** day 2 of 22.

## Why

Voice loses to a keyboard on short, precise tasks. It wins on compound errands —
"move my 3pm to tomorrow and tell Priya why" — which cost 30–120 seconds of
navigating and context-switching, and four seconds spoken.

## Benchmarks

End of speech → spoken reply. n=15, single configuration.
qwen2.5:7b @ temp 0 via Ollama, whisper-small on MLX, M5 MacBook Air 16GB.

| stage | day 2 | day 19 |
|---|---|---|
| ASR (after speech ends) | 274 ms | |
| model (2–3 calls) | 1,448 ms | |
| tool execution | ~30 ms | |
| **total p50** | **1,761 ms** | |
| total p95 | 2,955 ms | |

Started at 10,432 ms. Four changes, measured one at a time:

| change | effect |
|---|---|
| dropped reasoning model (qwen3 to qwen2.5) | 8,700 ms to 1,900 ms |
| raised max_tokens (reasoning truncated the tool call) | fixed silent empty responses |
| temperature 1.0 to 0 | fewer retries, no malformed output |
| one-line positive tool descriptions | correct tool selection |

Not yet measured: endpointing (currently push-to-talk) and TTS (currently macOS `say`).

## Tools

open_app, spotify_control, now_playing, get_calendar_today, get_clipboard, list_files

## Running

    uv venv && source .venv/bin/activate
    uv pip install sounddevice soundfile mlx-whisper openai pynput pynput
    ollama pull qwen2.5:7b
    python conduit.py
