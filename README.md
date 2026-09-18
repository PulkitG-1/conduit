# Conduit

A voice agent for macOS. Hold a key, say what you want, your machine does it.

**Status:** in progress — day 1 of 22.

## Why

Voice loses to a keyboard on short, precise tasks. It wins on compound errands —
"move my 3pm to tomorrow morning and tell Priya why" — which cost 30–120 seconds
of navigating and context-switching, and four seconds spoken.

## Benchmarks

Voice stop → first audio out. p50 over 30 utterances, on an M5 MacBook Air.

| stage | day 4 | day 19 |
|---|---|---|
| endpoint decision | | |
| ASR tail | | |
| LLM TTFT | | |
| tool execution | | |
| TTS first audio | | |
| orchestration | | |
| **total p50** | | |

## Stack

On-device ASR (Whisper via MLX) · Anthropic API · MCP tool server · OpenTelemetry

## Running

TBD
