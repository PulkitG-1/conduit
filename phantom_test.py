import sounddevice as sd, soundfile as sf, mlx_whisper
from conduit import ASR_MODEL, ASR_PROMPT

RATE = 16000
input("Press Enter, then stay completely silent for 2 seconds...")
a = sd.rec(int(2 * RATE), samplerate=RATE, channels=1, dtype="float32"); sd.wait()
sf.write("recordings/_silence.wav", a, RATE)

variants = {
    "no prompt":      {},
    "current prompt": {"initial_prompt": ASR_PROMPT},
    "nouns only":     {"initial_prompt": "Spotify, Chrome, Safari, Notes, Downloads, Desktop, Documents"},
}
for name, kw in variants.items():
    for i in range(3):
        out = mlx_whisper.transcribe("recordings/_silence.wav", path_or_hf_repo=ASR_MODEL,
                                     language="en", **kw)["text"].strip()
        print(f"{name:15} run {i+1}: {out!r}")
