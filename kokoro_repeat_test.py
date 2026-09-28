import time
from kokoro_onnx import Kokoro

k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
k.create("warm up", voice="af_heart")          # first call pays any one-time setup

for text in ("Paused.", "Opened Spotify.", "Paused."):
    times = []
    for _ in range(3):
        t = time.perf_counter()
        samples, rate = k.create(text, voice="af_heart")
        times.append((time.perf_counter() - t) * 1000)
    audio_s = len(samples) / rate
    print(f"{text:18} {[f'{x:.0f}ms' for x in times]}   audio {audio_s:.2f}s")
