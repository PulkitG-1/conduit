import inspect, time
import numpy as np, sounddevice as sd, soundfile as sf
from kokoro_onnx import Kokoro

k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("methods:", [m for m in dir(k) if not m.startswith("_")])
for name in ("create", "generate"):
    if hasattr(k, name):
        print(f"{name}{inspect.signature(getattr(k, name))}")

fn = getattr(k, "create", None) or k.generate
t = time.perf_counter()
out = fn("Spotify is now open.", voice="af_heart")
gen_ms = (time.perf_counter() - t) * 1000

samples, rate = out if isinstance(out, tuple) else (out, getattr(k, "sample_rate", 24000))
print(f"generated {len(samples)/rate:.2f}s of audio in {gen_ms:.0f}ms   (rate {rate})")
sf.write("kokoro_test.wav", samples, rate)
sd.play(np.asarray(samples), rate); sd.wait()
