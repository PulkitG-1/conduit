import sounddevice as sd, soundfile as sf, numpy as np, time, mlx_whisper

MODEL = "mlx-community/whisper-small-mlx"
RATE, SECONDS = 16000, 6

for i in (3, 2, 1):
    print(i, end=" ", flush=True); time.sleep(1)
print("\nSPEAK NOW")

audio = sd.rec(int(SECONDS*RATE), samplerate=RATE, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

peak = float(np.abs(audio).max())
print(f"raw   peak={peak:.4f}  rms={float(np.sqrt((audio**2).mean())):.4f}")

norm = audio / max(peak, 1e-6) * 0.95
print(f"norm  peak={float(np.abs(norm).max()):.4f}  rms={float(np.sqrt((norm**2).mean())):.4f}")

sf.write("raw.wav", audio, RATE)
sf.write("norm.wav", norm, RATE)

for name in ("raw.wav", "norm.wav"):
    out = mlx_whisper.transcribe(name, path_or_hf_repo=MODEL, language="en")
    print(f"{name:9} -> {out['text']!r}")
