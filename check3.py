import mlx_whisper, soundfile as sf, numpy as np

a, sr = sf.read("test.wav")
print("duration:", round(len(a)/sr, 2), "s   rms:", round(float(np.sqrt((a**2).mean())), 4))

out = mlx_whisper.transcribe("test.wav",
        path_or_hf_repo="mlx-community/whisper-small-mlx",
        language="en", condition_on_previous_text=False)

print("text:", repr(out["text"]))
for s in out["segments"]:
    print(f"  [{s['start']:.1f}–{s['end']:.1f}] {s['text']!r}  no_speech={s.get('no_speech_prob', 0):.2f}")
