import sounddevice as sd, soundfile as sf, numpy as np, time, mlx_whisper

RATE, SECONDS = 16000, 5
for i in (3, 2, 1):
    print(i); time.sleep(1)
print("SPEAK NOW —", SECONDS, "seconds")

audio = sd.rec(int(SECONDS*RATE), samplerate=RATE, channels=1, dtype='float32')
sd.wait()
print("peak level:", round(float(np.abs(audio).max()), 4))
sf.write("test.wav", audio, RATE)

out = mlx_whisper.transcribe("test.wav",
        path_or_hf_repo="mlx-community/whisper-small-mlx")
print("heard:", repr(out["text"]))
