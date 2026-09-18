import sounddevice as sd, soundfile as sf, mlx_whisper

print(sd.query_devices())

SECONDS, RATE = 4, 16000
print("speak now...")
audio = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1)
sd.wait()
sf.write("test.wav", audio, RATE)

out = mlx_whisper.transcribe("test.wav",
        path_or_hf_repo="mlx-community/whisper-small-mlx")
print("heard:", out["text"])
