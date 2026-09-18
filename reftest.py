import mlx_whisper
out = mlx_whisper.transcribe("ref.wav",
        path_or_hf_repo="mlx-community/whisper-small-mlx", language="en")
print("heard:", repr(out["text"]))
