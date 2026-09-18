import soundfile as sf, numpy as np
a, sr = sf.read("test.wav")
print("samples:", len(a), "rate:", sr)
print("peak:", np.abs(a).max(), " rms:", np.sqrt((a**2).mean()))
