import csv, glob, os, subprocess, mlx_whisper
from conduit import ASR_MODEL, ASR_PROMPT

LABELS = "labels.csv"
done = {}
if os.path.exists(LABELS):
    with open(LABELS) as f:
        done = {r["file"]: r["truth"] for r in csv.DictReader(f)}
files = [p for p in sorted(glob.glob("recordings/2*.wav")) if p not in done]
print(f"{len(files)} clips to label.")
print("Listen, then type what you ACTUALLY said.")
print("Enter = Whisper got it exactly right · s = skip unclear clip · r = replay · q = quit\n")

new_file = not os.path.exists(LABELS)
with open(LABELS, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "truth"])
    if new_file:
        w.writeheader()
    for path in files:
        heard = mlx_whisper.transcribe(path, path_or_hf_repo=ASR_MODEL, language="en",
                                       initial_prompt=ASR_PROMPT)["text"].strip()
        while True:
            subprocess.run(["afplay", path])
            ans = input(f"{os.path.basename(path)}  whisper heard: {heard!r}\n  you said > ").strip()
            if ans != "r":
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        w.writerow({"file": path, "truth": ans or heard})
        f.flush()
print("saved to", LABELS)
