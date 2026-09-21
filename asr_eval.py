import csv, re, statistics, time, mlx_whisper
from conduit import ASR_PROMPT

rows = list(csv.DictReader(open("labels.csv")))
VARIANTS = {
    "small / no prompt": ("mlx-community/whisper-small-mlx", None),
    "small / prompt":    ("mlx-community/whisper-small-mlx", ASR_PROMPT),
    "turbo / no prompt": ("mlx-community/whisper-large-v3-turbo", None),
    "turbo / prompt":    ("mlx-community/whisper-large-v3-turbo", ASR_PROMPT),
}

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().replace("'", ""))).strip()

def wer(ref, hyp):   # word error rate: edits needed / words said
    r, h = ref.split(), hyp.split()
    row = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        diag, row[0] = row[0], i
        for j in range(1, len(h) + 1):
            diag, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diag + (r[i - 1] != h[j - 1]))
    return row[-1] / max(len(r), 1)

for name, (model, prompt) in VARIANTS.items():
    kw = {"initial_prompt": prompt} if prompt else {}
    mlx_whisper.transcribe(rows[0]["file"], path_or_hf_repo=model, language="en", **kw)  # load + warm
    exact, wers, ms, misses = 0, [], [], []
    for r in rows:
        t = time.perf_counter()
        out = mlx_whisper.transcribe(r["file"], path_or_hf_repo=model, language="en", **kw)["text"]
        ms.append((time.perf_counter() - t) * 1000)
        ref, hyp = norm(r["truth"]), norm(out)
        wers.append(wer(ref, hyp))
        if ref == hyp:
            exact += 1
        else:
            misses.append(f"{r['truth']!r} -> {out.strip()!r}")
    print(f"\n{name:18} exact {exact}/{len(rows)}   WER {statistics.mean(wers):.1%}   p50 {statistics.median(ms):.0f}ms")
    for m in misses:
        print("    miss:", m)
