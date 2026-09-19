import json, statistics, sys
rows = [json.loads(l) for l in open("turns.jsonl")]
if len(sys.argv) > 1:
    rows = rows[-int(sys.argv[1]):]
t = sorted(r["to_speech_ms"] for r in rows)
a = sorted(r["asr_ms"] for r in rows)
l = sorted(sum(r["llm_ms"]) for r in rows)
print(f"n={len(t)}")
print(f"total  p50 {statistics.median(t):.0f}ms   p95 {t[int(len(t)*0.95)-1]}ms   min {t[0]}ms")
print(f"asr    p50 {statistics.median(a):.0f}ms")
print(f"llm    p50 {statistics.median(l):.0f}ms")
