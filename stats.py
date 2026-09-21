import json, statistics, sys
from collections import defaultdict
rows = [json.loads(l) for l in open("turns.jsonl")]
if len(sys.argv) > 1:
    rows = rows[-int(sys.argv[1]):]
groups = defaultdict(list)
for r in rows:
    groups[r.get("path", "model")].append(r["to_speech_ms"])
for path, t in sorted(groups.items()):
    t.sort()
    print(f"{path:6} n={len(t):3}  p50 {statistics.median(t):.0f}ms  min {t[0]}ms  max {t[-1]}ms")
