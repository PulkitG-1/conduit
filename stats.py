import json, statistics, sys
from collections import defaultdict
rows = [json.loads(l) for l in open("turns.jsonl")]
if len(sys.argv) > 1:
    rows = rows[-int(sys.argv[1]):]
groups = defaultdict(list)
for r in rows:
    groups[(r.get("model", "?"), r.get("path", "model"))].append(r["to_speech_ms"])
for (model, path), t in sorted(groups.items()):
    t.sort()
    print(f"{model:14} {path:7} n={len(t):3}  p50 {statistics.median(t):.0f}ms  "
          f"min {t[0]}ms  max {t[-1]}ms")
