"""Drive the full agent loop with typed text: no mic, no speech, nothing logged."""
import sys, time, conduit
conduit.speak = lambda text: None
conduit.record = lambda *a, **k: None
for text in sys.argv[1:]:
    print(f"\n>>> {text}")
    conduit.loop(text, time.perf_counter(), 0)
