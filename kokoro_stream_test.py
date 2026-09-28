import asyncio, inspect, time
from kokoro_onnx import Kokoro

k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print("create_stream", inspect.signature(k.create_stream))

TEXT = ("There are 39 items in your downloads. The most recent are a WhatsApp image, "
        "a resume template, and an invoice.")

async def main():
    for label, text in (("short", "Opened Spotify."), ("long", TEXT)):
        t = time.perf_counter()
        first_ms, chunks, secs = None, 0, 0.0
        async for item in k.create_stream(text, voice="af_heart"):
            if first_ms is None:
                first_ms = (time.perf_counter() - t) * 1000
                print(f"  first item: {type(item).__name__}")
            samples, rate = item if isinstance(item, tuple) else (item, 24000)
            chunks += 1
            secs += len(samples) / rate
        total_ms = (time.perf_counter() - t) * 1000
        print(f"{label:6} first audio {first_ms:.0f}ms | {chunks} chunks | "
              f"{secs:.2f}s audio | {total_ms:.0f}ms total")

asyncio.run(main())
