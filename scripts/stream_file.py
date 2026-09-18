"""Headless client: stream an audio file into the running server as if it were the microphone.

    python scripts/stream_file.py path/to/audio.wav [--url ws://localhost:8000/ws] [--realtime]

Prints every server event; useful for testing the pipeline without a browser.
"""
import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--realtime", action="store_true", help="pace at 1x instead of as fast as possible")
    a = ap.parse_args()

    audio, sr = sf.read(a.file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        idx = np.arange(0, len(audio), sr / 16000)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    pcm = (audio * 32767).astype(np.int16)

    async with websockets.connect(a.url, max_size=None) as ws:
        async def reader():
            async for msg in ws:
                m = json.loads(msg)
                if m["type"] == "vad":
                    continue
                if m["type"] == "llm_delta":
                    print(m["text"], end="", flush=True)
                elif m["type"] == "transcript":
                    print(f"\n### utt {m['id']} [{m['qe']['label']} {m['qe']['score']}]: {m['asr']['text']}\n{m['qe']['context']}\n")
                else:
                    print(f"\n<{m['type']}> {json.dumps({k: v for k, v in m.items() if k != 'type'}, ensure_ascii=False)[:300]}")
        rt = asyncio.create_task(reader())
        step = 1600
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step].tobytes())
            if a.realtime:
                await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "flush"}))
        await asyncio.sleep(15)     # let ASR + LLM finish
        rt.cancel()


asyncio.run(main())
