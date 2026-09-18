"""FastAPI server: serves the UI and runs Audio -> VAD -> ASR -> QE -> LLM over one WebSocket per browser tab.

Client -> server : binary frames of int16 PCM, mono, 16 kHz (any chunk size)
                   text  {"type": "flush"}   close the in-progress utterance (file playback ended / mic stopped)
                   text  {"type": "reset"}   clear chat history on both sides
Server -> client : {"type":"ready", ...}                     backends in use
                   {"type":"vad", "prob": f, "speaking": b}  throttled, for the level meter
                   {"type":"utterance", "id": n}             speech ended, ASR running
                   {"type":"transcript", "id", "asr", "qe"}  ASR + QE result
                   {"type":"dropped", "id", "reason"}
                   {"type":"llm_delta", "id", "side", "text"} side = plain | qe
                   {"type":"llm_done",  "id", "side", "text"}
                   {"type":"error", "message"}
"""
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config as C
from .asr import build_asr
from .llm import build_llm
from .qe import QE_GUIDANCE, estimate
from .vad import UtteranceSegmenter

log = logging.getLogger("asr_qe")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="ASR + QE demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

asr = llm = None
asr_lock = asyncio.Lock()   # whisper on CPU: one utterance at a time


@app.on_event("startup")
async def _load_models():
    global asr, llm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("loading ASR backend=%s model=%s ...", C.ASR_BACKEND, C.ASR_MODEL)
    asr = await asyncio.to_thread(build_asr)
    llm = build_llm()
    log.info("ready: asr=%s llm=%s (%s)", type(asr).__name__, C.LLM_BACKEND, C.LLM_MODEL)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


class Session:
    """Per-connection state: VAD segmenter, two chat histories, background tasks."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.seg = UtteranceSegmenter()
        self.history = {"plain": [], "qe": []}
        self.llm_locks = {"plain": asyncio.Lock(), "qe": asyncio.Lock()}
        self.tasks: set[asyncio.Task] = set()
        self.next_id = 0
        self.frames_since_vad_msg = 0

    async def send(self, **msg):
        try:
            await self.ws.send_json(msg)
        except Exception:      # client went away; the receive loop will notice
            pass

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    # ---- audio in -------------------------------------------------------------------------------------
    async def on_audio(self, data: bytes):
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        for ev, payload in self.seg.feed(pcm):
            if ev == "prob":
                self.frames_since_vad_msg += 1
                if self.frames_since_vad_msg >= 3:          # ~every 100 ms
                    self.frames_since_vad_msg = 0
                    await self.send(type="vad", prob=round(payload, 3), speaking=self.seg.speaking)
            elif ev == "start":
                await self.send(type="vad", prob=round(self.seg.last_prob, 3), speaking=True)
            elif ev == "end":
                self.spawn(self.process(payload))

    async def flush(self):
        utt = self.seg.flush()
        if utt is not None:
            self.spawn(self.process(utt))

    # ---- utterance -> ASR -> QE -> 2x LLM --------------------------------------------------------------
    async def process(self, utt):
        uid = self.next_id
        self.next_id += 1
        await self.send(type="utterance", id=uid, duration=round(utt.duration, 2))
        try:
            async with asr_lock:
                tr = await asyncio.to_thread(asr.transcribe, utt.audio)
        except Exception as e:
            log.exception("ASR failed")
            await self.send(type="dropped", id=uid, reason=f"ASR error: {e}")
            return
        if not tr.text.strip():
            await self.send(type="dropped", id=uid, reason="no words recognized")
            return
        qe = estimate(tr, utt)
        log.info("utt %d (%.1fs) [%s %.2f]: %s", uid, utt.duration, qe.label, qe.score, tr.text)
        await self.send(type="transcript", id=uid, asr=tr.to_dict(), qe=qe.to_dict())
        await asyncio.gather(
            self.reply("plain", uid, tr.text, C.ASSISTANT_PROMPT),
            self.reply("qe", uid, f"{tr.text}\n\n{qe.context}", f"{C.ASSISTANT_PROMPT}\n\n{QE_GUIDANCE}"),
        )

    async def reply(self, side: str, uid: int, user_text: str, system: str):
        async with self.llm_locks[side]:       # keep each side's history in order
            hist = self.history[side]
            hist.append({"role": "user", "content": user_text})
            out = []
            try:
                async for delta in llm.stream(system, hist):
                    out.append(delta)
                    await self.send(type="llm_delta", id=uid, side=side, text=delta)
            except Exception as e:
                log.exception("LLM failed")
                await self.send(type="error", message=f"LLM error ({side}): {e}")
                hist.pop()
                return
            text = "".join(out)
            hist.append({"role": "assistant", "content": text})
            await self.send(type="llm_done", id=uid, side=side, text=text)

    async def close(self):
        for t in self.tasks:
            t.cancel()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    s = Session(ws)
    await s.send(type="ready", asr=C.ASR_BACKEND, asr_model=C.ASR_MODEL, llm=C.LLM_BACKEND, llm_model=C.LLM_MODEL,
                 sample_rate=C.SAMPLE_RATE)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await s.on_audio(msg["bytes"])
            elif msg.get("text"):
                kind = json.loads(msg["text"]).get("type")
                if kind == "flush":
                    await s.flush()
                elif kind == "reset":
                    s.history = {"plain": [], "qe": []}
                    s.seg.reset()
    except WebSocketDisconnect:
        pass
    finally:
        await s.close()
