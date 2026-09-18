"""ASR backends. Both return a Transcript with word-level timing + confidence, which is what QE consumes."""
from dataclasses import dataclass, field, asdict
import random

import numpy as np

from . import config as C


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = "unknown"
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0

    def to_dict(self):
        return asdict(self)


class WhisperASR:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(C.ASR_MODEL, device=C.ASR_DEVICE, compute_type=C.ASR_COMPUTE_TYPE)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        segments, info = self.model.transcribe(
            audio,
            language=C.ASR_LANGUAGE,
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,          # we already segmented with Silero
            condition_on_previous_text=False,
        )
        words, texts, logps, nsp, crs = [], [], [], [], []
        for seg in segments:
            texts.append(seg.text.strip())
            logps.append(seg.avg_logprob)
            nsp.append(seg.no_speech_prob)
            crs.append(seg.compression_ratio)
            for w in seg.words or []:
                words.append(Word(w.word.strip(), round(w.start, 2), round(w.end, 2), round(w.probability, 3)))
        return Transcript(
            text=" ".join(t for t in texts if t),
            words=words,
            language=info.language,
            avg_logprob=float(np.mean(logps)) if logps else -1.0,
            no_speech_prob=float(max(nsp)) if nsp else 1.0,
            compression_ratio=float(max(crs)) if crs else 1.0,
        )


class MockASR:
    """No model download needed. Produces a plausible transcript with deliberately shaky words so the QE side has
    something to show. Handy for UI work and for CI-less containers."""

    SENTENCES = [
        "tôi muốn đặt vé đi Đà Nẵng vào thứ sáu tuần sau",
        "cho tôi hỏi số dư tài khoản tiết kiệm của tôi",
        "nhắc tôi gọi cho anh Minh lúc ba giờ chiều",
        "I would like to book a table for four people tonight",
        "what is the weather going to be like tomorrow morning",
    ]

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        dur = len(audio) / C.SAMPLE_RATE
        toks = self.rng.choice(self.SENTENCES).split()
        n = max(2, min(len(toks), int(dur * 2.2)))
        toks = toks[:n]
        words, t = [], 0.15
        step = max(0.05, (dur - 0.4) / n)
        for i, tok in enumerate(toks):
            p = self.rng.uniform(0.85, 0.99)
            if self.rng.random() < 0.25:
                p = self.rng.uniform(0.2, 0.6)            # a shaky word
            gap = step * 2.2 if self.rng.random() < 0.12 else step * 0.1   # occasionally a gap (dropped word)
            words.append(Word(tok, round(t, 2), round(t + step * 0.8, 2), round(p, 3)))
            t += step * 0.8 + gap
        return Transcript(
            text=" ".join(w.text for w in words),
            words=words,
            language="vi" if any(ord(c) > 127 for c in " ".join(toks)) else "en",
            avg_logprob=float(np.log(np.mean([w.prob for w in words]))),
            no_speech_prob=0.05,
            compression_ratio=1.2,
        )


def build_asr():
    return MockASR() if C.ASR_BACKEND == "mock" else WhisperASR()
