"""Silero VAD (ONNX, no torch) + a small state machine that cuts the mic stream into utterances."""
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from . import config as C

WINDOW = 512          # samples per VAD frame at 16 kHz (32 ms)
CONTEXT = 64          # Silero v5 prepends the last 64 samples of the previous frame


class SileroVAD:
    def __init__(self, path=C.VAD_MODEL_PATH):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """frame: float32[512] in [-1, 1]. Returns speech probability."""
        x = np.concatenate([self.context, frame]).astype(np.float32)[None, :]
        out, self.state = self.sess.run(
            None, {"input": x, "state": self.state, "sr": np.array(C.SAMPLE_RATE, dtype=np.int64)}
        )
        self.context = frame[-CONTEXT:]
        return float(out[0, 0])


@dataclass
class Utterance:
    audio: np.ndarray                 # float32 mono 16 kHz, includes pre-roll
    probs: list[float] = field(default_factory=list)   # per-frame speech prob, aligned to `audio`
    forced: bool = False              # cut by max length rather than silence

    @property
    def duration(self) -> float:
        return len(self.audio) / C.SAMPLE_RATE

    def speech_ratio(self, t0: float, t1: float) -> float:
        """Fraction of VAD frames in [t0, t1] seconds that were judged speech. Used by QE."""
        f0, f1 = int(t0 * C.SAMPLE_RATE / WINDOW), int(np.ceil(t1 * C.SAMPLE_RATE / WINDOW))
        seg = self.probs[max(f0, 0):max(f1, f0 + 1)]
        return float(np.mean([p > C.VAD_END_THRESHOLD for p in seg])) if seg else 0.0


class UtteranceSegmenter:
    """Feed 16 kHz float32 audio in any chunk size; get back finished utterances.

    speech starts  : prob > START_THRESHOLD
    speech ends    : prob < END_THRESHOLD for MIN_SILENCE_MS
    """

    def __init__(self, vad: SileroVAD | None = None):
        self.vad = vad or SileroVAD()
        self.pending = np.zeros(0, dtype=np.float32)
        self.pre_roll_frames = max(1, C.VAD_PRE_ROLL_MS * C.SAMPLE_RATE // 1000 // WINDOW)
        self.silence_frames_needed = C.VAD_MIN_SILENCE_MS * C.SAMPLE_RATE // 1000 // WINDOW
        self.max_frames = int(C.VAD_MAX_UTTERANCE_S * C.SAMPLE_RATE // WINDOW)
        self.min_speech_frames = C.VAD_MIN_SPEECH_MS * C.SAMPLE_RATE // 1000 // WINDOW
        self.reset()

    def reset(self):
        self.vad.reset()
        self.speaking = False
        self.ring: list[tuple[np.ndarray, float]] = []   # pre-roll ring buffer
        self.frames: list[np.ndarray] = []
        self.probs: list[float] = []
        self.silence_run = 0
        self.speech_frames = 0
        self.last_prob = 0.0

    def feed(self, audio: np.ndarray):
        """Yields (event, payload): ('prob', float) per frame, ('start', None), ('end', Utterance)."""
        self.pending = np.concatenate([self.pending, audio.astype(np.float32)])
        while len(self.pending) >= WINDOW:
            frame, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            prob = self.vad(frame)
            self.last_prob = prob
            yield "prob", prob

            if not self.speaking:
                self.ring.append((frame, prob))
                self.ring = self.ring[-self.pre_roll_frames:]
                if prob > C.VAD_START_THRESHOLD:
                    self.speaking = True
                    self.frames = [f for f, _ in self.ring]
                    self.probs = [p for _, p in self.ring]
                    self.silence_run, self.speech_frames = 0, 1
                    yield "start", None
                continue

            self.frames.append(frame)
            self.probs.append(prob)
            if prob < C.VAD_END_THRESHOLD:
                self.silence_run += 1
            else:
                self.silence_run = 0
                self.speech_frames += 1

            if self.silence_run >= self.silence_frames_needed or len(self.frames) >= self.max_frames:
                utt = self._finish(forced=len(self.frames) >= self.max_frames)
                if utt is not None:
                    yield "end", utt

    def flush(self) -> Utterance | None:
        """Close the in-progress utterance (end of a file / mic stopped)."""
        return self._finish(forced=True) if self.speaking else None

    def _finish(self, forced: bool) -> Utterance | None:
        frames, probs = self.frames, self.probs
        self.speaking, self.frames, self.probs, self.ring = False, [], [], []
        if self.speech_frames < self.min_speech_frames:
            return None
        if not forced:  # trim most of the trailing silence, keep ~200 ms
            keep = len(frames) - self.silence_run + 200 * C.SAMPLE_RATE // 1000 // WINDOW
            frames, probs = frames[:keep], probs[:keep]
        return Utterance(audio=np.concatenate(frames), probs=probs, forced=forced)
