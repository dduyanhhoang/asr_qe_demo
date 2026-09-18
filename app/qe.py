"""Quality estimation for an ASR hypothesis.

This is a heuristic placeholder with the *interface* a real QE model would have:
    estimate(transcript, utterance) -> QEResult
Signals used today: per-word posterior from the decoder, timing gaps that the VAD says contained speech
(-> deletions), and segment-level stats (repetition -> hallucination, no-speech prob). Swap the body of
`estimate` for a trained QE model and the UI/LLM side keeps working.
"""
from dataclasses import dataclass, field, asdict

import numpy as np

from . import config as C
from .asr import Transcript
from .vad import Utterance


@dataclass
class QEWord:
    text: str
    start: float
    end: float
    prob: float
    level: str            # ok | mid | low


@dataclass
class Gap:
    after: int            # index of the word the gap follows; -1 = before the first word
    seconds: float
    speech_ratio: float


@dataclass
class QEResult:
    score: float
    label: str            # good | uncertain | poor
    words: list[QEWord] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    context: str = ""     # the block handed to the LLM

    def to_dict(self):
        return asdict(self)


def _level(p: float) -> str:
    return "low" if p < C.QE_LOW_CONF else "mid" if p < C.QE_MID_CONF else "ok"


def estimate(tr: Transcript, utt: Utterance) -> QEResult:
    words = [QEWord(w.text, w.start, w.end, w.prob, _level(w.prob)) for w in tr.words]
    gaps: list[Gap] = []
    notes: list[str] = []

    # --- deletions: stretches the VAD called speech but ASR produced nothing for -------------------
    bounds = [(-1, 0.0, words[0].start if words else utt.duration)]
    bounds += [(i, words[i].end, words[i + 1].start) for i in range(len(words) - 1)]
    if words:
        bounds.append((len(words) - 1, words[-1].end, utt.duration))
    for after, t0, t1 in bounds:
        if t1 - t0 >= C.QE_GAP_S:
            ratio = utt.speech_ratio(t0, t1)
            if ratio >= 0.5:
                gaps.append(Gap(after, round(t1 - t0, 2), round(ratio, 2)))

    # --- segment-level signals -----------------------------------------------------------------------
    if tr.compression_ratio > 2.4:
        notes.append("high repetition ratio: part of the transcript may be hallucinated/looped")
    if tr.no_speech_prob > 0.6:
        notes.append("decoder thinks this may not be speech at all")
    if tr.avg_logprob < -1.0:
        notes.append("low overall decoder likelihood")
    if utt.forced:
        notes.append("utterance was cut at the max length; the end may be truncated")
    if words and utt.duration > 1.5 and len(words) / utt.duration < 1.0:
        notes.append("very few words for the amount of speech; some words may be missing")

    # --- score --------------------------------------------------------------------------------------
    if words:
        score = float(np.exp(np.mean(np.log(np.clip([w.prob for w in words], 1e-3, 1)))))
    else:
        score = 0.0
    score -= 0.1 * len(gaps) + 0.15 * len(notes)
    score = round(float(np.clip(score, 0, 1)), 2)
    label = "good" if score >= 0.8 else "uncertain" if score >= 0.6 else "poor"

    res = QEResult(score, label, words, gaps, notes)
    res.context = _context(tr, res)
    return res


def _marked_transcript(words: list[QEWord], gaps: list[Gap]) -> str:
    gap_after = {g.after: g for g in gaps}
    out = []
    if -1 in gap_after:
        out.append("[missing?]")
    for i, w in enumerate(words):
        out.append(f"{{{w.text}|{w.prob:.2f}}}" if w.level != "ok" else w.text)
        if i in gap_after:
            out.append("[missing?]")
    return " ".join(out)


def _context(tr: Transcript, res: QEResult) -> str:
    lines = [
        "[ASR quality estimation]",
        f"Overall confidence: {res.score:.2f} ({res.label})",
        f"Transcript with confidence marks: \"{_marked_transcript(res.words, res.gaps)}\"",
        "  ({word|p} = word with confidence p, may be misrecognized; [missing?] = speech the ASR produced no words for)",
    ]
    low = [w for w in res.words if w.level == "low"]
    mid = [w for w in res.words if w.level == "mid"]
    if low:
        lines.append("Likely misrecognized: " + ", ".join(f"\"{w.text}\" ({w.prob:.2f})" for w in low))
    if mid:
        lines.append("Somewhat uncertain: " + ", ".join(f"\"{w.text}\" ({w.prob:.2f})" for w in mid))
    for g in res.gaps:
        where = "at the start" if g.after == -1 else (
            "at the end" if g.after == len(res.words) - 1 else
            f"between \"{res.words[g.after].text}\" and \"{res.words[g.after + 1].text}\"")
        lines.append(f"Possible missing words {where} (~{g.seconds:.1f}s of speech with no transcript)")
    for n in res.notes:
        lines.append(f"Note: {n}")
    return "\n".join(lines)


QE_GUIDANCE = (
    "The user message ends with an [ASR quality estimation] block produced by a quality-estimation model. "
    "Use it: treat marked words as possibly wrong and infer the intended word from context when it is obvious; "
    "where words may be missing, do not assume what was said. If a detail that matters for your answer "
    "(a name, number, date, place, amount, command) is low-confidence or missing, ask a short confirming question "
    "instead of guessing. Never mention the QE block or confidence numbers themselves; just act on them naturally."
)
