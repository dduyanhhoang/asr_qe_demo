# ASR + QE demo

Always-on microphone → **Silero VAD** → **Whisper ASR** → **QE** (quality estimation) → **LLM**, shown as two
side-by-side chats so you can see what QE changes:

| left panel | right panel |
|---|---|
| transcript → LLM | transcript **+ QE context** → LLM |

The QE context tells the LLM which words are probably wrong, where words are probably missing, and whether the
segment looks hallucinated, so it can infer or ask instead of acting on a bad transcript.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e .                                         # or: uv pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...                      # optional; without it a mock LLM is used
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>, click **Start listening**, talk. The mic needs `localhost` or HTTPS.
The first run downloads the Whisper model (`small` by default, ~500 MB).

No microphone or no model handy? **Play audio file** streams any audio file through the same VAD path, and

```bash
ASR_BACKEND=mock LLM_BACKEND=mock uvicorn app.main:app --port 8000
python scripts/stream_file.py some_speech.wav          # headless client, prints every event
```

runs the whole thing offline with fake ASR output (random shaky words, so the QE side has something to show).

## How it works

```
browser                          server (FastAPI, one WebSocket per tab)
──────────                       ─────────────────────────────────────────────────────────
mic → AudioWorklet ─int16 PCM──▶ UtteranceSegmenter (Silero VAD, onnxruntime)
   16 kHz mono                       │ speech start / 700 ms silence → Utterance (audio + per-frame probs)
                                     ▼
                                 WhisperASR (faster-whisper, word timestamps + word probabilities)
                                     ▼
                                 qe.estimate(transcript, utterance) → QEResult
                                     │  words with confidence level, gaps, notes, `context` text
                                     ├──▶ LLM(system, history + transcript)                → left chat
                                     └──▶ LLM(system + QE guidance, history + transcript
                                                                    + QE context)          → right chat
                                 streamed back as JSON events
```

* `app/vad.py` – Silero VAD v5 via ONNX (no torch) and the utterance state machine (pre-roll, min silence,
  min speech, max length). `app/models/silero_vad.onnx` is the model file from the `silero-vad` package (MIT).
* `app/asr.py` – `WhisperASR` (faster-whisper) and `MockASR`. Both return `Transcript(text, words[text,start,end,prob], …)`.
* `app/qe.py` – heuristic QE with the interface a trained QE model would have. Signals today:
  * per-word decoder probability → `ok / mid / low`
  * gaps between words where the VAD saw speech but ASR produced nothing → *possible missing words*
  * segment stats (compression ratio → hallucination, no-speech prob, avg logprob), forced cut, low words/sec
  * everything is rendered into a `[ASR quality estimation]` block for the LLM plus JSON for the UI
* `app/llm.py` – `AnthropicLLM` (streaming, `claude-opus-5`, low effort) and `MockLLM`.
* `app/main.py` – WebSocket protocol (documented at the top of the file) and the per-connection pipeline.
* `app/static/` – plain HTML/CSS/JS, no build step.

## Knobs (env vars)

| var | default | |
|---|---|---|
| `ASR_BACKEND` | `whisper` | `whisper` or `mock` |
| `ASR_MODEL` | `small` | any faster-whisper size / CTranslate2 path (`medium`, `large-v3`, …) |
| `ASR_LANGUAGE` | auto | e.g. `vi` – fixing it speeds up short utterances |
| `ASR_DEVICE`, `ASR_COMPUTE_TYPE` | `cpu`, `int8` | `cuda` + `float16` on a GPU |
| `LLM_BACKEND` | `anthropic` if `ANTHROPIC_API_KEY` is set, else `mock` | |
| `LLM_MODEL`, `LLM_EFFORT` | `claude-opus-5`, `low` | |
| `ASSISTANT_PROMPT` | generic voice assistant | persona for both panels |
| `VAD_MIN_SILENCE_MS` | `700` | silence that ends an utterance |
| `VAD_START_THRESHOLD` / `VAD_END_THRESHOLD` | `0.5` / `0.35` | |
| `QE_LOW_CONF` / `QE_MID_CONF` | `0.5` / `0.75` | word confidence buckets |
| `QE_GAP_S` | `0.4` | min untranscribed speech gap flagged as missing words |

## Plugging in a real QE model

Replace the body of `estimate()` in `app/qe.py`; keep returning a `QEResult` (`words`, `gaps`, `notes`, `score`,
`context`). The UI and the LLM prompt only depend on that shape.
