// Captures mic audio, resamples to 16 kHz if the context runs at another rate, and posts Float32 chunks.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.buf = [];
    this.pos = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.buf.push(ch[i]);
    } else {                                   // linear resample
      for (; this.pos < ch.length - 1; this.pos += this.ratio) {
        const i = Math.floor(this.pos), f = this.pos - i;
        this.buf.push(ch[i] * (1 - f) + ch[i + 1] * f);
      }
      this.pos -= ch.length;
    }
    if (this.buf.length >= 1024) {             // ~64 ms per message
      this.port.postMessage(Float32Array.from(this.buf));
      this.buf = [];
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
