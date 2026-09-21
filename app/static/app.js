// Browser side: capture mic (or a file), stream int16 PCM over a WebSocket, render the two chats.
const $ = (s, el = document) => el.querySelector(s);
const chats = { plain: $("#plain .chat"), qe: $("#qe .chat") };
const stateEl = $("#state"), levelEl = $("#level"), micBtn = $("#mic");
const turns = new Map();          // utterance id -> {plain: {user, bot}, qe: {user, bot}}
let ws, ctx, workletNode, mediaStream, listening = false, fileTimer = null, busy = 0;

// ---- websocket ------------------------------------------------------------------------------------
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setState("disconnected"); setTimeout(connect, 1500); };
}
const send = (obj) => ws && ws.readyState === 1 && ws.send(obj instanceof ArrayBuffer ? obj : JSON.stringify(obj));

function sendPcm(f32) {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
  send(i16.buffer);
}

// ---- state pill -----------------------------------------------------------------------------------
function setState(s) {
  stateEl.textContent = s; stateEl.className = "pill " + s;
  micBtn.disabled = $("#reset").disabled = s === "disconnected";
}
function refreshState() {
  if (busy > 0) return;
  setState(listening || fileTimer ? "listening" : "idle");
}

// ---- server events --------------------------------------------------------------------------------
function handle(m) {
  switch (m.type) {
    case "ready":
      $("#backend").textContent = `asr: ${m.asr}${m.asr === "whisper" ? " (" + m.asr_model + ")" : ""} · llm: ${m.llm}${m.llm === "anthropic" ? " (" + m.llm_model + ")" : ""}`;
      refreshState(); break;
    case "vad":
      levelEl.style.width = Math.round(m.prob * 100) + "%";
      if (busy === 0) setState(m.speaking ? "speaking" : (listening || fileTimer ? "listening" : "idle"));
      break;
    case "utterance": {
      busy++; setState("transcribing");
      const t = { plain: {}, qe: {} };
      for (const side of ["plain", "qe"]) {
        t[side].user = bubble(side, "user pending", `… transcribing ${m.duration}s of speech`);
        t[side].user.dataset.utt = m.id + 1; t[side].user.dataset.dur = m.duration;
      }
      turns.set(m.id, t); break;
    }
    case "dropped": {
      const t = turns.get(m.id); busy--; refreshState();
      for (const side of ["plain", "qe"]) { t[side].user.className = "msg user pending"; t[side].user.textContent = `(dropped: ${m.reason})`; }
      break;
    }
    case "transcript": {
      const t = turns.get(m.id); setState("thinking");
      t.plain.user.className = "msg user"; t.plain.user.textContent = m.asr.text;
      t.qe.user.className = "msg user"; t.qe.user.replaceChildren(...renderQe(m.asr, m.qe));
      for (const side of ["plain", "qe"]) t[side].bot = bubble(side, "bot streaming", "");
      break;
    }
    case "llm_delta": { const b = turns.get(m.id)[m.side].bot; b.textContent += m.text; scroll(m.side); break; }
    case "llm_done": {
      const t = turns.get(m.id); t[m.side].bot.classList.remove("streaming"); t[m.side].done = true;
      if (t.plain.done && t.qe.done) { busy--; refreshState(); }
      break;
    }
    case "error": bubble("qe", "bot error", m.message); bubble("plain", "bot error", m.message); break;
  }
}

function bubble(side, cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls; el.textContent = text;
  chats[side].appendChild(el); scroll(side); return el;
}
function scroll(side) { chats[side].scrollTop = chats[side].scrollHeight; }

// The QE-side user bubble: words coloured by confidence, ⟂ where speech had no transcript, context for the LLM.
function renderQe(asr, qe) {
  const frag = [];
  const gapAfter = new Map(qe.gaps.map(g => [g.after, g]));
  const gapMark = (g) => { const s = document.createElement("span"); s.className = "gap"; s.textContent = `${g.seconds}s`; s.title = `possible missing words: ~${g.seconds}s of speech, no transcript`; return s; };
  if (gapAfter.has(-1)) frag.push(gapMark(gapAfter.get(-1)));
  qe.words.forEach((w, i) => {
    const s = document.createElement("span"); s.className = "w " + w.level; s.textContent = w.text; s.title = `confidence ${w.prob}`;
    frag.push(s, document.createTextNode(" "));
    if (gapAfter.has(i)) frag.push(gapMark(gapAfter.get(i)));
  });
  if (!qe.words.length) frag.push(document.createTextNode(asr.text));
  const badge = document.createElement("span"); badge.className = "badge " + qe.label; badge.textContent = `QE ${qe.score.toFixed(2)} ${qe.label}`;
  const meta = document.createElement("span"); meta.className = "meta";
  meta.textContent = `${qe.words.filter(w => w.level === "low").length} low-conf · ${qe.gaps.length} gap(s) · ${qe.notes.length} note(s) · lang ${asr.language}`;
  const foot = document.createElement("div"); foot.className = "qe-foot"; foot.append(badge, meta);
  const det = document.createElement("details");
  det.innerHTML = `<summary>context sent to LLM</summary><pre></pre>`; $("pre", det).textContent = qe.context;
  frag.push(foot, det);
  return frag;
}

// ---- microphone -----------------------------------------------------------------------------------
async function startMic() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  ctx = new AudioContext({ sampleRate: 16000 });        // browsers that ignore this get resampled in the worklet
  await ctx.audioWorklet.addModule("/static/worklet.js");
  const src = ctx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(ctx, "pcm-capture");
  workletNode.port.onmessage = (e) => sendPcm(e.data);
  const mute = ctx.createGain(); mute.gain.value = 0;    // graph must reach the destination to be pulled; keep it silent
  src.connect(workletNode); workletNode.connect(mute); mute.connect(ctx.destination);
  await ctx.resume();
  listening = true; micBtn.classList.add("on"); micBtn.setAttribute("aria-pressed", "true"); refreshState();
}
function stopMic() {
  mediaStream?.getTracks().forEach(t => t.stop()); ctx?.close();
  listening = false; micBtn.classList.remove("on"); micBtn.setAttribute("aria-pressed", "false");
  send({ type: "flush" }); levelEl.style.width = "0"; refreshState();
}
micBtn.onclick = () => (listening ? stopMic() : startMic().catch(err => handle({ type: "error", message: `Microphone error: ${err.message} (needs https or localhost)` })));

// ---- file playback: decode -> 16 kHz mono -> stream in real time through the same VAD path ------------
$("#file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const raw = await new AudioContext().decodeAudioData(await f.arrayBuffer());
  const off = new OfflineAudioContext(1, Math.ceil(raw.duration * 16000), 16000);
  const s = off.createBufferSource(); s.buffer = raw; s.connect(off.destination); s.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  clearInterval(fileTimer);
  let pos = 0; const step = 1600;                         // 100 ms every 100 ms
  fileTimer = setInterval(() => {
    if (pos >= pcm.length) { clearInterval(fileTimer); fileTimer = null; send({ type: "flush" }); levelEl.style.width = "0"; refreshState(); return; }
    sendPcm(pcm.subarray(pos, pos + step)); pos += step;
  }, 100);
  refreshState(); e.target.value = "";
};

$("#reset").onclick = () => { send({ type: "reset" }); turns.clear(); chats.plain.replaceChildren(); chats.qe.replaceChildren(); busy = 0; refreshState(); };

connect();
