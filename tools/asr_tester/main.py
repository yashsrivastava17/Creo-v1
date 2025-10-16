from __future__ import annotations

import asyncio
import os
import time
import wave
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

try:
    from jarvis.transcription import CheetahStream, VoskStream
except Exception:
    CheetahStream = None  # type: ignore
    VoskStream = None  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


def _load_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if load_dotenv is not None:
        if env_path.exists():
            load_dotenv(env_path, override=False)
    else:
        if env_path.exists():
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip().strip('"').strip("'")


_device_env = _env("AUDIO_INPUT_DEVICE")
if _device_env is None:
    DEVICE: str | int | None = 0
else:
    DEVICE = int(_device_env) if _device_env.isdigit() else _device_env

SAMPLE_RATE = int(_env("REALTIME_STT_SAMPLE_RATE", "16000") or 16000)
FRAME_MS = int(_env("REALTIME_STT_FRAME_MS", "30") or 30)
ENERGY_THRESHOLD = float(_env("AUDIO_ENERGY_THRESHOLD", "150") or 150.0)

CHEETAH_KEY = _env("CHEETAH_ACCESS_KEY")
CHEETAH_MODEL = _env("CHEETAH_MODEL_PATH")
CHEETAH_ENDPOINT = float(_env("CHEETAH_ENDPOINT_SEC", "0.6") or 0.6)
CHEETAH_AUTO_PUNCT = (_env("CHEETAH_AUTO_PUNCT", "true") or "true").lower() == "true"

VOSK_EN_IN = _env("VOSK_MODEL_PATH_EN_IN") or _env("VOSK_MODEL_PATH")
VOSK_HI = _env("VOSK_MODEL_PATH_HI")


app = FastAPI(title="ASR Tester")

SESSION_DIR = (Path(__file__).resolve().parent / "sessions")
SESSION_DIR.mkdir(exist_ok=True)

INDEX_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>ASR Tester</title>
<style>body{font-family:sans-serif;background:#0b0b0b;color:#f3f3f3;margin:0;padding:20px}
.row{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.btn{background:#14b8a6;border:0;color:#0b0b0b;padding:8px 12px;border-radius:8px;cursor:pointer}
.btn[disabled]{opacity:.6;cursor:not-allowed}
.meter{width:400px;max-width:80vw;height:10px;background:#1e293b;border-radius:999px;overflow:hidden}
.fill{height:100%;width:0;background:linear-gradient(90deg,#22d3ee,#14b8a6);transition:width .08s}
.box{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px;margin-top:16px}
.mono{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
canvas{display:block;background:#0f172a;border-radius:8px;border:1px solid #1f2937}
</style></head>
<body>
<div class=row>
  <button id=btnStart class=btn onclick="doStart()">Start</button>
  <button id=btnStop class=btn onclick="doStop()">Stop</button>
  <button id=btnTranscribe class=btn onclick="doTranscribe()">Transcribe</button>
  <select id=engine onchange="toggle(this.value)" class=btn>
    <option value="cheetah">cheetah</option>
    <option value="vosk-en-in">vosk-en-in</option>
    <option value="vosk-hi">vosk-hi</option>
  </select>
  <div>CPU: <span id=cpu>--</span>%</div>
  <div>Status: <span id=status>idle</span></div>
</div>
<div class="meter"><div id=fill class=fill></div></div>
<div class=box>
  <div>Engine: <span id=eng>--</span></div>
  <div>Transcript: <span id=txt class=mono></span></div>
</div>
<div class=box>
  <div>Waveform</div>
  <canvas id=wave width=640 height=120></canvas>
</div>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e)=>{
  const d = JSON.parse(e.data);
  document.getElementById('cpu').textContent = (d.cpu||0).toFixed(1);
  document.getElementById('eng').textContent = d.engine||'--';
  document.getElementById('txt').textContent = d.text||'';
  const lvl = Math.max(0, Math.min((d.audio_level||0)/2000, 1));
  document.getElementById('fill').style.width = `${Math.round(Math.pow(lvl,0.6)*100)}%`;
  if (d.engine) document.getElementById('engine').value = d.engine;
  if (d.state) document.getElementById('status').textContent = d.state;
  if (Array.isArray(d.levels)) drawWave(d.levels);
  if (Array.isArray(d.events)) updateEvents(d.events);
  if (d.audio_url) showPlayer(d.audio_url);
};
async function post(p){ const r = await fetch(p,{method:'POST'}); if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function toggle(v){ setStatus('switching'); await post(`/toggle?engine=${encodeURIComponent(v)}`); }
function setBusy(btn, on){ btn.disabled=on; }
function setStatus(s){ document.getElementById('status').textContent = s; }
async function doStart(){ const btn=document.getElementById('btnStart'); setBusy(btn,true); setStatus('starting'); try{ const res=await post('/start'); if(res.state) setStatus(res.state); else setStatus('recording'); } finally { setBusy(btn,false);} }
async function doStop(){ const btn=document.getElementById('btnStop'); setBusy(btn,true); setStatus('stopping'); try{ const res=await post('/stop'); if(res.state) setStatus(res.state); else setStatus('stopped'); } finally { setBusy(btn,false);} }
async function doTranscribe(){ const btn=document.getElementById('btnTranscribe'); setBusy(btn,true); setStatus('transcribing'); try{ const res=await post('/transcribe'); if(res.state) setStatus(res.state); } catch(err){ console.error(err); setStatus('error'); } finally { setBusy(btn,false);} }

function drawWave(levels){
  const c=document.getElementById('wave'); const ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  const h=c.height, w=c.width; const n=levels.length||1; const step=w/Math.max(n,1);
  ctx.strokeStyle='#22d3ee'; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<n;i++){
    const v = Math.max(0, Math.min(levels[i]/2000,1));
    const y = h - (Math.pow(v,0.6)*h);
    const x = i*step;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function updateEvents(events){
  let list = document.getElementById('event-log');
  if(!list){
    const box=document.createElement('div'); box.className='box';
    const title=document.createElement('div'); title.textContent='Events';
    list=document.createElement('div'); list.id='event-log'; list.style.fontSize='0.75rem'; list.style.maxHeight='180px'; list.style.overflowY='auto'; list.className='mono';
    box.appendChild(title); box.appendChild(list);
    document.body.appendChild(box);
  }
  list.innerHTML=events.map(e=>`<div>${e}</div>`).join('');
  list.scrollTop=list.scrollHeight;
}
function showPlayer(url){
  let box=document.getElementById('audio-box');
  if(!box){
    box=document.createElement('div'); box.id='audio-box'; box.className='box';
    const title=document.createElement('div'); title.textContent='Last Recording';
    const player=document.createElement('audio'); player.id='audio-player'; player.controls=true; player.style.width='100%';
    box.appendChild(title); box.appendChild(player);
    document.body.appendChild(box);
  }
  if(!url){
    const player=document.getElementById('audio-player');
    if(player){ player.removeAttribute('src'); player.load(); }
    return;
  }
  const player=document.getElementById('audio-player');
  if(player && player.src !== `${location.origin}${url}`){
    player.src = url;
    player.load();
  }
}
</script>
</body></html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = (SESSION_DIR / filename).resolve()
    if not path.is_file() or path.parent != SESSION_DIR.resolve():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="audio/wav")


class SimpleCapture:
    def __init__(self, samplerate: int, frame_ms: int, device: str | int | None) -> None:
        self.samplerate = samplerate
        self.frame_samples = int(samplerate * frame_ms / 1000)
        self.device = device
        self.q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        if self.stream:
            return
        self.q = asyncio.Queue()

        def cb(indata, frames, time_info, status):  # type: ignore[override]
            pcm = (indata.copy() * 32767).astype(np.int16).tobytes()
            try:
                self.q.put_nowait(pcm)
            except Exception:
                pass

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            blocksize=self.frame_samples,
            dtype="float32",
            callback=cb,
            device=self.device,
        )
        self.stream.start()

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            item = await self.q.get()
            if item is None:
                break
            yield item

    async def stop(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        await self.q.put(None)


class Tester:
    def __init__(self) -> None:
        self.engine_name: str = os.getenv("TRANSCRIPTION_ENGINE", "cheetah")
        self.transcriber: Any | None = None
        self.capture = SimpleCapture(SAMPLE_RATE, FRAME_MS, DEVICE if DEVICE is not None else None)
        self.audio_task: asyncio.Task | None = None
        self.stream_task: asyncio.Task | None = None
        self.clients: set[WebSocket] = set()
        self.current_text = ""
        self.audio_level = 0.0
        self.levels: deque[float] = deque(maxlen=256)
        self.state: str = "idle"
        self.events: deque[str] = deque(maxlen=50)
        self.buffer = bytearray()
        self.session_path: Path | None = None
        self.session_url: str | None = None
        self.transcribe_task: asyncio.Task | None = None

    def _build(self, engine_name: str) -> tuple[str, Any]:
        if engine_name == "cheetah":
            if CheetahStream is None:
                raise RuntimeError("pvcheetah not available")
            if not CHEETAH_KEY:
                raise RuntimeError("CHEETAH_ACCESS_KEY not set")
            tr = CheetahStream(access_key=CHEETAH_KEY, model_path=CHEETAH_MODEL, endpoint_sec=CHEETAH_ENDPOINT, auto_punct=CHEETAH_AUTO_PUNCT)
            return engine_name, tr
        elif engine_name == "vosk-hi":
            if VoskStream is None:
                raise RuntimeError("vosk not available")
            if not VOSK_HI:
                raise RuntimeError("VOSK_MODEL_PATH_HI not set")
            return engine_name, VoskStream(model_path=VOSK_HI, sample_rate=SAMPLE_RATE)
        else:
            if VoskStream is None:
                raise RuntimeError("vosk not available")
            if not VOSK_EN_IN:
                raise RuntimeError("VOSK_MODEL_PATH_EN_IN not set")
            return "vosk-en-in", VoskStream(model_path=VOSK_EN_IN, sample_rate=SAMPLE_RATE)

    async def _broadcast(self, extra: dict[str, Any] | None = None) -> None:
        if not self.clients:
            return
        payload = {
            "engine": self.engine_name,
            "text": self.current_text,
            "audio_level": self.audio_level,
            "cpu": psutil.cpu_percent(interval=None),
            "state": self.state,
            "levels": list(self.levels),
            "events": list(self.events),
            "audio_url": self.session_url or "",
        }
        if extra:
            payload.update(extra)
        await asyncio.gather(*[c.send_json(payload) for c in list(self.clients)], return_exceptions=True)

    async def start(self) -> None:
        if self.audio_task or self.stream_task:
            return
        self._event("starting capture")
        self.state = "starting"
        self.levels.clear()
        self.buffer.clear()
        self._clear_session()
        await self._broadcast()
        try:
            self.engine_name, self.transcriber = self._build(self.engine_name)
        except Exception as exc:
            self.state = "error"
            self._event(f"error: {exc}")
            await self._broadcast()
            raise
        self.capture.start()
        self.audio_task = asyncio.create_task(self._pump_audio(), name="tester-audio")
        self.stream_task = asyncio.create_task(self._pump_stream(), name="tester-stream")
        self.state = "recording"
        self._event("recording started")
        await self._broadcast()

    async def stop(self) -> None:
        self._event("stopping capture")
        self.state = "stopping"
        await self._broadcast()
        if self.stream_task:
            try:
                await self.transcriber.enqueue_audio(b"", time.time(), False, force=True)
            except Exception:
                pass
        if self.audio_task:
            await self.capture.stop()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        else:
            await self.capture.stop()
        if self.stream_task:
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        self.stream_task = None
        self.audio_task = None
        if self.transcriber:
            try:
                await self.transcriber.close()
            except Exception:
                pass
            self.transcriber = None
        await self._finalise_recording()
        self.state = "stopped"
        self.audio_level = 0.0
        self._event("recording stopped")
        await self._broadcast({"text": "", "audio_level": 0})

    async def toggle(self, engine: str) -> None:
        self._event(f"switching engine -> {engine}")
        await self.stop()
        self.engine_name = engine
        await self.start()
        await self._broadcast()

    async def _pump_audio(self) -> None:
        assert self.transcriber is not None
        async for pcm in self.capture.frames():
            self.audio_level = float(np.abs(np.frombuffer(pcm, dtype=np.int16)).mean())
            self.levels.append(self.audio_level)
            self.buffer.extend(pcm)
            vad = self.audio_level > ENERGY_THRESHOLD
            await self.transcriber.enqueue_audio(pcm, time.time(), vad)
            if self.clients:
                await self._broadcast()

    async def _pump_stream(self) -> None:
        assert self.transcriber is not None
        async for chunk in self.transcriber.stream():
            text = chunk.text.strip()
            if text:
                self.current_text = text
                # Briefly show processing when a final comes in
                if chunk.is_final:
                    self.state = "processing"
                    self._event(f"final: {text}")
                    await self._broadcast({"text": text})
                    # yield back to loop then flip to recording again
                    await asyncio.sleep(0)
                    self.state = "recording"
                else:
                    self._event(f"partial: {text}")
                    await self._broadcast({"text": text})

    def _event(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(f"[tester] {message}")
        self.events.appendleft(entry)
        if len(self.events) > self.events.maxlen:
            self.events.pop()

    async def _finalise_recording(self) -> None:
        if not self.buffer:
            self._clear_session()
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"session-{timestamp}.wav"
        path = SESSION_DIR / filename
        samples = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32)
        if samples.size:
            samples -= samples.mean()
            cutoff = ENERGY_THRESHOLD * 0.8
            samples[np.abs(samples) < cutoff] = 0
        cleaned = samples.clip(-32768, 32767).astype(np.int16).tobytes()
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(cleaned)
        self.session_path = path
        self.session_url = f"/audio/{filename}"
        self._event(f"saved audio {filename}")
        await self._broadcast({"audio_url": self.session_url})

    def _clear_session(self) -> None:
        if self.session_path and self.session_path.exists():
            try:
                self.session_path.unlink()
            except OSError:
                pass
        self.session_path = None
        self.session_url = ""

    async def transcribe_last(self, engine: str | None = None) -> str:
        if not self.session_path or not self.session_path.exists():
            raise RuntimeError("No recording available. Record something first.")
        engine_to_use = engine or self.engine_name
        self.state = "transcribing"
        self._event(f"transcribe start ({engine_to_use})")
        await self._broadcast()
        name, transcriber = self._build(engine_to_use)
        result = ""
        try:
            with wave.open(str(self.session_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            frame_bytes = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
            for offset in range(0, len(raw), frame_bytes):
                chunk = raw[offset : offset + frame_bytes]
                if not chunk:
                    continue
                energy = float(np.abs(np.frombuffer(chunk, dtype=np.int16)).mean())
                await transcriber.enqueue_audio(chunk, time.time(), energy > ENERGY_THRESHOLD)
            await transcriber.enqueue_audio(b"", time.time(), False, force=True)

            async for chunk in transcriber.stream():
                text = chunk.text.strip()
                if text:
                    result = text
                    kind = "final" if chunk.is_final else "partial"
                    self._event(f"transcribe {kind}: {text}")
                    await self._broadcast({"text": text})
                if chunk.is_final:
                    break
        finally:
            try:
                await transcriber.close()
            except Exception:
                pass

        self.state = "stopped"
        self.current_text = result
        self._event("transcribe complete")
        await self._broadcast({"text": result})
        return result


tester = Tester()


@app.post("/start")
async def start() -> dict[str, str]:
    try:
        await tester.start()
    except Exception as exc:
        tester.state = "error"
        tester._event(f"error: {exc}")
        await tester._broadcast()
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc)})
    return {"status": "ok", "state": tester.state}


@app.post("/stop")
async def stop() -> dict[str, str]:
    await tester.stop()
    return {"status": "ok", "state": tester.state}


@app.post("/toggle")
async def toggle(engine: str) -> dict[str, str]:
    try:
        await tester.toggle(engine)
    except Exception as exc:
        tester.state = "error"
        tester._event(f"error: {exc}")
        await tester._broadcast()
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc)})
    return {"status": "ok", "engine": tester.engine_name, "state": tester.state}


@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    await ws.accept()
    tester.clients.add(ws)
    try:
        await tester._broadcast()
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        tester.clients.discard(ws)


@app.post("/transcribe")
async def transcribe(engine: str | None = None) -> dict[str, str]:
    try:
        text = await tester.transcribe_last(engine)
    except Exception as exc:
        tester.state = "error"
        tester._event(f"error: {exc}")
        await tester._broadcast()
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc)})
    return {"status": "ok", "text": text, "state": tester.state}
