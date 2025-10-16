# Jarvis Voice Agent — Tech Handoff for Codex

This is a concise, implementation‑ready handoff for building the first working slice of the Jarvis voice agent. It focuses on:

- **Voice-first orchestration** with a single orchestrator service
- **Local small LLM** via **Ollama (Llama 3.2 small)**
- **Gemini API** for heavyweight/online tasks
- **Kokoro TTS** for responses
- **Whisper (medium)** for transcription, always-on mic, **wake-word gated** LLM turns
- **Persistent memory** (short-term + long-term) with embeddings and user profile/state

---

## 0) Success Criteria (Milestone M0 → M1)

**M0 Demo** (end-to-end):

- Agent listens continuously, transcribes locally with Whisper-medium.
- Wake word (e.g., “Jarvis”) detected; only then the last N seconds of transcript are **packaged and sent** to LLM.
- LLM routing: local (Ollama Llama 3.2) by default; escalates to Gemini for web/complex asks.
- Response synthesized via Kokoro.
- Persistent memory: user name, preferences, and 3 recent facts recalled in next turn.

**M1 Hardening**:

- Memory store with pgvector retrieval and scoped recall.
- Tooling: web search, calendar stub, and a simple RAG note tool.
- Telemetry: per-turn logs and spans.

---

## 1) High-level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Voice Orchestrator                        │
│  (finite-state machine + event loop + policy + router + memory I/O) │
└───────┬───────────────────────────────┬──────────────────────────────┘
        │                               │
        │ (pulls audio frames)          │ (publishes/consumes events)
        ▼                               ▼
┌──────────────┐     wake-word      ┌───────────────┐      embeddings
│  Audio I/O   │ ──►  Detector  ──► │ Transcription │ ──►  Memory Svc
│ (VAD, ring   │                   │ (Whisper med) │ ◄──  (Postgres+
│  buffer)     │ ◄──────────────────│ (streaming)   │      pgvector)
└─────┬────────┘  interim captions  └─┬─────────────┘
      │                               │ transcript chunks
      │                               ▼
      │                         ┌───────────────┐   tool calls
      │                         │  NLU Router   │ ──────────────► Tools Svc
      │                         │ (LLM Gateway) │ ◄──────────────  (web, fs,
      │                         └──────┬────────┘   calendar, etc.)
      │                                │
      │                         ┌──────▼────────┐
      │                         │  LLM Backends │
      │                         │  • Ollama     │
      │                         │  • Gemini     │
      │                         └──────┬────────┘
      │                                │ text
      │                                ▼
      │                         ┌──────────────┐
      │                         │   Kokoro     │
      │                         │    TTS       │
      │                         └──────┬───────┘
      │                                │ audio
      ▼                                ▼
  Playback                         User hears
```

**Eventing:** Recommend **Redis Streams** (simple, local dev) or **NATS** (easy fanout). For M0, in-process async queues are fine.

---

## 2) Technology Choices (locked for M0/M1)

- **Whisper**: `medium` model, streaming via chunks (500–1000 ms). Use VAD gating (Silero VAD) to pause transcription when silent.
- **Wake Word**: **openWakeWord** (lightweight, local) with custom keyword (“Jarvis” or user’s choice). Alternative: Porcupine (commercial).
- **Small LLM**: **Ollama** running **Llama 3.2** "small" (e.g., 3B-ish). Provide JSON-mode and function/tool calling schema.
- **Heavy LLM**: **Gemini API** (1.5/2.0) for web reasoning, multimodal, and long context tasks.
- **TTS**: **Kokoro** for final output; ensure low-latency streaming if available (chunked WAV/PCM playback).
- **Memory**: **Postgres + pgvector**; **Qdrant** is fine too. Use embeddings from local `all-MiniLM-L6-v2` or Gemini Embeddings when online.
- **Runtime**: Python (FastAPI) or TypeScript (Node). Below samples assume Python + FastAPI + asyncio.

---

## 3) Core Data Contracts

### 3.1 Events

```jsonc
// AudioFrame (internal)
{
  "ts": 1697045342.312,          // epoch seconds
  "pcm16le": "<bytes>",         // raw 16kHz mono
  "vad": true                    // voice activity present
}

// WakeWordHit
{
  "ts": 1697045343.100,
  "keyword": "jarvis",
  "confidence": 0.87,
  "buffer_ref_ms": 4000          // how much pre-roll audio to export
}

// TranscriptChunk
{
  "ts": 1697045343.800,
  "text": "what's the weather tomorrow",
  "range_ms": [12000, 18000],
  "is_final": true
}

// UserTurn (assembled on wake word)
{
  "turn_id": "uuid",
  "wake_word_ts": 1697045343.100,
  "prompt_text": "what\'s the weather tomorrow",
  "context": { "recent_memory": [...], "speaker_id": "u1" },
  "audio_refs": ["ringbuf:1697045340-1697045348"],
  "tools_allowed": ["web.search", "calendar.read"]
}

// ToolCall
{
  "turn_id": "uuid",
  "tool": "web.search",
  "args": { "q": "weather in mumbai tomorrow" }
}

// AssistantMessage
{
  "turn_id": "uuid",
  "text": "Tomorrow looks rainy in Mumbai. Take an umbrella.",
  "citations": [],
  "memory_writes": [
    { "type": "ephemeral", "data": {"topic": "weather-interest"}},
    { "type": "fact", "data": {"user_city": "Mumbai"}}
  ]
}
```

### 3.2 Memory Schema

**Tables** (Postgres):

- `users(id, handle, created_at, last_seen)`
- `sessions(id, user_id, started_at, metadata jsonb)`
- `memory_facts(id, user_id, kind, content jsonb, created_at, updated_at)`
- `memory_vectors(id, user_id, fact_id, embedding vector(384), dim int)`
- `turns(id, user_id, session_id, transcript text, assistant text, ts, labels text[])`
- `tool_results(id, turn_id, tool, args jsonb, result jsonb, ts)`

**Kinds**:

- `profile` (name, locale, tz, voice)
- `preference` (music likes, notification style)
- `fact` ("has dog named Luna")
- `ephemeral` (valid for 24h unless reinforced)

**Recall Policy**:

- On every `UserTurn`, recall:
  - 3 most recent `ephemeral` (decayed)
  - top‑k semantic neighbors to `prompt_text` from `memory_vectors` (k=5)
  - always include `profile`

**Write Policy**:

- After each `AssistantMessage`, parse model‑proposed `memory_writes` but gate them with rules (PII, duplicates, confidence).

---

## 4) Orchestrator (FSM) Outline

States: `IDLE` → `LISTENING` → `HOTWORD_HEARD` → `COMPOSING` → `SPEAKING` → back to `IDLE`.

Transitions:

- `LISTENING`: mic open; VAD on; Whisper streaming; emit `TranscriptChunk`.
- `HOTWORD_HEARD`: assemble trailing transcript window (e.g., last 6–10s) into a `UserTurn`.
- `COMPOSING`: route to LLM (Ollama default). If tool calls requested, execute through Tools Svc. Pull memory; attach to prompt.
- `SPEAKING`: send final text to Kokoro; stream audio out; optionally display captions.

```python
async def handle_turn(user_turn: UserTurn):
    ctx = memory.recall(user_turn)
    plan = router.decide(user_turn.prompt_text, ctx)

    if plan.provider == "ollama":
        resp = await llm_ollama.chat(plan.prompt, tools=plan.tools)
    else:
        resp = await llm_gemini.chat(plan.prompt, tools=plan.tools)

    tool_results = []
    for call in resp.tool_calls:
        r = await tools.invoke(call)
        tool_results.append(r)

    if tool_results:
        resp = await llm_ollama.chat(plan.followup_prompt(tool_results))

    gated_writes = memory.guard(resp.memory_writes)
    memory.write(gated_writes)

    await tts_kokoro.say(resp.text)
```

---

## 5) Prompting & Routing

**System Prompt (base):**

- You are Jarvis, a concise, helpful voice assistant. Prefer short, actionable answers. Offer to remember stable preferences.
- If a question requires browsing, tool use, or complex math, request appropriate tools.
- Only write `memory_writes` as JSON objects when you are confident a fact should persist.

**Routing Heuristics (M0):**

- Default to **Ollama**; switch to **Gemini** when:
  - needs web or long context (>8k)
  - code generation or complex reasoning flagged
  - user explicitly says “use online” or “check the web”

**Function/Tool Schema (JSON):**

- `web.search(q: string)`
- `calendar.read(range: {from,to})`
- `notes.search(query)` / `notes.upsert(id?, text)`

---

## 6) Always-on Transcription with Wake-word Gating

- Configure audio capture at 16kHz mono PCM, 20–30ms frames.
- Maintain a ring buffer of \~10–15 seconds.
- Run **openWakeWord** on frames; when hit, **export N seconds pre‑roll + next M seconds until pause** for context.
- Whisper runs continuously but **LLM is invoked only on wake-word** to avoid needless reasoning calls.
- Display interim captions locally (optional) but don’t store unless wake-word fires.

**Edge Cases**:

- False positives: require **double‑confirm** with immediate follow-up word (e.g., confidence threshold + VAD duration).
- Overlap with TTS playback: auto-mute mic when speaking to avoid feedback; re-open after 300ms hangover.

---

## 7) TTS (Kokoro) Notes

- Prefer streaming synthesis API if available; otherwise chunk responses (sentence/phrase) to reduce latency.
- Voices: store user preference in `profile` memory; default to neutral EN.
- Add SSML support for prosody and pauses.

---

## 8) Memory Implementation Details

**Embedding**: use `all-MiniLM-L6-v2` locally (384 dims) for speed; normalize vectors.

**Upsert**:

```sql
INSERT INTO memory_facts(user_id, kind, content)
VALUES ($1, $2, $3)
ON CONFLICT (...) DO UPDATE ...
RETURNING id;
```

Then compute embedding on `content->>'text'` and store into `memory_vectors`.

**Recall SQL (pgvector)**:

```sql
SELECT f.id, f.kind, f.content
FROM memory_vectors v
JOIN memory_facts f ON f.id = v.fact_id
WHERE v.user_id = $1
ORDER BY v.embedding <-> $2
LIMIT 5;
```

**Decay/Retention**:

- `ephemeral` rows get `expires_at` and daily decay score; purge job nightly.
- Promote ephemeral→fact if referenced 3+ times within 7 days.

---

## 9) Security & Privacy

- Mic hot by design; **store audio only on wake events** (ring buffer flushed otherwise).
- Encrypt PII at rest; separate `profile` table with column-level encryption.
- API keys (Gemini) via env/secret manager; never log prompts with tokens.
- Redact secrets from transcripts (regex for card numbers, etc.).

---

## 10) Observability

- Structured logs per turn: `turn_id`, latencies (ASR, routing, LLM, tools, TTS), wake-word conf.
- OpenTelemetry traces across orchestrator → LLM → tools → memory.
- Metrics: WER (Whisper), hotword FPR/TPR, turn success, TTFB, CTR for memory recall usefulness.

---

## 11) Local Dev & Docker

**docker-compose.yml (sketch)** \`
