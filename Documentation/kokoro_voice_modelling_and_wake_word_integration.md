# Kokoro Voice Modelling and Wake Word Integration

This document supplements the main Jarvis Voice Agent handoff and focuses on **voice model control**, **wake-word configuration**, and **Porcupine integration**.

---

## 1) Wake Word Integration (Porcupine)

### Default Setup
- **Wake Word:** `creo` (previously `jarvis`)
- **Engine:** [Picovoice Porcupine](https://github.com/Picovoice/porcupine) — used for robust, offline wake-word detection.
- **Model:** `porcupine_params.pv`
- **Keyword file:** `creo.ppn`
- **Sensitivity:** Default 0.65 (tunable per environment)

### Architecture Placement
- Porcupine runs in the **audio capture pipeline**, parallel to the VAD and ring buffer.
- When Porcupine detects `creo`, it emits a `WakeWordHit` event identical to openWakeWord.

```python
import pvporcupine
import pyaudio

porcupine = pvporcupine.create(keyword_paths=['creo.ppn'], sensitivities=[0.65])
pa = pyaudio.PyAudio()
stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=512)

while True:
    pcm = stream.read(512)
    pcm = np.frombuffer(pcm, dtype=np.int16)
    result = porcupine.process(pcm)
    if result >= 0:
        print("Wake word detected: Creo!")
        emit_wakeword_event('creo')
```

### Switching Between Wake Word Engines
- **Default**: Porcupine
- **Fallback**: openWakeWord (open source)
- Agent can switch dynamically when instructed, e.g.:
  - “Use openWakeWord instead.” → toggles config
  - “Switch wake word back to Jarvis.” → reloads model `jarvis.ppn`

---

## 2) Kokoro Voice Modelling Breakdown

Voice selection and tuning parameters can be switched dynamically through conversational commands such as:
> “Change to Hindi female voice.”  
> “Use English male number two.”  
> “Make it weird.”

Each voice preset combines **model identifiers** and **parameter tuning** for Kokoro’s voice generation API.

---

### Normal Voices

| Type | Kokoro ID | Parameters |
|------|------------|-------------|
| **Normal Hindi Female** | hf_alpha | `hf_alpha=-1.2`, `af_sky=0.7` |
| **Normal English Female** | hf_alpha | `Hf_alpha=0.4`, `Af_sky=1` |
| **Normal Hindi Male** | hm_omega | `hm_omega=1.3`, `am_puck=1` |
| **Normal English Male** | hm_puck | `Hm_puck=1`, `Hm_omega=0.8` |

---

### Alternate Voices

| Variant | Description | Substitution |
|----------|--------------|---------------|
| **Male Voice #2** | Alternate male tone | Replace `am_puck` → `am_micheal` |
| **Female Voice #2** | Alternate female tone | Replace `af_sky` → `af_aoede` |

> Both Hindi (Hinglish) and English versions follow the same substitution pattern.

---

### Hinglish (Future Variant)

- Hinglish voices will be hybrid models mixing English and Hindi phonetic structures.
- Placeholder: identical to Hindi variants for now.

---

### Special / Fun Voices

| Mode | ID | Notes |
|------|----|-------|
| **Fun Male (Weird)** | `Pm_alex` | Playful French-like accent, used when user says “mix it up.” |
| **Fun Female (Weird)** | `Pf_Dora` | Cheerful female tone, used with “make it weird.” |
| **Blackie Mode** | `jf_alpha` | Triggered when camera detects Blackie (pet). Voice playfully exclaims “Blackieee!” |

---

## 3) Dynamic Voice Switching Logic

Voice models are stored as presets in the agent’s configuration. When the user requests a switch, the orchestrator updates the current TTS context.

```python
VOICE_PROFILES = {
    'hindi_female': {'hf_alpha': -1.2, 'af_sky': 0.7},
    'english_female': {'Hf_alpha': 0.4, 'Af_sky': 1},
    'hindi_male': {'hm_omega': 1.3, 'am_puck': 1},
    'english_male': {'Hm_puck': 1, 'Hm_omega': 0.8},
    'male2': {'am_micheal': 1.0},
    'female2': {'af_aoede': 1.0},
    'fun_male': {'Pm_alex': 1.0},
    'fun_female': {'Pf_Dora': 1.0},
    'blackie_mode': {'jf_alpha': 1.0}
}

current_voice = 'english_male'

async def set_voice_mode(mode):
    global current_voice
    if mode in VOICE_PROFILES:
        current_voice = mode
        print(f"Switched to voice profile: {mode}")
        await kokoro.set_voice(VOICE_PROFILES[mode])
    else:
        print(f"Unknown voice mode: {mode}")
```

### Example Voice Switch Commands
- “Switch to female two.” → `set_voice_mode('female2')`
- “Change to Hindi male.” → `set_voice_mode('hindi_male')`
- “Make it weird.” → `set_voice_mode('fun_male')`
- “Use Blackie mode.” → `set_voice_mode('blackie_mode')`

---

## 4) Integration Notes

- Kokoro parameter changes are hot-swappable; updates should apply mid-session without restarting the agent.
- Persist last-used voice in `profile` memory (Postgres `memory_facts` kind=`profile` → `preferred_voice`).
- Reapply preferred voice on next startup or session resume.
- Optionally expose `/voice/current` and `/voice/set` endpoints for external control.

---

**Default Voice:** `english_male`

**Default Wake Word:** `creo` (Porcupine)

**Fallback Wake Word Engine:** openWakeWord

---

**Next Steps:**
1. Integrate Porcupine wake-word pipeline into orchestrator.
2. Implement `set_voice_mode` API route and connect to Kokoro TTS module.
3. Add persistence for last-used voice and wake-word engine.
4. Create small test harness to switch between voices using text commands.

