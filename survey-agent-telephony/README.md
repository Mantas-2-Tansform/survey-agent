# Survey Agent — VICIdial + Jambonz Telephony Integration

## How to Apply This Zip

### Step 1: Unzip into your survey-agent repo

```bash
cd /path/to/survey-agent-main
unzip survey-agent-telephony.zip
```

This gives you a `survey-agent-telephony/` folder. Now copy the files in:

```bash
# Copy the new telephony package (4 files)
cp -r survey-agent-telephony/telephony/ ./telephony/

# Replace the two modified root files
cp survey-agent-telephony/application.py ./application.py
cp survey-agent-telephony/requirements.txt ./requirements.txt
```

### Step 2: Install new dependency

```bash
pip install httpx numpy
```

Or if deploying to Cloud Run:
```bash
# requirements.txt already has httpx and numpy — Cloud Build will pick them up
```

### Step 3: Set environment variables

Add these to your Cloud Run service (or `.env` file for local dev):

```bash
# ── VICIdial API ──
VICIDIAL_API_URL=https://your-vicidial-server.com     # Base URL of VICIdial
VICIDIAL_API_USER=api_admin                            # Non-Agent API username
VICIDIAL_API_PASS=your_api_password                    # Non-Agent API password
VICIDIAL_AGENT_USER=survey_ai_agent                    # Agent username for our AI
VICIDIAL_AGENT_PASS=agent_password                     # Agent password
VICIDIAL_SOURCE=survey_ai                              # API source identifier

# ── Jambonz (SIP↔WebSocket bridge) ──
# These are only needed if you keep Jambonz as the SIP bridge.
# Jambonz is already configured to hit your /jambonz/call-hook etc.
# No extra env vars needed for the receiving side.
```

### Step 4: Configure Jambonz application

In the Jambonz portal, create (or update) an application with these webhooks:

| Setting        | Value                                          |
|----------------|------------------------------------------------|
| Call Hook      | `https://your-server.example.com/jambonz/call-hook`   |
| Status Hook    | `https://your-server.example.com/jambonz/status-hook`  |

The listen action hook is set dynamically per call.

### Step 5: Configure VICIdial SIP Transfer

Once the VICIdial team responds, configure their SIP transfer to point to your
Jambonz SIP trunk. They need:

- **SIP Destination**: Your Jambonz SIP IP and port
- **Audio Codec**: PCMU (G.711 μ-law) or PCMA — Jambonz handles conversion
- **Custom SIP Headers** (ask them to send these):
  - `X-Lead-ID` → VICIdial lead ID
  - `X-Uniqueid` → VICIdial call uniqueid
  - `X-Campaign-ID` → Campaign identifier
  - `X-List-ID` → List identifier

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VICIdial (dialer)                                                       │
│  ┌─────────────┐    SIP INVITE        ┌──────────────┐                  │
│  │ Predictive   │ ──(with X-Lead-ID)──→│   Jambonz    │                  │
│  │ Dialer       │    + audio           │ SIP↔WS Bridge│                  │
│  └─────────────┘                       └──────┬───────┘                  │
│                                               │                          │
│            ┌──────────────────────────────────┘                          │
│            │  1. POST /jambonz/call-hook (metadata + lead_id)            │
│            │  2. Returns "listen" verb                                    │
│            │  3. WS /jambonz/listen/{session_id} opens                   │
│            ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  FastAPI Server (your survey-agent)                              │     │
│  │                                                                  │     │
│  │  telephony_router.py                                             │     │
│  │       │                                                          │     │
│  │       ▼                                                          │     │
│  │  SurveyAudioBridge                                               │     │
│  │       │  Phone audio ──→ VoiceAgent.send_audio() ──→ Gemini     │     │
│  │       │  Gemini response ──→ resample 24k→16k ──→ Jambonz WS   │     │
│  │       │                                                          │     │
│  │       │  survey_complete:                                        │     │
│  │       │    1. LLM extract answers from transcript                │     │
│  │       │    2. Append row to Google Sheet                         │     │
│  │       │    3. POST disposition to VICIdial API                   │     │
│  │       │    4. Jambonz disconnect                                 │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  VICIdial API (receives disposition updates)                     │     │
│  │    POST /vicidial/non_agent_api.php?function=update_lead         │     │
│  │      lead_id=12345 & status=SVYCMP & comments=...                │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

## File Inventory

| File | Lines | What it does |
|------|-------|--------------|
| `telephony/__init__.py` | 5 | Package marker |
| `telephony/vicidial_client.py` | 310 | VICIdial API client: disposition updates, lead field writes, call hangup, lead upload, campaign control |
| `telephony/survey_bridge.py` | 310 | Audio bridge: Jambonz PCM ↔ VoiceAgent, post-survey Sheet write + VICIdial disposition push |
| `telephony/telephony_router.py` | 300 | Jambonz webhooks (call-hook, listen WS, status-hook) + VICIdial admin endpoints (hangup, upload-leads, campaign-control) |
| `application.py` | 330 | 2 lines added: import + `app.include_router(telephony_router)` |
| `requirements.txt` | 16 | Added `httpx`, `numpy` |

## API Endpoints Added

### Jambonz Webhooks (called by Jambonz, not by you)
- `POST /jambonz/call-hook` — New call → returns listen verb
- `POST /jambonz/listen-action` — Listen ended → cleanup
- `POST /jambonz/status-hook` — Call status logging
- `WS /jambonz/listen/{session_id}` — Bidirectional audio

### Your Control Endpoints
- `POST /telephony/hangup` — Hang up an active call `{"lead_id": "..."}`
- `GET /telephony/agent-status` — Check AI agent status in VICIdial
- `GET /telephony/active-calls` — List active survey calls
- `POST /telephony/upload-leads` — Upload leads to VICIdial (Phase 2)
- `POST /telephony/campaign-control` — Start/pause/stop campaign (Phase 2)

## Things to Update After VICIdial Team Responds

1. **VICIdial API paths** — `AGENT_API_PATH` and `NON_AGENT_API_PATH` in `vicidial_client.py`
2. **Disposition codes** — `SurveyDisposition` enum in `vicidial_client.py`
3. **SIP header names** — `_extract_vicidial_meta()` in `telephony_router.py`
4. **Lead field mapping** — `_push_vicidial_disposition()` in `survey_bridge.py`
5. **Agent username** — `VICIDIAL_AGENT_USER` env var
6. **Campaign control** — `campaign_control()` method once they confirm the API
