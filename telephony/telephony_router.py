# telephony/telephony_router.py
"""
Telephony Router (VICIdial + Jambonz)
=======================================
Call flow:
  1. VICIdial dials lead via its predictive/progressive dialer
  2. Lead answers → VICIdial SIP-transfers to Jambonz
  3. Jambonz receives SIP → POSTs our /jambonz/call-hook
  4. We return a "listen" verb → Jambonz opens bidirectional audio WS
  5. Audio bridge feeds phone audio to VoiceAgent ↔ Gemini Live
  6. Survey completes → we push disposition back to VICIdial API

Endpoints:
  POST  /jambonz/call-hook       — Jambonz webhook: new call arrives
  POST  /jambonz/listen-action   — Jambonz webhook: listen verb completed
  POST  /jambonz/status-hook     — Jambonz webhook: call status changes
  WS    /jambonz/listen/{sid}    — Bidirectional audio stream

  POST  /telephony/hangup        — Hang up an active call via VICIdial API
  POST  /telephony/upload-leads  — Upload leads to VICIdial (Phase 2)
  GET   /telephony/agent-status  — Check our AI agent's status in VICIdial

VICIdial SIP Headers:
  When VICIdial transfers a call, it can pass metadata via custom SIP headers.
  These arrive in Jambonz's call-hook body under "sip_headers" or "customerData".
  Typical headers:
    X-Lead-ID      → lead_id
    X-Uniqueid     → uniqueid (VICIdial's internal call ID)
    X-Campaign-ID  → campaign_id
    X-List-ID      → list_id
    X-Phone-Number → phone number that was dialed
"""

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import logging
import json
import uuid

from telephony.survey_bridge import SurveyAudioBridge

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

# Active bridges: session_id → SurveyAudioBridge
_active_bridges: Dict[str, SurveyAudioBridge] = {}


# =============================================================================
# HELPERS: Extract VICIdial metadata from Jambonz call-hook body
# =============================================================================

def _extract_vicidial_meta(body: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract VICIdial call context from Jambonz webhook body.

    Jambonz passes SIP headers in several possible locations depending on
    configuration. We check all of them.

    Returns dict with: lead_id, uniqueid, campaign_id, list_id,
                       phone_number, first_name, last_name, direction
    """
    meta = {}

    # Source 1: customerData / tag (set during outbound call creation)
    customer_data = body.get("customerData", {}) or body.get("tag", {})

    # Source 2: SIP headers (custom X- headers from VICIdial)
    sip_headers = body.get("sip_headers", {}) or body.get("sipHeaders", {})

    # Source 3: Flat fields on the body itself
    # (some Jambonz versions flatten headers)

    # Extract with fallback chain for each field
    meta["lead_id"] = (
        customer_data.get("lead_id")
        or sip_headers.get("X-Lead-ID")
        or sip_headers.get("x-lead-id")
        or body.get("lead_id")
    )
    meta["uniqueid"] = (
        customer_data.get("uniqueid")
        or sip_headers.get("X-Uniqueid")
        or sip_headers.get("x-uniqueid")
        or body.get("uniqueid")
    )
    meta["campaign_id"] = (
        customer_data.get("campaign_id")
        or sip_headers.get("X-Campaign-ID")
        or sip_headers.get("x-campaign-id")
        or body.get("campaign_id")
    )
    meta["list_id"] = (
        customer_data.get("list_id")
        or sip_headers.get("X-List-ID")
        or sip_headers.get("x-list-id")
    )
    meta["phone_number"] = (
        body.get("from")
        or customer_data.get("phone_number")
        or sip_headers.get("X-Phone-Number")
    )
    meta["first_name"] = (
        customer_data.get("first_name")
        or sip_headers.get("X-First-Name")
        or ""
    )
    meta["last_name"] = (
        customer_data.get("last_name")
        or sip_headers.get("X-Last-Name")
        or ""
    )
    meta["direction"] = body.get("direction", "inbound")

    # Clean None values to empty strings
    return {k: (v or "") for k, v in meta.items()}


# =============================================================================
# JAMBONZ WEBHOOKS
# =============================================================================

@router.post("/jambonz/call-hook")
async def call_hook(request: Request):
    """
    Called by Jambonz when a VICIdial SIP transfer arrives (or inbound call).
    Returns Jambonz "listen" verb to start bidirectional audio.
    """
    body = await request.json()
    logger.info(f"📞 Call-hook received: {json.dumps(body, indent=2)}")

    call_sid = body.get("callSid", "unknown")
    vicidial_meta = _extract_vicidial_meta(body)
    session_id = str(uuid.uuid4())[:12]

    logger.info(
        f"📞 Call {call_sid}: lead_id={vicidial_meta.get('lead_id')} "
        f"campaign={vicidial_meta.get('campaign_id')} "
        f"phone={vicidial_meta.get('phone_number')} "
        f"session={session_id}"
    )

    # Build WebSocket URL for bidirectional audio
    server_host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss" if request.url.scheme == "https" else "ws"
    listen_url = f"{scheme}://{server_host}/jambonz/listen/{session_id}"

    # Stash VICIdial metadata so the listen WS endpoint can retrieve it
    _pending_metadata[session_id] = {
        "call_sid": call_sid,
        "vicidial_meta": vicidial_meta,
    }

    return [
        {
            "verb": "listen",
            "url": listen_url,
            "actionHook": f"https://{server_host}/jambonz/listen-action",
            "metadata": {
                "session_id": session_id,
                "call_sid": call_sid,
                **vicidial_meta,
            },
            "sampleRate": 16000,
            "bidirectionalAudio": {
                "enabled": True,
                "streaming": True,
                "sampleRate": 16000,
            },
            "mixType": "mono",
            "maxLength": 1800,  # 30 min max
            "passDtmf": True,
        }
    ]


# Temporary storage for metadata between call-hook and listen WS connect
_pending_metadata: Dict[str, Dict] = {}


@router.post("/jambonz/listen-action")
async def listen_action_hook(request: Request):
    """Listen verb completed — clean up resources."""
    body = await request.json()
    logger.info(f"📴 Listen action completed: {json.dumps(body, indent=2)}")

    metadata = body.get("customerData", {})
    session_id = metadata.get("session_id")
    if session_id and session_id in _active_bridges:
        bridge = _active_bridges.pop(session_id)
        await bridge.close()
        logger.info(f"🧹 Cleaned up bridge: session={session_id}")

    # Clean pending metadata too
    if session_id and session_id in _pending_metadata:
        del _pending_metadata[session_id]

    return []


@router.post("/jambonz/status-hook")
async def status_hook(request: Request):
    """Call status changes: trying → ringing → in-progress → completed."""
    body = await request.json()
    call_sid = body.get("callSid", "unknown")
    call_status = body.get("callStatus", "unknown")
    logger.info(f"📊 Call status: {call_sid} → {call_status}")

    if call_status == "completed":
        duration = body.get("duration", 0)
        logger.info(f"✅ Call {call_sid} completed, duration: {duration}s")
    elif call_status in ("failed", "no-answer", "busy"):
        sip_status = body.get("sipStatus", "")
        logger.warning(f"❌ Call {call_sid}: {call_status} (SIP: {sip_status})")

    return {"status": "ok"}


# =============================================================================
# WEBSOCKET: BIDIRECTIONAL AUDIO STREAM
# =============================================================================

@router.websocket("/jambonz/listen/{session_id}")
async def listen_websocket(websocket: WebSocket, session_id: str):
    """
    Jambonz connects here for bidirectional audio.

    Protocol:
    1. First message: JSON metadata (callSid, sampleRate, mixType, etc.)
    2. Binary frames: L16 PCM audio from the phone (caller's voice)
    3. We send back: binary L16 PCM (agent's voice) + JSON control frames
    """
    await websocket.accept(subprotocol="ws.jambonz.org")
    logger.info(f"🔌 Jambonz audio WS connected: session={session_id}")

    bridge = None
    try:
        # First message is JSON metadata from Jambonz
        first_message = await websocket.receive_text()
        jambonz_meta = json.loads(first_message)
        logger.info(
            f"📋 Jambonz WS metadata: call_sid={jambonz_meta.get('callSid')} "
            f"sampleRate={jambonz_meta.get('sampleRate')}"
        )

        call_sid = jambonz_meta.get("callSid", "unknown")
        input_sample_rate = jambonz_meta.get("sampleRate", 16000)

        # Retrieve VICIdial metadata stashed by call-hook
        pending = _pending_metadata.pop(session_id, {})
        vicidial_meta = pending.get("vicidial_meta", {})

        # Create audio bridge with full VICIdial context
        bridge = SurveyAudioBridge(
            jambonz_ws=websocket,
            call_sid=call_sid,
            session_id=session_id,
            input_sample_rate=input_sample_rate,
            vicidial_meta=vicidial_meta,
        )
        _active_bridges[session_id] = bridge

        # Initialize VoiceAgent → Gemini Live
        await bridge.initialize()

        # Main audio loop
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                await bridge.handle_phone_audio(message["bytes"])

            elif "text" in message:
                text_data = json.loads(message["text"])
                msg_type = text_data.get("type", "")
                if msg_type == "dtmf":
                    logger.info(f"📱 DTMF: {text_data.get('digit')}")
                elif msg_type == "mark":
                    logger.debug(f"🔖 Mark: {text_data.get('name')}")

    except WebSocketDisconnect:
        logger.info(f"🔌 Jambonz WS disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"❌ Error in audio stream: {e}", exc_info=True)
    finally:
        if bridge:
            await bridge.close()
        _active_bridges.pop(session_id, None)
        _pending_metadata.pop(session_id, None)
        logger.info(f"🧹 Audio bridge closed: session={session_id}")


# =============================================================================
# CALL CONTROL: Hang up via VICIdial API
# =============================================================================

@router.post("/telephony/hangup")
async def hangup_call(request: Request):
    """
    Programmatically hang up an active call.

    Request body:
    {
        "lead_id": "12345",       // VICIdial lead ID
        "uniqueid": "1234567890"  // VICIdial call uniqueid (optional)
    }
    """
    from telephony.vicidial_client import vicidial

    body = await request.json()
    lead_id = body.get("lead_id")
    uniqueid = body.get("uniqueid")

    if not lead_id and not uniqueid:
        return {"error": "lead_id or uniqueid required"}, 400

    result = await vicidial.hangup_call(lead_id=lead_id, uniqueid=uniqueid)
    return result


# =============================================================================
# AI AGENT STATUS IN VICIDIAL
# =============================================================================

@router.get("/telephony/agent-status")
async def get_agent_status():
    """Check our AI agent's status in VICIdial."""
    from telephony.vicidial_client import vicidial
    result = await vicidial.get_agent_status()
    return result


@router.get("/telephony/active-calls")
async def get_active_calls():
    """List all currently active survey calls."""
    calls = []
    for sid, bridge in _active_bridges.items():
        calls.append({
            "session_id": sid,
            "call_sid": bridge.call_sid,
            "call_id": bridge.call_id,
            "lead_id": bridge.lead_id,
            "campaign_id": bridge.campaign_id,
            "phone_number": bridge.vicidial_meta.get("phone_number", ""),
        })
    return {"active_calls": calls, "count": len(calls)}


# =============================================================================
# PHASE 2: CAMPAIGN MANAGEMENT
# =============================================================================

@router.post("/telephony/upload-leads")
async def upload_leads(request: Request):
    """
    Upload a list of leads to VICIdial for a survey campaign.

    Request body:
    {
        "campaign_id": "SURVEY01",
        "list_id": "101",
        "leads": [
            {"phone_number": "9876543210", "first_name": "Raj", "last_name": "Kumar"},
            {"phone_number": "9876543211"}
        ]
    }
    """
    from telephony.vicidial_client import vicidial

    body = await request.json()
    campaign_id = body.get("campaign_id")
    list_id = body.get("list_id")
    leads = body.get("leads", [])

    if not leads:
        return {"error": "leads list is required"}, 400

    results = await vicidial.add_leads_batch(
        leads=leads,
        campaign_id=campaign_id,
        list_id=list_id,
    )

    succeeded = sum(1 for r in results if r.get("success"))
    return {
        "total": len(leads),
        "succeeded": succeeded,
        "failed": len(leads) - succeeded,
        "results": results,
    }


@router.post("/telephony/campaign-control")
async def campaign_control(request: Request):
    """
    Start/pause/stop a VICIdial campaign. (Phase 2)

    Request body:
    {
        "campaign_id": "SURVEY01",
        "action": "start"   // "start" | "pause" | "stop"
    }
    """
    from telephony.vicidial_client import vicidial

    body = await request.json()
    campaign_id = body.get("campaign_id")
    action = body.get("action")

    if not campaign_id or not action:
        return {"error": "campaign_id and action required"}, 400

    result = await vicidial.campaign_control(campaign_id, action)
    return result
