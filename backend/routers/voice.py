"""
backend/routers/voice.py

Dynamic WebSocket voice endpoint.

URL: /ws/voice?campaign_id=<uuid>&token=<jwt>

Flow:
  1. Validate JWT from query param.
  2. Load campaign + questions from DB.
  3. Generate dynamic system prompt via prompt_builder.
  4. Initialise VoiceAgent with that prompt.
  5. Stream audio bidirectionally until call ends.
  6. On call end: extract answers -> save to DB -> optionally append to Sheet.

Fixes from original:
  - campaign_id is str not UUID (matches String(36) model).
  - Graceful handling when Vertex AI / Sheets are not configured locally.
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# agent.py lives one level above backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent import VoiceAgent

from database.db import get_db
from database.models import Campaign, Question, Response
from services.extraction_service import extract_answers
from services.prompt_builder import build_system_prompt
from services.sheet_service import append_response_row
from utils.security import decode_access_token

logger = logging.getLogger(__name__)
router  = APIRouter(tags=["Voice"])

# ---------------------------------------------------------------------------
# Environment config — falls back to config.py (which tries GCP Secret Manager)
# ---------------------------------------------------------------------------
def _get_config_value(env_var: str, config_attr: str, default: str = "") -> str:
    """Read from env var first, then from config.py as fallback."""
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    try:
        import config as _cfg
        return str(getattr(_cfg, config_attr, default)).strip()
    except Exception:
        return default

VERTEX_MODEL_RESOURCE: str   = _get_config_value("VERTEX_MODEL_RESOURCE", "VERTEX_MODEL_RESOURCE")
VERTEX_WS_URL: str           = _get_config_value("VERTEX_WS_URL", "VERTEX_WS_URL")
VERTEX_PROJECT_ID: str       = _get_config_value("VERTEX_PROJECT_ID", "VERTEX_PROJECT_ID")
VERTEX_LOCATION: str         = os.environ.get("VERTEX_LOCATION", "us-central1")
GOOGLE_SHEET_ID: str         = _get_config_value("GOOGLE_SHEET_ID", "GOOGLE_SHEET_ID")
# ENABLE_SHEETS: auto-enable when GOOGLE_SHEET_ID is configured, or honour explicit env var
_enable_sheets_env = os.environ.get("ENABLE_SHEETS", "").strip().lower()
ENABLE_SHEETS: bool = (
    _enable_sheets_env == "true"
    if _enable_sheets_env in ("true", "false")
    else bool(GOOGLE_SHEET_ID)   # auto-enable when sheet ID is available
)
INACTIVITY_TIMEOUT_SECONDS: int     = int(os.environ.get("INACTIVITY_TIMEOUT_SECONDS", "120"))
USER_SILENCE_DETECTION_SECONDS: int = int(os.environ.get("USER_SILENCE_DETECTION_SECONDS", "8"))

# In-memory session state (per-WebSocket connection)
_active_agents:    Dict[str, VoiceAgent] = {}
_active_call_ids:  Dict[str, str]        = {}
_active_campaigns: Dict[str, Campaign]   = {}
_active_questions: Dict[str, List]       = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _authenticate_ws(token: Optional[str]) -> Optional[dict]:
    """Decode JWT from WebSocket query param. Returns payload dict or None."""
    if not token:
        return None
    try:
        return decode_access_token(token)
    except Exception:
        return None


def _build_transcript(agent: VoiceAgent) -> str:
    """Assemble a human-readable transcript from agent conversation history."""
    lines = []
    for h in getattr(agent, "conversation_history", []):
        label = "Sneha" if h["role"] == "assistant" else "Respondent"
        lines.append(f"{label}: {h['text']}")
    return "\n".join(lines).strip()


async def _persist_response(
    session_id: str,
    call_id: str,
    agent: VoiceAgent,
    campaign: Campaign,
    questions: list,
    db: AsyncSession,
) -> dict:
    """
    Extract structured answers from transcript, optionally write to Sheet,
    and persist the Response record to the database.
    Returns the answers dict.
    """
    transcript          = _build_transcript(agent)
    detected_gender_raw = getattr(agent, "detected_gender", "unknown")
    gender_label        = {"male": "M", "female": "F"}.get(detected_gender_raw, "Unknown")

    q_dicts = [
        {
            "id": str(q.id),
            "question_order": q.question_order,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "options": q.options or [],
        }
        for q in questions
    ]

    # Extract answers via LLM (falls back to all "No Response" if Vertex not configured)
    answers: dict = {}
    if VERTEX_PROJECT_ID and transcript:
        try:
            answers = extract_answers(
                transcript=transcript,
                questions=q_dicts,
                vertex_project=VERTEX_PROJECT_ID,
                vertex_location=VERTEX_LOCATION,
                detected_gender=gender_label,
            )
        except Exception as e:
            logger.warning("Answer extraction failed: %s", e)

    meta = answers.pop("_meta", {})

    # Optionally append to Google Sheet
    if ENABLE_SHEETS and GOOGLE_SHEET_ID and campaign.google_sheet_tab_name:
        try:
            append_response_row(
                spreadsheet_id=GOOGLE_SHEET_ID,
                tab_name=campaign.google_sheet_tab_name,
                call_id=call_id,
                structured_answers=answers,
                questions=q_dicts,
                transcript=transcript,
                detected_gender=meta.get("detected_gender", "Unknown"),
            )
        except Exception as e:
            logger.warning("Sheet append failed (non-fatal): %s", e)

    # Always persist to local DB
    response_record = Response(
        campaign_id=str(campaign.id),
        call_id=call_id,
        structured_answers=answers,
        transcript=transcript,
        detected_gender=meta.get("detected_gender", "Unknown"),
    )
    db.add(response_record)
    try:
        await db.commit()
        logger.info("Response persisted: call_id=%s campaign=%s", call_id, campaign.id)
    except Exception as e:
        logger.exception("DB commit for response failed: %s", e)
        await db.rollback()

    return answers


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    campaign_id: str = Query(..., description="Campaign ID (UUID string)"),  # FIX: str not UUID
    token: Optional[str] = Query(None, description="JWT access token"),
    db: AsyncSession = Depends(get_db),
):
    """
    Dynamic voice survey WebSocket.
    Connect with: ws://<host>/ws/voice?campaign_id=<id>&token=<jwt>
    """
    # ── Authentication ──────────────────────────────────────────────────────
    payload = _authenticate_ws(token)
    if payload is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    session_id = str(id(websocket))
    call_id    = str(uuid.uuid4())
    _active_call_ids[session_id] = call_id

    logger.info("WS connected: session=%s call=%s campaign=%s", session_id, call_id, campaign_id)

    # ── Load campaign ───────────────────────────────────────────────────────
    campaign = await db.get(Campaign, campaign_id)
    if not campaign or campaign.status != "active":
        await websocket.send_json({"type": "error", "message": "Campaign not found or not active"})
        await websocket.close()
        return

    # ── Load questions ──────────────────────────────────────────────────────
    q_result  = await db.execute(
        select(Question)
        .where(Question.campaign_id == campaign_id)
        .order_by(Question.question_order)
    )
    questions = q_result.scalars().all()

    if not questions:
        await websocket.send_json({"type": "error", "message": "Campaign has no questions configured"})
        await websocket.close()
        return

    _active_campaigns[session_id] = campaign
    _active_questions[session_id] = questions

    # ── Build dynamic system prompt ─────────────────────────────────────────
    system_prompt = build_system_prompt(campaign, questions)
    logger.debug("System prompt built (%d chars)", len(system_prompt))

    # ── Callback: forward messages to client + handle survey_complete ───────
    async def send_to_client(message: dict) -> None:
        try:
            if message.get("type") == "survey_complete":
                agent = _active_agents.get(session_id)
                # Guard: persist only once
                if getattr(agent, "_sheet_written", False):
                    return
                if agent:
                    agent._sheet_written = True

                ans = await _persist_response(
                    session_id=session_id,
                    call_id=call_id,
                    agent=agent,
                    campaign=_active_campaigns[session_id],
                    questions=_active_questions[session_id],
                    db=db,
                )
                try:
                    await websocket.send_json({
                        "type": "survey_result",
                        "call_id": call_id,
                        "campaign_id": campaign_id,
                        "answers": ans,
                    })
                except Exception:
                    pass

                await asyncio.sleep(1.5)
                if agent:
                    await agent.end_conversation()
                # Close WebSocket so the receive() in the main loop unblocks immediately
                try:
                    await websocket.close()
                except Exception:
                    pass
                return

            await websocket.send_json(message)
        except Exception as e:
            logger.debug("send_to_client error (likely disconnected): %s", e)

    # ── Create VoiceAgent ───────────────────────────────────────────────────
    agent = VoiceAgent(
        model_resource=VERTEX_MODEL_RESOURCE,
        ws_url=VERTEX_WS_URL,
        system_prompt=system_prompt,
        response_callback=send_to_client,
        inactivity_timeout=INACTIVITY_TIMEOUT_SECONDS,
        silence_detection=USER_SILENCE_DETECTION_SECONDS,
    )
    _active_agents[session_id] = agent

    try:
        if not VERTEX_MODEL_RESOURCE or not VERTEX_WS_URL:
            await websocket.send_json({
                "type": "error",
                "message": "Voice model not configured. Set VERTEX_MODEL_RESOURCE and VERTEX_WS_URL.",
            })
            await websocket.close()
            return

        success = await agent.start_conversation()
        if not success:
            await websocket.send_json({"type": "error", "message": "Failed to connect to voice model"})
            await websocket.close()
            return

        # ── Main message loop ───────────────────────────────────────────────
        # We poll with a short timeout so that when agent.end_conversation()
        # is called from the send_to_client callback (survey_complete path),
        # conversation_active becomes False and we can exit cleanly without
        # waiting forever on websocket.receive().
        while True:
            # Exit if the agent has been shut down (e.g. by survey_complete handler)
            if not agent.conversation_active:
                logger.info("Agent inactive — exiting message loop: session=%s", session_id)
                break

            try:
                # Short timeout so we can re-check conversation_active frequently
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

                if "bytes" in message:
                    await agent.send_audio(message["bytes"])

                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "end_conversation":
                        logger.info("Client requested end: session=%s", session_id)
                        await agent.end_conversation()
                        break
                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Normal — just re-check conversation_active on next iteration
                continue
            except WebSocketDisconnect:
                logger.info("Client disconnected: session=%s", session_id)
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info("Client already disconnected: %s", session_id)
                else:
                    logger.error("RuntimeError in message loop: %s", e, exc_info=True)
                break
            except Exception as e:
                logger.error("Message loop error: %s", e, exc_info=True)
                break

    except Exception as e:
        logger.error("WebSocket handler error: %s", e, exc_info=True)

    finally:
        # ── Fallback persistence if survey_complete was never emitted ───────
        agent = _active_agents.get(session_id)
        if agent:
            already_written = getattr(agent, "_sheet_written", False)
            if getattr(agent, "conversation_active", False):
                await agent.end_conversation()
            if not already_written:
                try:
                    await _persist_response(
                        session_id=session_id,
                        call_id=call_id,
                        agent=agent,
                        campaign=_active_campaigns.get(session_id, campaign),
                        questions=_active_questions.get(session_id, questions),
                        db=db,
                    )
                    logger.info("Fallback persistence completed: call_id=%s", call_id)
                except Exception as e:
                    logger.exception("Fallback persistence failed: %s", e)

        # Cleanup session state
        _active_agents.pop(session_id, None)
        _active_call_ids.pop(session_id, None)
        _active_campaigns.pop(session_id, None)
        _active_questions.pop(session_id, None)

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("WS closed: session=%s call=%s", session_id, call_id)