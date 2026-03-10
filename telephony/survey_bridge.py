# telephony/survey_bridge.py
"""
Jambonz ↔ VoiceAgent Audio Bridge (Survey + VICIdial)
======================================================
Makes a VICIdial-transferred phone call look exactly like a browser
WebSocket session to VoiceAgent.

Call flow:
  VICIdial dials lead → lead answers → VICIdial SIP-transfers to Jambonz
  → Jambonz streams audio to this bridge via WebSocket
  → We feed audio into VoiceAgent.send_audio()
  → VoiceAgent talks to Gemini Live, gets responses
  → We intercept responses, send audio back to Jambonz → phone speaker

When survey_complete fires:
  1. Extract answers from transcript via LLM
  2. Append row to Google Sheet
  3. Push disposition + answers to VICIdial API
  4. Tell Jambonz to disconnect
"""

import asyncio
import base64
import json
import logging
import struct
import time
import uuid
from typing import Dict, Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class SurveyAudioBridge:
    """
    Bridges audio between a Jambonz WebSocket and a survey VoiceAgent instance.
    Carries VICIdial call metadata (lead_id, uniqueid, campaign_id) through the
    entire lifecycle so we can push disposition back at the end.

    Jambonz Protocol:
    - Receives: binary frames of L16 PCM audio (16-bit signed, little-endian)
    - Sends:    binary frames of L16 PCM audio back for playback
    - Text frames: {"type": "killAudio"} or {"type": "disconnect"}

    VoiceAgent callback protocol:
    - {"type": "audio", "data": "<base64 pcm>", "sample_rate": 24000}
    - {"type": "transcript_chunk", "role": "assistant"|"user", "text": "..."}
    - {"type": "transcript", "role": "assistant", "text": "..."}
    - {"type": "survey_complete"}
    - {"type": "turn_complete"}
    - {"type": "interrupted"}
    """

    def __init__(
        self,
        jambonz_ws: WebSocket,
        call_sid: str,
        session_id: str,
        input_sample_rate: int = 16000,
        vicidial_meta: Dict[str, str] = None,
    ):
        """
        Args:
            jambonz_ws:       The Jambonz WebSocket connection
            call_sid:         Jambonz call SID
            session_id:       Our internal session identifier
            input_sample_rate: Audio rate from Jambonz (usually 16kHz)
            vicidial_meta:    Metadata from VICIdial SIP headers:
                              {"lead_id", "uniqueid", "campaign_id", "list_id",
                               "phone_number", "first_name", "last_name"}
        """
        self.jambonz_ws = jambonz_ws
        self.call_sid = call_sid
        self.session_id = session_id
        self.call_id = str(uuid.uuid4())
        self.input_sample_rate = input_sample_rate

        # VICIdial call context — used for disposition push at end
        self.vicidial_meta = vicidial_meta or {}
        self.lead_id = self.vicidial_meta.get("lead_id")
        self.uniqueid = self.vicidial_meta.get("uniqueid")
        self.campaign_id = self.vicidial_meta.get("campaign_id")

        self.agent = None
        self._closed = False
        self._send_lock = asyncio.Lock()
        self._sheet_written = False
        self._disposition_pushed = False

        # Audio buffer: accumulate small Jambonz frames before sending to Gemini
        self._audio_buffer = bytearray()
        self._buffer_flush_task = None
        self._BUFFER_FLUSH_INTERVAL = 0.1  # 100ms

    async def initialize(self):
        """
        Create and initialize a VoiceAgent for this phone call.
        Mirrors what application.py's websocket_endpoint() does.
        """
        from config import (
            VERTEX_MODEL_RESOURCE,
            VERTEX_WS_URL,
            SYSTEM_PROMPT,
            INACTIVITY_TIMEOUT_SECONDS,
            USER_SILENCE_DETECTION_SECONDS,
        )
        from agent import VoiceAgent

        logger.info(
            f"🚀 Initializing survey agent: call_sid={self.call_sid} "
            f"lead_id={self.lead_id} campaign={self.campaign_id}"
        )

        self.agent = VoiceAgent(
            model_resource=VERTEX_MODEL_RESOURCE,
            ws_url=VERTEX_WS_URL,
            system_prompt=SYSTEM_PROMPT,
            response_callback=self._agent_response_callback,
            inactivity_timeout=INACTIVITY_TIMEOUT_SECONDS,
            silence_detection=USER_SILENCE_DETECTION_SECONDS,
        )

        success = await self.agent.start_conversation()
        if not success:
            raise RuntimeError("Failed to start VoiceAgent conversation with Gemini")

        self._buffer_flush_task = asyncio.create_task(self._audio_buffer_flusher())
        logger.info(f"✅ Survey agent ready: session={self.session_id}")

    # =========================================================================
    # CALLBACK: Intercepts VoiceAgent responses → forwards to Jambonz
    # =========================================================================

    async def _agent_response_callback(self, data: Dict[str, Any]):
        """
        Called by VoiceAgent whenever it has a response.
        In phone mode: extract audio, resample, send to Jambonz.
        """
        if self._closed:
            return

        msg_type = data.get("type", "")

        if msg_type == "audio":
            audio_b64 = data.get("data", "")
            sample_rate = data.get("sample_rate", 24000)
            if audio_b64:
                try:
                    pcm_raw = base64.b64decode(audio_b64)
                    pcm_16k = self._resample_audio(pcm_raw, sample_rate, 16000)
                    async with self._send_lock:
                        if not self._closed:
                            await self.jambonz_ws.send_bytes(pcm_16k)
                except Exception as e:
                    logger.error(f"Error sending audio to Jambonz: {e}")

        elif msg_type == "transcript":
            role = data.get("role", "")
            text = data.get("text", "")
            logger.info(f"📝 [{role.upper()}]: {text[:120]}...")

        elif msg_type == "survey_complete":
            logger.info(f"📋 Survey complete: call_sid={self.call_sid} lead={self.lead_id}")
            await self._handle_survey_complete()

        elif msg_type == "interrupted":
            # User barged in — stop Jambonz playback
            try:
                async with self._send_lock:
                    if not self._closed:
                        await self.jambonz_ws.send_text(json.dumps({"type": "killAudio"}))
            except Exception:
                pass

    # =========================================================================
    # SURVEY COMPLETE → Sheet + VICIdial disposition
    # =========================================================================

    async def _handle_survey_complete(self):
        """Extract answers, write to Sheet, push disposition to VICIdial."""
        if self._sheet_written:
            return
        self._sheet_written = True

        transcript = self._build_transcript()
        detected_gender_raw = getattr(self.agent, "detected_gender", "unknown")
        gender_label = {"male": "M", "female": "F"}.get(detected_gender_raw, "Unknown")
        answers = {}

        # 1. Extract answers + write to Google Sheet
        try:
            from survey_storage import extract_answers_from_transcript, append_survey_to_sheet

            answers = extract_answers_from_transcript(
                transcript, detected_gender=gender_label
            )
            append_survey_to_sheet(self.call_id, answers, transcript)
            logger.info(f"✅ Sheet write OK: call_id={self.call_id}")
        except Exception as e:
            logger.exception("Survey storage failed: %s", e)

        # 2. Push disposition to VICIdial
        await self._push_vicidial_disposition(answers, transcript)

        # 3. Allow agent to finish speaking, then disconnect
        await asyncio.sleep(2.0)
        try:
            async with self._send_lock:
                if not self._closed:
                    await self.jambonz_ws.send_text(json.dumps({"type": "disconnect"}))
        except Exception:
            pass

    async def _push_vicidial_disposition(
        self, answers: Dict[str, Any], transcript: str
    ):
        """
        Push survey results back to VICIdial.
        Two API calls:
          1. update_disposition — set the lead status code
          2. update_lead_fields — write survey answers to lead custom fields
        """
        if self._disposition_pushed or not self.lead_id:
            if not self.lead_id:
                logger.warning("No lead_id — skipping VICIdial disposition push")
            return
        self._disposition_pushed = True

        try:
            from telephony.vicidial_client import vicidial, map_survey_to_disposition

            disposition = map_survey_to_disposition(answers, transcript)

            # Build a short comment with key survey outcomes
            comment_parts = []
            for key in ("q1_satisfaction", "q3_vote", "q4_greatest_sportsman"):
                val = answers.get(key)
                if val and val != "No Response":
                    comment_parts.append(f"{key}: {val}")
            comment = "; ".join(comment_parts) if comment_parts else "Survey call"

            # 1. Set disposition
            result = await vicidial.update_disposition(
                lead_id=self.lead_id,
                status=disposition,
                uniqueid=self.uniqueid,
                campaign_id=self.campaign_id,
                comments=comment,
            )
            logger.info(f"VICIdial disposition: {result}")

            # 2. Write answers to lead custom fields
            # Map our answer keys → VICIdial field names
            # UPDATE this mapping once VICIdial team confirms their field names
            field_map = {
                "age": answers.get("age", ""),
                "gender": answers.get("gender", ""),
                "state": answers.get("state", ""),
                "comments": comment,
                # Add more mappings as VICIdial custom fields are defined:
                # "q1_satisfaction": answers.get("q1_satisfaction", ""),
                # "q2_price_rise": answers.get("q2_price_rise", ""),
                # etc.
            }
            # Only send non-empty fields
            fields = {k: v for k, v in field_map.items() if v and v != "No Response"}
            if fields:
                result2 = await vicidial.update_lead_fields(
                    lead_id=self.lead_id,
                    fields=fields,
                )
                logger.info(f"VICIdial lead update: {result2}")

        except Exception as e:
            logger.exception("VICIdial disposition push failed: %s", e)

    def _build_transcript(self) -> str:
        """Build readable transcript from agent conversation history."""
        lines = []
        for h in getattr(self.agent, "conversation_history", []):
            role_label = "Sneha" if h["role"] == "assistant" else "Respondent"
            lines.append(f"{role_label}: {h['text']}")
        return "\n".join(lines).strip()

    # =========================================================================
    # INBOUND AUDIO: Jambonz → VoiceAgent
    # =========================================================================

    async def handle_phone_audio(self, audio_data: bytes):
        """Buffer incoming phone audio for periodic flush to agent."""
        if self._closed or not self.agent:
            return
        if self.input_sample_rate != 16000:
            audio_data = self._resample_audio(
                audio_data, self.input_sample_rate, 16000
            )
        self._audio_buffer.extend(audio_data)

    async def _audio_buffer_flusher(self):
        """Periodically flush buffered phone audio to VoiceAgent."""
        try:
            while not self._closed:
                await asyncio.sleep(self._BUFFER_FLUSH_INTERVAL)
                if self._audio_buffer and self.agent:
                    chunk = bytes(self._audio_buffer)
                    self._audio_buffer.clear()
                    try:
                        await self.agent.send_audio(chunk)
                    except Exception as e:
                        logger.error(f"Error sending audio to agent: {e}")
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def close(self):
        """Clean up bridge, agent, and push fallback disposition."""
        if self._closed:
            return
        self._closed = True

        if self._buffer_flush_task:
            self._buffer_flush_task.cancel()
            try:
                await self._buffer_flush_task
            except asyncio.CancelledError:
                pass

        # Fallback: write sheet + push disposition if survey_complete never fired
        if not self._sheet_written and self.agent:
            transcript = self._build_transcript()
            if transcript:
                self._sheet_written = True
                try:
                    from survey_storage import (
                        extract_answers_from_transcript,
                        append_survey_to_sheet,
                    )
                    detected_gender_raw = getattr(self.agent, "detected_gender", "unknown")
                    gender_label = {"male": "M", "female": "F"}.get(
                        detected_gender_raw, "Unknown"
                    )
                    answers = extract_answers_from_transcript(
                        transcript, detected_gender=gender_label
                    )
                    append_survey_to_sheet(self.call_id, answers, transcript)
                    await self._push_vicidial_disposition(answers, transcript)
                    logger.info(f"📝 Fallback write: call_id={self.call_id}")
                except Exception as e:
                    logger.exception("Fallback storage failed: %s", e)

        if self.agent:
            try:
                await self.agent.end_conversation()
            except Exception as e:
                logger.error(f"Error closing agent: {e}")

        logger.info(
            f"🧹 Bridge closed: call_sid={self.call_sid} "
            f"lead={self.lead_id} session={self.session_id}"
        )

    # =========================================================================
    # AUDIO UTILITIES
    # =========================================================================

    @staticmethod
    def _resample_audio(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Linear interpolation resampling for 16-bit signed LE PCM."""
        if from_rate == to_rate:
            return pcm_data
        if len(pcm_data) < 2:
            return pcm_data

        num_samples = len(pcm_data) // 2
        try:
            samples = struct.unpack(f"<{num_samples}h", pcm_data[: num_samples * 2])
        except struct.error:
            return pcm_data

        ratio = to_rate / from_rate
        out_len = int(num_samples * ratio)
        if out_len == 0:
            return b""

        output = []
        for i in range(out_len):
            src_idx = i / ratio
            idx = int(src_idx)
            frac = src_idx - idx
            if idx + 1 < num_samples:
                sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            elif idx < num_samples:
                sample = samples[idx]
            else:
                break
            output.append(max(-32768, min(32767, sample)))

        return struct.pack(f"<{len(output)}h", *output)
