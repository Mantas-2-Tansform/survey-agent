import asyncio
import base64
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, Optional
from websockets.asyncio.client import connect as ws_connect
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import numpy as np

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Voice survey agent using Gemini Live. conversation_history holds full transcript for storage."""
    
    def __init__(
        self,
        model_resource: str,
        ws_url: str,
        system_prompt: str,
        response_callback: Optional[Callable] = None,
        inactivity_timeout: int = 90,
        silence_detection: float = 1.2,
    ):
        """
        Initialize the voice agent.
        
        Args:
            model_resource: Full Vertex AI model resource path
            ws_url: WebSocket URL for Gemini Live
            system_prompt: System instructions for the AI
            response_callback: Async callback for sending responses to client
            inactivity_timeout: Seconds of silence before ending conversation
            silence_detection: Seconds of user silence before AI responds
        """
        self.model_resource = model_resource
        self.ws_url = ws_url
        self.system_prompt = system_prompt
        self.response_callback = response_callback
        self.inactivity_timeout = inactivity_timeout
        self.silence_detection = silence_detection
        
        # Connection state
        self.websocket = None
        self.conversation_active = False
        self.last_audio_time = time.time()
        
        # Audio processing
        self.audio_buffer = []
        self.last_send_time = 0
        self.min_chunk_interval_ms = 200
        
        # Conversation tracking
        self.conversation_history = []
        self.turn_number = 0
        self._survey_complete_sent = False  # Prevent duplicate end triggers
        # Transcript buffering - accumulate word chunks into full turns
        self._ai_turn_buf = ""       # Accumulates AI word chunks until turnComplete
        self._user_utt_buf = ""      # Accumulates user speech until they stop talking

        # Voice-based gender detection
        self._gender_sample_buffer = bytearray()   # Raw PCM bytes from user mic
        self._gender_analysis_triggered = False     # Only analyse once per call
        self.detected_gender = "unknown"            # Result stored here
        self.gender_confidence = 0.0
        self._ai_is_speaking = False  # True while AI audio is being streamed out; blocks gender buffering
        
        # Tasks
        self.receive_task = None
        self.inactivity_task = None
        
    async def start_conversation(self):
        """Initialize WebSocket connection and start the conversation."""
        try:
            # Get Google Cloud credentials
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(GoogleAuthRequest())
            
            # Connect to Gemini Live
            self.websocket = await ws_connect(
                self.ws_url,
                additional_headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {credentials.token}",
                },
            )
            
            logger.info("✅ Connected to Gemini Live WebSocket")
            
            # Send initial setup message
            # IMPORTANT: input_audio_transcription / output_audio_transcription are
            # siblings of generation_config inside "setup" — NOT nested inside it.
            setup_message = {
                "setup": {
                    "model": self.model_resource,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": "Kore"
                                }
                            }
                        },
                    },
                    "system_instruction": {
                        "parts": [{"text": self.system_prompt}]
                    },
                    # Transcription at setup level (sibling of generation_config)
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                    # Prevent unprompted bot speech
                    "proactivity": {"proactive_audio": False},
                }
            }
            
            await self.websocket.send(json.dumps(setup_message))
            logger.info("📤 Sent setup message to Gemini")
            
            # Wait for setup complete
            setup_response = await self.websocket.recv()
            setup_data = json.loads(setup_response)
            
            if "setupComplete" in setup_data:
                logger.info("✅ Setup complete, conversation ready")
                self.conversation_active = True
                
                # Start background tasks
                self.receive_task = asyncio.create_task(self._receive_loop())
                self.inactivity_task = asyncio.create_task(self._monitor_inactivity())
                
                # Send ready notification to client
                if self.response_callback:
                    await self.response_callback({
                        "type": "ready",
                        "message": "Voice bot is ready to talk!"
                    })
                
                # Trigger bot to speak first (send initial user turn so model responds with intro)
                await self._send_initial_trigger()
                    
                return True
            else:
                logger.error(f"Setup failed: {setup_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start conversation: {e}", exc_info=True)
            if self.response_callback:
                await self.response_callback({
                    "type": "error",
                    "message": f"Failed to start: {str(e)}"
                })
            return False
    
    async def _receive_loop(self):
        """Continuously receive and process messages from Gemini."""
        try:
            while self.conversation_active and self.websocket:
                try:
                    raw_message = await self.websocket.recv()
                    message = json.loads(raw_message)
                    await self._handle_gemini_message(message)
                    
                except Exception as e:
                    if self.conversation_active:
                        logger.error(f"Error in receive loop: {e}")
                        
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in receive loop: {e}", exc_info=True)
    
    async def _handle_gemini_message(self, message: Dict[str, Any]):
        """Process messages from Gemini."""
        try:
            server_content = message.get("serverContent") or message.get("server_content")
            if not server_content:
                logger.debug(f"No serverContent in message keys: {list(message.keys())}")
                return

            # ── 1. Output transcription chunks (buffer until turnComplete) ──
            out_tx = server_content.get("outputTranscription") or server_content.get("output_transcription")
            if out_tx:
                chunk = (out_tx.get("text") or "").strip()
                if chunk:
                    # Accumulate into buffer with a space separator
                    self._ai_turn_buf = (self._ai_turn_buf + " " + chunk).strip()
                    # Stream each chunk live to the UI transcript panel
                    if self.response_callback:
                        await self.response_callback({
                            "type": "transcript_chunk",
                            "role": "assistant",
                            "text": chunk
                        })

            # ── 2. Input transcription (user speech) ──
            inp_tx = server_content.get("inputTranscription") or server_content.get("input_transcription")
            if inp_tx:
                user_text = (inp_tx.get("text") or "").strip()
                is_final = inp_tx.get("isFinal") or inp_tx.get("is_final") or inp_tx.get("final") or False
                if user_text:
                    # Accumulate user speech
                    if not self._user_utt_buf:
                        self._user_utt_buf = user_text
                    elif not self._user_utt_buf.endswith(user_text):
                        self._user_utt_buf = (self._user_utt_buf + " " + user_text).strip()
                    logger.info(f"👤 User ({'final' if is_final else 'interim'}): {user_text}")
                    # Stream to UI
                    if self.response_callback:
                        await self.response_callback({
                            "type": "transcript_chunk",
                            "role": "user",
                            "text": user_text,
                            "is_final": is_final
                        })
                    # Flush user buffer to history on final
                    if is_final and self._user_utt_buf:
                        self.conversation_history.append({
                            "role": "user",
                            "text": self._user_utt_buf,
                            "turn": self.turn_number,
                            "timestamp": time.time()
                        })
                        self.turn_number += 1
                        self._user_utt_buf = ""

            # ── 3. modelTurn inline text (fallback when transcription not available) ──
            model_turn = server_content.get("modelTurn") or server_content.get("model_turn")
            if model_turn and not out_tx:
                for part in model_turn.get("parts", []):
                    if "text" in part:
                        chunk = part["text"].strip()
                        if chunk:
                            self._ai_turn_buf = (self._ai_turn_buf + " " + chunk).strip()
                            if self.response_callback:
                                await self.response_callback({
                                    "type": "transcript_chunk",
                                    "role": "assistant",
                                    "text": chunk
                                })

            # ── 4. Audio chunks ──
            audio_data_out, sample_rate = self._decode_audio_output(message)
            if audio_data_out:
                self._ai_is_speaking = True  # AI is outputting audio — block gender buffering
                if self.response_callback:
                    await self.response_callback({
                        "type": "audio",
                        "data": base64.b64encode(audio_data_out).decode(),
                        "sample_rate": sample_rate,
                        "format": "pcm16"
                    })

            # ── 5. Turn complete — flush AI buffer to history ──
            tc = server_content.get("turnComplete") or server_content.get("turn_complete")
            if tc:
                self._ai_is_speaking = False  # AI finished — user audio may resume for gender buffering
                full_ai_text = self._ai_turn_buf.strip()
                self._ai_turn_buf = ""  # Reset buffer

                if full_ai_text:
                    self.conversation_history.append({
                        "role": "assistant",
                        "text": full_ai_text,
                        "turn": self.turn_number,
                        "timestamp": time.time()
                    })
                    self.turn_number += 1
                    logger.info(f"🤖 AI (full turn): {full_ai_text}")

                    # Also flush any pending user speech before this turn
                    if self._user_utt_buf:
                        # Insert user turn before AI turn
                        self.conversation_history.insert(-1, {
                            "role": "user",
                            "text": self._user_utt_buf,
                            "turn": self.turn_number,
                            "timestamp": time.time()
                        })
                        self.turn_number += 1
                        self._user_utt_buf = ""

                    # Send final assembled turn to UI
                    if self.response_callback:
                        await self.response_callback({
                            "type": "transcript",
                            "role": "assistant",
                            "text": full_ai_text
                        })

                    # Check goodbye trigger on full assembled text
                    if self._is_goodbye(full_ai_text) and not self._survey_complete_sent:
                        self._survey_complete_sent = True
                        logger.info("👋 Goodbye detected – triggering call end")
                        if self.response_callback:
                            await self.response_callback({"type": "survey_complete"})

                logger.info("✅ AI turn complete")
                if self.response_callback:
                    await self.response_callback({"type": "turn_complete"})

            # ── 6. Interrupted ──
            if server_content.get("interrupted"):
                logger.info("🤚 Bot interrupted by user")
                self._ai_is_speaking = False  # Interrupted — user audio resumes
                # Flush user buffer to history before clearing AI buffer
                if self._user_utt_buf:
                    self.conversation_history.append({
                        "role": "user",
                        "text": self._user_utt_buf,
                        "turn": self.turn_number,
                        "timestamp": time.time()
                    })
                    self.turn_number += 1
                    self._user_utt_buf = ""
                self._ai_turn_buf = ""  # Discard incomplete AI turn
                if self.response_callback:
                    await self.response_callback({"type": "interrupted"})

        except Exception as e:
            logger.error(f"Error handling Gemini message: {e}", exc_info=True)

    def _is_goodbye(self, text: str) -> bool:
        """
        Return True only when the bot is clearly ending the call.
        Only triggers on explicit farewell words near the end of the utterance.
        Deliberately excludes 'take care', 'have a great day', 'thank you for your time'
        as these all appear naturally mid-conversation and cause false positives.
        """
        text_lower = text.lower().strip()
        # Only these words are unambiguous end-of-call signals
        farewell_words = ["goodbye", "good bye", "bye bye", "good night"]
        has_farewell = any(w in text_lower for w in farewell_words)
        if not has_farewell:
            return False
        # Must appear in the final 80 characters (i.e. the bot is signing off, not mid-sentence)
        tail = text_lower[-80:]
        return any(w in tail for w in farewell_words)
    
    def _decode_audio_output(self, response: dict) -> tuple[bytes, int]:
        """Extract audio and sample rate from Gemini response."""
        result_bytes = []
        sample_rate = 24000
        
        server_content = response.get('serverContent', {})
        model_turn = server_content.get('modelTurn', {})
        
        for part in model_turn.get('parts', []):
            inline_data = part.get('inlineData', {})
            mime_type_str = inline_data.get('mimeType', '')
            
            if 'audio' in mime_type_str.lower():
                # Extract sample rate if available
                if 'rate=' in mime_type_str:
                    try:
                        rate_str = mime_type_str.split('rate=')[1]
                        sample_rate = int(rate_str)
                    except (ValueError, IndexError):
                        pass
                
                # Decode audio data
                data = inline_data.get('data', '')
                if data:
                    try:
                        decoded_chunk = base64.b64decode(data)
                        result_bytes.append(decoded_chunk)
                    except Exception as e:
                        logger.error(f"Error decoding audio: {e}")
        
        return b''.join(result_bytes), sample_rate
    
    async def send_audio(self, audio_data: bytes):
        """Send user audio to Gemini and buffer for gender detection."""
        if not self.conversation_active or not self.websocket:
            return
        
        try:
            self.last_audio_time = time.time()
            self.audio_buffer.append(audio_data)

            # ── Gender detection: buffer ~3 seconds of VALID user speech ──
            # Fix: Only buffer if AI is quiet AND the audio isn't just silence
            if not self._gender_analysis_triggered and not self._ai_is_speaking:
                
                # Check if this specific chunk contains actual speech (volume check)
                if self._is_speech(audio_data):
                    self._gender_sample_buffer.extend(audio_data)
                    
                    # Log progress occasionally so we know it's working
                    if len(self._gender_sample_buffer) % 16000 == 0:
                        logger.debug(f"🎙️ Gender buffer progress: {len(self._gender_sample_buffer)}/96000 bytes")

                if len(self._gender_sample_buffer) >= 96000:
                    self._gender_analysis_triggered = True
                    logger.info("🎙️ Collected 3s of active user speech – launching gender classification")
                    asyncio.create_task(
                        self._classify_gender_via_async_llm(
                            bytearray(self._gender_sample_buffer[:96000])
                        )
                    )
            
            # Send buffered audio to Gemini
            current_time = time.time() * 1000
            if current_time - self.last_send_time >= self.min_chunk_interval_ms:
                await self._flush_audio()
                
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    def _is_speech(self, audio_data: bytes, threshold: int = 500) -> bool:
        """
        Simple RMS (Root Mean Square) volume check to distinguish 
        actual speech from silence/background hum.
        """
        if not audio_data:
            return False
        # Convert bytes to Int16 array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # Calculate RMS (Volume)
        rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
        return rms > threshold
    
    async def _flush_audio(self, force: bool = False):
        """Send buffered audio to Gemini."""
        if not self.audio_buffer:
            return
        
        try:
            combined_audio = b''.join(self.audio_buffer)
            self.audio_buffer.clear()
            
            if not combined_audio:
                return
            
            # Encode and send
            encoded_audio = base64.b64encode(combined_audio).decode()
            
            message = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": encoded_audio
                        }
                    ]
                }
            }
            
            await self.websocket.send(json.dumps(message))
            self.last_send_time = time.time() * 1000
            
        except Exception as e:
            logger.error(f"Error flushing audio: {e}")
    
    def _generate_wav_header(self, sample_rate: int, num_frames: int,
                             num_channels: int, sample_width: int) -> bytes:
        """Build a WAV file header for the given PCM properties."""
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        return struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + num_frames * block_align,
            b'WAVE',
            b'fmt ',
            16, 1,
            num_channels, sample_rate,
            byte_rate, block_align,
            sample_width * 8,
            b'data',
            num_frames * block_align,
        )

    async def _classify_gender_via_async_llm(self, audio_buffer: bytearray):
        """
        Send ~3 seconds of PCM audio to Gemini (text model) to detect speaker gender.
        Fires once per call, runs in a thread so it never blocks the event loop.
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part
            from config import VERTEX_PROJECT_ID, VERTEX_LOCATION

            sample_rate = 16000
            sample_width = 2
            duration_sec = len(audio_buffer) / float(sample_rate * sample_width)
            logger.info(f"🎙️ Gender sample duration: {duration_sec:.2f}s")

            if duration_sec < 1.0:
                logger.info("🎙️ Sample too short – skipping gender classification")
                return

            # Build a proper WAV file from raw PCM
            header = self._generate_wav_header(
                sample_rate=sample_rate,
                num_frames=len(audio_buffer) // sample_width,
                num_channels=1,
                sample_width=sample_width,
            )
            wav_data = header + bytes(audio_buffer)

            prompt = (
                "Analyze ONLY the acoustic properties of the attached audio. "
                "Classify the speaker's perceived gender as 'male', 'female', or 'unknown'. "
                "Respond with ONLY a JSON object, no other text:\n"
                '{"gender": "male" or "female" or "unknown", "confidence": 0.0 to 1.0}'
            )

            # Try models in order (same fallback strategy as survey_storage)
            candidate_models = ["gemini-2.0-flash-001", "gemini-2.0-flash",
                                 "gemini-1.5-flash-002", "gemini-1.5-flash-001"]

            def sync_classify():
                vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
                audio_part = Part.from_data(data=wav_data, mime_type="audio/wav")
                last_err = None
                for model_id in candidate_models:
                    try:
                        model = GenerativeModel(model_id)
                        resp = model.generate_content(contents=[prompt, audio_part])
                        return resp.text.strip()
                    except Exception as e:
                        last_err = e
                raise Exception(f"No model available for gender classification: {last_err}")

            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1), sync_classify
            )

            # Strip markdown fences if present
            if "```" in response_text:
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(response_text)
            detected = (result.get("gender") or "unknown").lower()
            confidence = float(result.get("confidence", 0.0))

            logger.info(f"🎙️ ✅ Gender detected: {detected} (confidence: {confidence:.2f})")

            # Store on agent for use in transcript/sheet
            self.detected_gender = detected
            self.gender_confidence = confidence

            # Map to M/F/Unknown for the sheet
            gender_label = {"male": "M", "female": "F"}.get(detected, "Unknown")

            # Notify client so UI can display it immediately
            if self.response_callback:
                await self.response_callback({
                    "type": "voice_gender_update",
                    "detected_gender": detected,
                    "gender_label": gender_label,
                    "confidence": round(confidence, 2),
                })

        except Exception as e:
            logger.error(f"🎙️ Gender classification failed: {e}", exc_info=True)

    async def _send_initial_trigger(self):
        """Send a minimal user turn so the model speaks first with the talk-off."""
        try:
            if not self.websocket:
                return
            trigger = {
                "clientContent": {
                    "turns": [
                        {"role": "user", "parts": [{"text": "Begin the survey."}]}
                    ],
                    "turnComplete": True
                }
            }
            await self.websocket.send(json.dumps(trigger))
            logger.info("📤 Sent initial trigger so bot speaks first")
        except Exception as e:
            logger.error(f"Error sending initial trigger: {e}")
    
    async def _monitor_inactivity(self):
        """Monitor for inactivity and end conversation if timeout exceeded."""
        try:
            while self.conversation_active:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if time.time() - self.last_audio_time > self.inactivity_timeout:
                    logger.info("⏰ Inactivity timeout reached")
                    
                    if self.response_callback:
                        await self.response_callback({
                            "type": "inactivity_timeout",
                            "message": "Conversation ended due to inactivity"
                        })
                    
                    await self.end_conversation()
                    break
                    
        except asyncio.CancelledError:
            logger.info("Inactivity monitor cancelled")
    
    async def end_conversation(self):
        """End the conversation and cleanup."""
        if not self.conversation_active:
            return
        
        logger.info("🛑 Ending conversation")
        self.conversation_active = False
        
        try:
            # Flush any remaining audio
            await self._flush_audio(force=True)
            
            # Cancel tasks
            if self.receive_task:
                self.receive_task.cancel()
            if self.inactivity_task:
                self.inactivity_task.cancel()
            
            # Close WebSocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            # Send final notification
            if self.response_callback:
                await self.response_callback({
                    "type": "conversation_ended",
                    "turns": self.turn_number,
                    "duration": time.time() - (
                        self.conversation_history[0]["timestamp"] 
                        if self.conversation_history else time.time()
                    )
                })
            
            logger.info("✅ Conversation ended successfully")
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}", exc_info=True)