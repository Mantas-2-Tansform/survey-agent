from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import logging
import json
import uuid
from typing import Dict

from config import (
    PROJECT_NAME, VERSION, CORS_ORIGINS, SERVER_HOST, SERVER_PORT,
    VERTEX_MODEL_RESOURCE, VERTEX_WS_URL, SYSTEM_PROMPT,
    INACTIVITY_TIMEOUT_SECONDS, USER_SILENCE_DETECTION_SECONDS
)
from agent import VoiceAgent
from survey_storage import extract_answers_from_transcript, append_survey_to_sheet

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store active agent connections and call IDs
active_agents: Dict[str, VoiceAgent] = {}
active_call_ids: Dict[str, str] = {}


def _build_transcript(agent) -> str:
    """Build a clean, readable transcript from agent conversation history."""
    lines = []
    for h in getattr(agent, "conversation_history", []):
        role_label = "Sneha" if h["role"] == "assistant" else "Respondent"
        lines.append(f"{role_label}: {h['text']}")
    return "\n".join(lines).strip()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=PROJECT_NAME,
        version=VERSION,
        description="ABC Survey voice agent – mood of the nation (Gemini Live)"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "message": f"{PROJECT_NAME} is running",
            "version": VERSION,
            "status": "healthy"
        }
    
    @app.get("/client", response_class=HTMLResponse)
    async def get_client():
        """Serve the web client interface."""
        # Read and return the HTML file
        try:
            with open("client.html", "r") as f:
                return f.read()
        except FileNotFoundError:
            return "<h1>Client interface not found</h1><p>Please ensure client.html exists</p>"
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for survey voice conversation."""
        await websocket.accept()
        session_id = str(id(websocket))
        call_id = str(uuid.uuid4())
        active_call_ids[session_id] = call_id
        logger.info(f"🔌 WebSocket connected: session={session_id}, call_id={call_id}")
        
        # Create callback for sending messages to client (and handle survey_complete)
        async def send_to_client(message: dict):
            try:
                if message.get("type") == "survey_complete":
                    agent = active_agents.get(session_id)
                    cid = active_call_ids.get(session_id, "")
                    # Guard: only write to sheet once
                    if getattr(agent, "_sheet_written", False):
                        logger.info("Sheet already written – skipping duplicate survey_complete")
                        return
                    if agent:
                        agent._sheet_written = True
                    if agent and cid:
                        transcript = _build_transcript(agent)
                        # Get voice-detected gender (M/F/Unknown)
                        detected_gender_raw = getattr(agent, "detected_gender", "unknown")
                        gender_label = {"male": "M", "female": "F"}.get(detected_gender_raw, "Unknown")
                        answers = {}
                        try:
                            answers = extract_answers_from_transcript(transcript, detected_gender=gender_label)
                            append_survey_to_sheet(cid, answers, transcript)
                        except Exception as e:
                            logger.exception("Survey storage failed: %s", e)
                        try:
                            await websocket.send_json({
                                "type": "survey_result",
                                "call_id": cid,
                                "transcript": transcript,
                                "answers": answers,
                                "detected_gender": detected_gender_raw,
                                "gender_label": gender_label,
                            })
                        except Exception:
                            pass
                    # Give the client a moment to receive survey_result before we close
                    await asyncio.sleep(1.5)
                    if agent:
                        await agent.end_conversation()
                    return
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.debug(f"Could not send to client (likely disconnected): {e}")
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
        
        # Create agent instance
        agent = VoiceAgent(
            model_resource=VERTEX_MODEL_RESOURCE,
            ws_url=VERTEX_WS_URL,
            system_prompt=SYSTEM_PROMPT,
            response_callback=send_to_client,
            inactivity_timeout=INACTIVITY_TIMEOUT_SECONDS,
            silence_detection=USER_SILENCE_DETECTION_SECONDS,
        )
        
        active_agents[session_id] = agent
        
        try:
            # Start the conversation
            success = await agent.start_conversation()
            
            if not success:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to start conversation with Gemini"
                })
                await websocket.close()
                return
            
            # Main message loop
            while agent.conversation_active:
                try:
                    message = await websocket.receive()
                    
                    # Handle different message types
                    if "bytes" in message:
                        # Audio data from client
                        audio_data = message["bytes"]
                        await agent.send_audio(audio_data)
                    
                    elif "text" in message:
                        # JSON messages from client
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "end_conversation":
                            logger.info("Client requested to end conversation")
                            await agent.end_conversation()
                            break
                        
                        elif data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected: {session_id}")
                    break

                except RuntimeError as e:
                    # Starlette raises RuntimeError when receive() is called after disconnect
                    if "disconnect" in str(e).lower():
                        logger.info(f"Client already disconnected: {session_id}")
                    else:
                        logger.error(f"RuntimeError in message loop: {e}", exc_info=True)
                    break
                    
                except Exception as e:
                    logger.error(f"Error in message loop: {e}", exc_info=True)
                    break
        
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        
        finally:
            # Only write to sheet here if survey_complete path didn't already do it
            if session_id in active_agents:
                agent = active_agents[session_id]
                already_written = getattr(agent, "_sheet_written", False)
                if getattr(agent, "conversation_active", False):
                    await agent.end_conversation()
                if not already_written:
                    transcript = _build_transcript(agent)
                    cid = active_call_ids.get(session_id, "")
                    if transcript and cid:
                        try:
                            detected_gender_raw = getattr(agent, "detected_gender", "unknown")
                            gender_label = {"male": "M", "female": "F"}.get(detected_gender_raw, "Unknown")
                            answers = extract_answers_from_transcript(transcript, detected_gender=gender_label)
                            append_survey_to_sheet(cid, answers, transcript)
                            logger.info(f"Fallback sheet write for call_id={cid}")
                        except Exception as e:
                            logger.exception("Survey storage failed: %s", e)
                del active_agents[session_id]
            if session_id in active_call_ids:
                del active_call_ids[session_id]
            
            try:
                await websocket.close()
            except Exception:
                pass
            
            logger.info(f"🔌 WebSocket closed: {session_id}")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "application:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True
    )