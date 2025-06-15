import asyncio
import base64
import json
import logging
from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv
import os
import sys
from fastapi import FastAPI, WebSocket, Request, Response, WebSocketDisconnect
import uvicorn
from vad import  process_audio_with_vad
from agent import VoiceAgent

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    # Initialize voice agent
    voice_agent = VoiceAgent(logger)
except Exception as e:
    logger.error(f"Failed to initialize VoiceAgent: {e}")
    sys.exit(1)

@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio voice webhook endpoint
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    
    logger.info(f"Incoming call: {call_sid}")
    
    # TwiML response to start bidirectional stream
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Hello! I'm connecting you to our voice agent.</Say>
    <Connect>
        <Stream url="wss://{request.headers.get('host')}/websocket/stream/{call_sid}" />
    </Connect>
</Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")

@app.websocket("/websocket/stream/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """
    WebSocket endpoint for Twilio bidirectional streams
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for call: {call_sid}")
    
    # Initialize call data
    voice_agent.initialize_call(call_sid, websocket)
    
    try:
        while True:
            # Receive message from Twilio
            message = await websocket.receive_text()
            data = json.loads(message)
            
            event_type = data.get("event")
            
            if event_type == "connected":
                logger.info(f"Stream connected for call: {call_sid}")
                
            elif event_type == "start":
                stream_sid = data.get("streamSid")
                voice_agent.active_calls[call_sid]["stream_sid"] = stream_sid
                logger.info(f"Stream started: {stream_sid} for call: {call_sid}")
                
            elif event_type == "media":
                # Handle incoming audio
                if voice_agent.vad_model:
                    await handle_media_event(call_sid, data)
                else:
                    logger.error("Silero VAD model is not loaded. Cannot process media events.")
                
            elif event_type == "stop":
                logger.info(f"Stream stopped for call: {call_sid}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client for call {call_sid}.")
        
    except Exception as e:
        logger.error(f"WebSocket error for call {call_sid}: {e}")
    finally:
        # Cleanup
        if call_sid in voice_agent.active_calls:
            del voice_agent.active_calls[call_sid]
        
        # Optional: Delete the recordings directory for this call
        # recordings_dir = voice_agent.create_recordings_directory(call_sid)
        # if os.path.exists(recordings_dir):
        #     try:
        #         os.rmdir(recordings_dir)  # Remove the directory if empty
        #     except OSError as e:
        #         logger.error(f"Error removing recordings directory {recordings_dir}: {e}")
        
        logger.info(f"WebSocket connection closed for call: {call_sid}")

async def handle_media_event(call_sid: str, data: Dict[str, Any]):
    """
    Handles incoming audio from Twilio, performs VAD, and triggers processing.
    """
    try:
        call_data = voice_agent.active_calls.get(call_sid)
        if not call_data:
            return

        payload = data.get("media", {}).get("payload", "")
        if not payload:
            return

        # Decode Twilio's µ-law audio from base64
        audio_mulaw = base64.b64decode(payload)

        # Process audio with VAD
        vad_result = process_audio_with_vad(
            voice_agent.vad_model, audio_mulaw, call_data
        )

        if vad_result["end_of_speech"]:
            # Get the complete utterance
            full_utterance = vad_result["speech_buffer"]

            # Reset state for the next utterance
            call_data["speech_buffer"] = b""
            call_data["is_speaking"] = False
            call_data["silence_frames"] = 0

            # Process the audio in the background
            if full_utterance:
                asyncio.create_task(process_and_respond(call_sid, full_utterance, call_data))

    except Exception as e:
        logger.error(f"Error handling media event for call {call_sid}: {e}")

async def process_and_respond(call_sid: str, full_utterance: bytes, call_data: Dict[str, Any]):

    try:
        response_audio = await voice_agent.process_audio_chunk(call_sid, full_utterance)
        
        # Send response audio back to Twilio
        if response_audio and call_data.get("websocket"):
            await send_audio_to_twilio(
                call_data["websocket"],
                response_audio,
                call_data
            )
            
    except Exception as e:
        logger.error(f"Error in process_and_respond for call {call_sid}: {e}")

async def send_audio_to_twilio(websocket: WebSocket, audio_data: bytes, call_data: Dict[str, Any]):
    """
    Send audio data back to Twilio stream
    """
    try:
        # Encode audio as base64
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Create media message
        media_message = {
            "event": "media",
            "streamSid": call_data["stream_sid"],
            "media": {
                "payload": encoded_audio
            }
        }
        
        # Send to Twilio
        await websocket.send_text(json.dumps(media_message))
        logger.debug(f"Sent audio chunk to stream: {call_data['stream_sid']}")
        
    except Exception as e:
        logger.error(f"Error sending audio to Twilio: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )