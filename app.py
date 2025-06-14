import asyncio
import base64
import json
import logging
from typing import Dict, Any
from sarvam import SarvamClient
from dotenv import load_dotenv
import os
# import audioop
# import wave
# import io
from twilio.request_validator import RequestValidator

from fastapi import FastAPI, WebSocket, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Twilio request validator
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
requestValidator = RequestValidator(auth_token)

app = FastAPI()

class VoiceAgent:
    def __init__(self):
        self.sarvam_client = SarvamClient(api_key=os.environ.get("SARVAM_API_KEY"))
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
    async def process_audio_chunk(self, call_sid: str, audio_data: bytes) -> bytes:
        """
        Process incoming audio chunk and return response audio
        This is where you'll integrate:
        1. Sarvam AI Speech-to-Text
        2. LLM processing
        3. Sarvam AI Text-to-Speech
        """
        try:
            # Convert to text
            transcription = await self.sarvam_client.speech_to_text_from_ulaw(audio_data)
            
            if transcription:
                print(f"Transcription for call {call_sid}: {transcription}")
                # Get LLM response
                llm_response = await self.sarvam_client.get_llm_response(transcription)
                
                print(f"LLM response for call {call_sid}: {llm_response}")
                
                if llm_response is not None:
                    # Convert to speech
                    response_audio = await self.sarvam_client.text_to_speech_for_twilio(llm_response)
                    return response_audio
                
                return b'\xff' * len(audio_data)
            
            return b'\xff' * len(audio_data)  # Silence if no transcription
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return b'\xff' * len(audio_data)

# Initialize voice agent
voice_agent = VoiceAgent()

@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio voice webhook endpoint
    """
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    
    isValidRequest = requestValidator.validate(url, form_data, signature)
    
    if not isValidRequest:
        logger.error("Invalid Twilio request signature")
        return HTTPException(status_code=403, detail={"error": "Invalid signature"})
    
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
    voice_agent.active_calls[call_sid] = {
        "websocket": websocket,
        "stream_sid": None,
        "audio_buffer": b"",
        "sequence_number": 0
    }
    
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
                await handle_media_event(call_sid, data)
                
            elif event_type == "stop":
                logger.info(f"Stream stopped for call: {call_sid}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for call {call_sid}: {e}")
    finally:
        # Cleanup
        if call_sid in voice_agent.active_calls:
            del voice_agent.active_calls[call_sid]
        logger.info(f"WebSocket connection closed for call: {call_sid}")

async def handle_media_event(call_sid: str, data: Dict[str, Any]):
    """
    Handle incoming media (audio) from Twilio
    """
    try:
        call_data = voice_agent.active_calls.get(call_sid)
        if not call_data:
            return
            
        websocket = call_data["websocket"]
        
        # Get audio payload (base64 encoded μ-law)
        payload = data.get("media", {}).get("payload", "")
        if not payload:
            return
            
        # Decode audio data
        audio_data = base64.b64decode(payload)
        
        # Add to buffer
        call_data["audio_buffer"] += audio_data
        
        # Process audio chunks (e.g., every 1 second of audio)
        # μ-law 8kHz = 8000 bytes per second
        chunk_size = 8000 * 10 # 10 seconds of audio
        
        if len(call_data["audio_buffer"]) >= chunk_size:
            # Extract chunk for processing
            chunk = call_data["audio_buffer"][:chunk_size]
            call_data["audio_buffer"] = call_data["audio_buffer"][chunk_size:]
            
            # Process the audio chunk (STT -> LLM -> TTS)
            response_audio = await voice_agent.process_audio_chunk(call_sid, chunk)
            
            # Send response audio back to Twilio
            if response_audio:
                await send_audio_to_twilio(websocket, response_audio, call_data)
                
    except Exception as e:
        logger.error(f"Error handling media event for call {call_sid}: {e}")

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