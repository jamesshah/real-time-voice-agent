import asyncio
import base64
import json
import logging
from typing import Dict, Any

import numpy as np
import torch
from sarvam import SarvamClient
from dotenv import load_dotenv
import os
import audioop
# import wave
# import io

from fastapi import FastAPI, WebSocket, Request, Response, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import uvicorn
from silero_vad import load_silero_vad

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- VAD Configuration ---
# These parameters are tunable.
VAD_THRESHOLD = 0.5  # Speech probability threshold.
SILENCE_TIMEOUT_MS = 800 # How long of a silence triggers the end of a phrase.
TWILIO_SAMPLE_RATE = 8000 # Twilio sends audio at 8000 Hz.
VAD_SAMPLE_RATE = 16000
# Calculate the number of silent frames to wait for before processing.
# Twilio sends media packets every 20ms.
FRAMES_PER_20MS_TWILIO = int(TWILIO_SAMPLE_RATE * 0.02) # 160 samples at 8kHz
FRAMES_PER_20MS_16K = int(VAD_SAMPLE_RATE * 0.02) # 320 samples at 16kHz
SILENCE_FRAMES_THRESHOLD = int(SILENCE_TIMEOUT_MS / 20) # Number of 20ms silent chunks
VAD_CHUNK_SIZE_SAMPLES = 512 


# Setup VAD with silero-vad
try:
    # Use the official silero-vad package function to load the model    
    model = load_silero_vad()

except Exception as e:
    logger.error(f"Failed to load Silero VAD model. Error: {e}")
    model = None

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
        "is_speaking": False,
        "speech_buffer": b"", # Buffers the complete user utterance (µ-law 8kHz)
        "vad_audio_buffer": np.array([], dtype=np.int16), # Buffer for 16kHz PCM for VAD
        "silence_frames": 0, # Counts consecutive silent frames
        "resampler_state": None # Store the state for audioop.ratecv
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
                if model:
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

        # 1. Decode Twilio's µ-law audio from base64
        audio_mulaw = base64.b64decode(payload)

        # 2. Convert µ-law to 16-bit linear PCM for the VAD
        audio_pcm = audioop.ulaw2lin(audio_mulaw, 2)

        # 3. Resample from 8kHz to 16kHz for Silero VAD, maintaining state
        audio_pcm_16k, call_data["resampler_state"] = audioop.ratecv(
            audio_pcm, 2, 1, TWILIO_SAMPLE_RATE, VAD_SAMPLE_RATE, call_data["resampler_state"]
        )
        
        audio_np_16k = np.frombuffer(audio_pcm_16k, dtype=np.int16)


        # Accumulate 16kHz PCM for VAD
        call_data["vad_audio_buffer"] = np.concatenate((call_data["vad_audio_buffer"], audio_np_16k))

        while len(call_data["vad_audio_buffer"]) >= VAD_CHUNK_SIZE_SAMPLES:
            # Take a chunk of exactly VAD_CHUNK_SIZE_SAMPLES for VAD
            vad_chunk_np = call_data["vad_audio_buffer"][:VAD_CHUNK_SIZE_SAMPLES]
            
            # 4. Convert to a PyTorch tensor
            # Ensure it's 1D and float32, normalized to [-1, 1]
            audio_tensor = torch.from_numpy(vad_chunk_np.astype(np.float32)) / 32768.0
            
            # logger.debug(f"Processing audio chunk for call {call_sid} with shape {audio_tensor.shape}")

            # 5. Get speech probability from the loaded model
            speech_prob = model(audio_tensor, VAD_SAMPLE_RATE).item()
            
            # Remove the processed chunk from the buffer
            call_data["vad_audio_buffer"] = call_data["vad_audio_buffer"][VAD_CHUNK_SIZE_SAMPLES:]

            # 6. VAD Logic: Detect start and end of speech
            if speech_prob > VAD_THRESHOLD:
                # Speech detected
                if not call_data["is_speaking"]:
                    logger.info(f"Speech start detected for call {call_sid}.")
                call_data["is_speaking"] = True
                call_data["silence_frames"] = 0
                # Append the original µ-law data to our speech buffer
                call_data["speech_buffer"] += audio_mulaw
            else:
                # Silence detected
                if call_data["is_speaking"]:
                    call_data["silence_frames"] += 1
                    if call_data["silence_frames"] > SILENCE_FRAMES_THRESHOLD:
                        # End of utterance detected
                        logger.info(f"End of speech detected for call {call_sid}.")

                        # Get the complete utterance
                        full_utterance = call_data["speech_buffer"]

                        # Reset state for the next utterance
                        call_data["speech_buffer"] = b""
                        call_data["is_speaking"] = False
                        call_data["silence_frames"] = 0

                        # --- Process the audio in the background ---
                        if full_utterance:
                            asyncio.create_task(process_and_respond(call_sid, full_utterance, call_data))
                else:
                    # If not speaking and silence is detected, just keep accumulating
                    # if you want to buffer before the first speech detection,
                    # but for most conversational agents, you discard leading silence.
                    # Here, we discard it by not adding to speech_buffer until speaking starts.
                    pass # Do nothing, just wait for speech to start
                
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