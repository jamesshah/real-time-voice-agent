from sarvam import SarvamClient
from dotenv import load_dotenv
import os
import wave
import io
import audioop
import random
from typing import Dict, Any
from vad import initialize_vad_model
import logging
from fastapi import WebSocket
import numpy as np

load_dotenv()

class VoiceAgent:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.sarvam_client = SarvamClient(api_key=os.environ.get("SARVAM_API_KEY"))
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        try:
            self.vad_model = initialize_vad_model()
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model. Error: {e}")
            raise RuntimeError("Failed to initialize VAD model") from e            
    
    def initialize_call(self, call_sid: str, websocket: WebSocket) -> None:
        self.active_calls[call_sid] = {
        "websocket": websocket,
        "stream_sid": None,
        "is_speaking": False,
        "speech_buffer": b"", # Buffers the complete user utterance (µ-law 8kHz)
        "vad_audio_buffer": np.array([], dtype=np.int16), # Buffer VAD audio
        "silence_frames": 0, # Counts consecutive silent frames
        "resampler_state": None, # Store the state for audioop.ratecv,
        "messages": [
            {
                "role": "system", 
                "content": os.environ.get("SALES_AGENT_SYSTEM_PROMPT", "You are a helpful sales agent.")
            },
        ]  # Store messages for the call
    }
        
    def add_message(self, call_sid: str, role: str, content: str) -> None:
        """
        Add a message to the call's message history.
        
        Args:
            call_sid: Unique identifier for the call
            role: Role of the message sender (e.g., "user", "assistant")
            content: Content of the message
        """
        if call_sid in self.active_calls:
            self.active_calls[call_sid]["messages"].append({
                "role": role,
                "content": content
            })
        else:
            self.logger.warning(f"Call SID {call_sid} not found in active calls.")
    
    def convert_ulaw_to_wav(self, recordings_dir: str, ulaw_data: bytes, sample_rate: int = 8000) -> str:
        """
        Convert μ-law audio data to WAV format for Sarvam AI
        
        Args:
            ulaw_data: μ-law encoded audio bytes
            sample_rate: Sample rate (default 8000 for Twilio)
            
        Returns:
            WAV formatted audio bytes
        """
        try:
                        
            random_name = 'sample_' + str(random.randint(1, 5000)) + '.wav'
            wav_file_path = os.path.join(recordings_dir, random_name)
            # Convert μ-law to linear PCM (16-bit)
            linear_data = audioop.ulaw2lin(ulaw_data, 2)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(linear_data)
                
            wav_buffer.seek(0)
            with open(wav_file_path, 'wb') as f:
                f.write(wav_buffer.getvalue())
                
            return wav_file_path
            
        except Exception as e:
            self.logger.error(f"Error converting μ-law to WAV: {e}")
            raise e
        
    def convert_wav_to_ulaw(self, wav_data: bytes) -> bytes:
        """
        Convert WAV audio to μ-law format for Twilio
        
        Args:
            wav_data: WAV format audio bytes
            
        Returns:
            μ-law encoded audio bytes
        """
        try:
            # Read WAV data
            wav_buffer = io.BytesIO(wav_data)
            with wave.open(wav_buffer, 'rb') as wav_file:
                # Read audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert to mono if stereo
                if channels == 2:
                    frames = audioop.tomono(frames, sample_width, 1, 1)
                
                # Resample to 8kHz if needed (Twilio requirement)
                if framerate != 8000:
                    frames, _ = audioop.ratecv(frames, sample_width, 1, framerate, 8000, None)
                
                # Convert to μ-law
                ulaw_data = audioop.lin2ulaw(frames, sample_width)
                
                return ulaw_data
            
        except Exception as e:
            self.logger.error(f"Error converting WAV to μ-law: {e}")
            raise e
        
    def create_recordings_directory(self, call_sid: str) -> str:
        """
        Create a directory for storing call recordings.
        
        Args:
            call_sid: Unique identifier for the call
            
        Returns:
            Path to the created directory
        """
        
        try:
            recordings_dir = "recordings"
            if not os.path.exists(recordings_dir):
                os.makedirs(recordings_dir)
            
            call_recording_dir = os.path.join(recordings_dir, call_sid)
            if not os.path.exists(call_recording_dir):
                os.makedirs(call_recording_dir)
            
            return call_recording_dir
        
        except Exception as e:
            self.logger.error(f"Error creating recordings directory: {e}")
            raise e
        
    
    async def process_audio_chunk(self, call_sid: str, audio_data: bytes) -> bytes:
        """
        Process incoming audio chunk and return response audio
        1. Sarvam AI Speech-to-Text
        2. LLM processing
        3. Sarvam AI Text-to-Speech
        """
        try:
            # Create directory for call_sid if it does not exist in recordings directory
            call_recording_dir = self.create_recordings_directory(call_sid)
            
            # Convert μ-law audio to WAV format and save for Sarvam AI
            # Note: This step will be removed once streaming is supported by Sarvam AI
            wav_file_path = self.convert_ulaw_to_wav(call_recording_dir, audio_data)                        
            
            # Convert to text
            transcription = await self.sarvam_client.speech_to_text(wav_file_path)
            
            self.add_message(call_sid, "user", transcription)

            # Get LLM response
            llm_response = await self.sarvam_client.get_llm_response(self.active_calls[call_sid]["messages"])
            
            self.add_message(call_sid, "assistant", llm_response)
                                
            # Convert to speech
            response_audio = await self.sarvam_client.text_to_speech(llm_response)

            # Convert response audio to μ-law format for Twilio
            ulaw_audio = self.convert_wav_to_ulaw(response_audio)                        

            return ulaw_audio
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return b'\xff' * len(audio_data)