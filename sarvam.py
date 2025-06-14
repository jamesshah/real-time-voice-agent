import asyncio
import base64
import io
import logging
from typing import Optional
import wave
import audioop
from sarvamai import SarvamAI
import os
import random

logger = logging.getLogger(__name__)

class SarvamClient:
    """
    Sarvam AI client for speech-to-text and text-to-speech operations
    using the official sarvam Python package
    """
    
    def __init__(self, api_key: str):
        self.client = SarvamAI(api_subscription_key=api_key)
    
    async def get_llm_response(self, message: str, model: str = "sarvam-m") -> Optional[str]:
        """
        Get response from Sarvam AI LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for LLM response
            
        Returns:
            Response text or None if failed
        """
        try:            
            messages = [
                {"role": "system", "content": "You are a sales agent for a real estate company called '23 degree north' located in Ahmedabad city of India. You will receive housing inquiry related messages from the users. Your job is to answer them patiently and with respect. Take their preferences and then imagine that you have data regarding real estate listings in Ahmedabad and answer accordingly."},
                {"role": "user", "content": message}
            ]
            
            response = self.client.chat.completions(
                messages=messages
            )
            
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            
            logger.error("No valid response from LLM")
            return None
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return None
    
    async def speech_to_text(
        self, 
        wav_file_path: str, 
        language_code: str = "gu-IN",
        model: str = "saarika:v2"
    ) -> Optional[str]:
        """
        Convert speech to text using Sarvam AI ASR
        
        Args:
            audio_data: Raw audio bytes (will be converted to base64)
            language_code: Language code (hi-IN, en-IN, etc.)
            model: ASR model to use
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Convert audio to base64
            # audio_base64 =  base64.b64encode(audio_data).decode('utf-8')
            
            # Make API request using official client
                        
            response = self.client.speech_to_text.transcribe(
                file=open(wav_file_path, 'rb'),
                language_code=language_code,
                model=model
            )
            
            transcript = response.transcript
            logger.info(f"STT Success: {transcript}")
            
            return transcript
                    
        except Exception as e:
            logger.error(f"STT request failed: {e}")
            return None    
    
    async def speech_to_text_from_file(
        self, 
        file_path: str, 
        language_code: str = "gu-IN",
        model: str = "saarika:v2"
    ) -> Optional[str]:
        """
        Convert speech to text from audio file
        
        Args:
            file_path: Path to audio file
            language_code: Language code
            model: ASR model to use
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            with open(file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            return await self.speech_to_text(audio_data, language_code, model)
            
        except Exception as e:
            logger.error(f"Error reading audio file {file_path}: {e}")
            return None
    
    def convert_ulaw_to_wav(self, ulaw_data: bytes, sample_rate: int = 8000) -> str:
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
            with open(random_name, 'wb') as f:
                f.write(wav_buffer.getvalue())
                
            return random_name
            
        except Exception as e:
            logger.error(f"Error converting μ-law to WAV: {e}")
            return b""
    
    async def speech_to_text_from_ulaw(
        self, 
        ulaw_data: bytes, 
        language_code: str = "gu-IN",
        model: str = "saarika:v2",
        sample_rate: int = 8000
    ) -> Optional[str]:
        """
        Convert μ-law audio (from Twilio) to text
        
        Args:
            ulaw_data: μ-law encoded audio bytes from Twilio
            language_code: Language code
            model: ASR model to use
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Convert μ-law to WAV format
            wav_file_path = self.convert_ulaw_to_wav(ulaw_data, sample_rate)
            
            if not wav_file_path:
                logger.error("Failed to convert μ-law to WAV")
                return None
            
            # Send WAV data to STT
            return await self.speech_to_text(wav_file_path, language_code, model)
            
        except Exception as e:
            logger.error(f"Error processing μ-law audio: {e}")
            return None
    
    async def text_to_speech(
        self, 
        text: str, 
        target_language_code: str = "gu-IN",
        speaker: str = "meera",
        model: str = "bulbul:v2"
    ) -> Optional[bytes]:
        """
        Convert text to speech using Sarvam AI TTS
        
        Args:
            text: Text to convert to speech
            target_language_code: Target language code
            speaker: Speaker voice to use
            model: TTS model to use
            
        Returns:
            Audio bytes or None if failed
        """
        try:
            # Make API request using official client
            response = self.client.text_to_speech.convert(
                text=text,
                target_language_code=target_language_code,
                speaker=speaker,
                pace=1.1,
                model=model
            )
            
            # Get the base64 encoded audio from response
            if hasattr(response, 'audios') and response.audios:
                audio_base64 = response.audios[0]
                audio_data = base64.b64decode(audio_base64)
                
                logger.info(f"TTS Success: Generated {len(audio_data)} bytes")
                return audio_data
            else:
                logger.error("No audio data in TTS response")
                return None
                    
        except Exception as e:
            logger.error(f"TTS request failed: {e}")
            return None
    
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
            logger.error(f"Error converting WAV to μ-law: {e}")
            return b""
    
    async def text_to_speech_for_twilio(
        self, 
        text: str, 
        target_language_code: str = "hi-IN",
        speaker: str = "meera",
        model: str = "bulbul:v1"
    ) -> Optional[bytes]:
        """
        Convert text to speech and format for Twilio (μ-law)
        
        Args:
            text: Text to convert to speech
            target_language_code: Target language code
            speaker: Speaker voice
            model: TTS model to use
            
        Returns:
            μ-law encoded audio bytes for Twilio or None if failed
        """
        try:
            # Get TTS audio (likely in WAV format)
            wav_data = await self.text_to_speech(text, target_language_code, speaker, model)
            
            if not wav_data:
                return None
            
            # Convert to μ-law format for Twilio
            ulaw_data = self.convert_wav_to_ulaw(wav_data)
            
            return ulaw_data
            
        except Exception as e:
            logger.error(f"Error generating TTS for Twilio: {e}")
            return None

    async def translate_text(
        self,
        text: str,
        source_language_code: str = "hi-IN",
        target_language_code: str = "en-IN",
        model: str = "mayura:v1"
    ) -> Optional[str]:
        """
        Translate text using Sarvam AI translation
        
        Args:
            text: Text to translate
            source_language_code: Source language code
            target_language_code: Target language code
            model: Translation model to use
            
        Returns:
            Translated text or None if failed
        """
        try:
            response = self.client.translate(
                input=text,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                model=model
            )
            
            translated_text = response.translated_text
            logger.info(f"Translation Success: {text} -> {translated_text}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation request failed: {e}")
            return None

# Example usage
async def main():
    """Example usage of SarvamAI client"""
    
    # Initialize client
    client = SarvamAI(api_key=os.environ.get("SARVAM_API_KEY"))
    
    # Test STT with file
    transcript = await client.speech_to_text_from_file(
        "test_audio.wav", 
        language_code="hi-IN"
    )
    print(f"Transcript: {transcript}")
    
    # Test TTS
    if transcript:
        audio_data = await client.text_to_speech_for_twilio(
            f"You said: {transcript}",
            target_language_code="hi-IN"
        )
        
        if audio_data:
            print(f"Generated audio: {len(audio_data)} bytes")
    
    # Test translation
    if transcript:
        translated = await client.translate_text(
            transcript,
            source_language_code="hi-IN",
            target_language_code="en-IN"
        )
        print(f"Translation: {translated}")

if __name__ == "__main__":
    asyncio.run(main())