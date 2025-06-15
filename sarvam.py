import asyncio
import base64
import logging
from typing import Optional
from sarvamai import SarvamAI
import os

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
                        
            raise ValueError("No valid response from LLM", response)
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise e
    
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
            # Make API request using official client                        
            response = self.client.speech_to_text.transcribe(
                file=open(wav_file_path, 'rb'),
                language_code=language_code,
                model=model
            )
            
            if hasattr(response, 'transcript'):            
                return response.transcript
            
            raise ValueError("No transcript returned from STT", response)
                    
        except Exception as e:
            logger.error(f"STT request failed: {e}")
            raise e
    
    async def text_to_speech(
        self, 
        text: str, 
        target_language_code: str = "gu-IN",
        speaker: str = "anushka",
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
                                
                return audio_data
            else:
                raise ValueError("No audio data returned from TTS", response)
                    
        except Exception as e:
            logger.error(f"TTS request failed: {e}")
            raise e

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
            response = self.client.text.translate(
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
    client = SarvamClient(api_key=os.environ.get("SARVAM_API_KEY"))
    
    # Test STT with file
    transcript = await client.speech_to_text(
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