import numpy as np
import torch
from silero_vad import load_silero_vad
import audioop


# --- VAD Configuration ---
VAD_THRESHOLD = 0.5  # Speech probability threshold.
SILENCE_TIMEOUT_MS = 800 # How long of a silence triggers the end of a phrase.
TWILIO_SAMPLE_RATE = 8000 # Twilio sends audio at 8000 Hz.
VAD_SAMPLE_RATE = 16000

# Twilio sends media packets every 20ms.
FRAMES_PER_20MS_TWILIO = int(TWILIO_SAMPLE_RATE * 0.02) # 160 samples at 8kHz
FRAMES_PER_20MS_16K = int(VAD_SAMPLE_RATE * 0.02) # 320 samples at 16kHz
SILENCE_FRAMES_THRESHOLD = int(SILENCE_TIMEOUT_MS / 20) # Number of 20ms silent chunks
VAD_CHUNK_SIZE_SAMPLES = 512

def initialize_vad_model():
    """
    Initialize and return the Silero VAD model.
    """
    return load_silero_vad()

def process_audio_with_vad(model, audio_mulaw, call_data):
    """
    Process audio with VAD and return results.
    """
    # Convert µ-law to 16-bit linear PCM for the VAD
    audio_pcm = audioop.ulaw2lin(audio_mulaw, 2)

    # Resample from 8kHz to 16kHz for Silero VAD, maintaining state
    audio_pcm_16k, call_data["resampler_state"] = audioop.ratecv(
        audio_pcm, 2, 1, TWILIO_SAMPLE_RATE, VAD_SAMPLE_RATE, call_data["resampler_state"]
    )

    audio_np_16k = np.frombuffer(audio_pcm_16k, dtype=np.int16)

    # Accumulate 16kHz PCM for VAD
    call_data["vad_audio_buffer"] = np.concatenate((call_data["vad_audio_buffer"], audio_np_16k))

    end_of_speech = False

    while len(call_data["vad_audio_buffer"]) >= VAD_CHUNK_SIZE_SAMPLES:
        # Take a chunk of exactly VAD_CHUNK_SIZE_SAMPLES for VAD
        vad_chunk_np = call_data["vad_audio_buffer"][:VAD_CHUNK_SIZE_SAMPLES]

        # Convert to a PyTorch tensor
        audio_tensor = torch.from_numpy(vad_chunk_np.astype(np.float32)) / 32768.0

        # Get speech probability from the loaded model
        speech_prob = model(audio_tensor, VAD_SAMPLE_RATE).item()

        # Remove the processed chunk from the buffer
        call_data["vad_audio_buffer"] = call_data["vad_audio_buffer"][VAD_CHUNK_SIZE_SAMPLES:]

        # VAD Logic: Detect start and end of speech
        if speech_prob > VAD_THRESHOLD:
            if not call_data["is_speaking"]:
                call_data["is_speaking"] = True
            call_data["silence_frames"] = 0
            call_data["speech_buffer"] += audio_mulaw
        else:
            if call_data["is_speaking"]:
                call_data["silence_frames"] += 1
                if call_data["silence_frames"] > SILENCE_FRAMES_THRESHOLD:
                    end_of_speech = True
                    break

    return {
        "end_of_speech": end_of_speech,
        "speech_buffer": call_data["speech_buffer"]
    }
