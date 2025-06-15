# Real-Time Voice Agent For Inbound Sales Inquires for a Real Estate company in Indian Languages

This contains various versions of real-time voice agent attempts using various different technologies and frameworks

v2: 06-14-2025

-   In this version, we add silero-vad package for Voice Activity Detection
-   Everytime we receive audio data from Twilio, we pass it to VAD model, and it checks if the speech is identified or not, until user stops talking, we buffer the data (unlike a fixed window size buffer in v1)

v1: 06-13-2025

-   This is the first version using Twilio and SarvamAI for telephony conversation
-   In this, we use Twilio's Bi-directional media stream on websocket, to get raw audio and then process that through the pipeline of Speech-To-Text (STT), LLM (Large Language Model), and Text-To-Speech (TTS) to get the audio response and finally streaming it back to twilio.
-   In this version, we use a fixed chunk size to process the audio (i.e., every 10 sec or whatever the window_size is - we collect audio for that duration and process it)
-   To run it locally, follow below steps
-   Create Twilio Account. Get a Phone Number.
-   Get SarvamAI API Key and put it in .env
-   Clone repo and install dependencies using `pip3 install -r requirements.txt`
-   Start server using `python3 app.py`
-   Run ngrok for port 8000 using `ngrok http 8000`
-   Use the `https` url in the Twilio portal for the voice webhook
-   Call on your Twilio phone number and test
