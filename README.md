# Real Time Voice Agent

This project implements a real-time voice agent using Twilio and SarvamAI, enabling telephony conversations with features like speech-to-text, text-to-speech, and large language model integration.

## Features

-   Real-time audio processing with Twilio's Bi-directional media stream.
-   Speech-to-Text (STT) for converting spoken language into text.
-   Large Language Model (LLM) for understanding and generating responses.
-   Text-to-Speech (TTS) for converting text responses back into spoken language.
-   Voice Activity Detection (VAD) to dynamically buffer audio data until the user stops talking.
-   Memory for storing entire conversations and responding accordingly.
-   Default greeting from the voice agent for a more natural call experience.
-   Configurable through a `.env` file for easy model changes.

## Requirements

-   Python 3.8 or higher
-   Twilio account with access to the Programmable Voice API
-   SarvamAI account with access to the required models
-   Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone
    ```
2. Navigate to the project directory:
    ```bash
    cd real-time-voice-agent
    ```
3. Install the required packages:
    ```bash
     pip install -r requirements.txt
    ```
4. Create a `.env` file in the project root using `.env.example` and configure your Twilio and SarvamAI credentials:
    ```bash
    cp .env.example .env
    ```
5. Update the `.env` file with your Twilio and SarvamAI credentials, and any other configurations as needed.
6. Run the application:
    ```bash
    python src/app.py
    ```
7. Run ngrok to expose your local server to the internet (for Twilio to reach your local server):
    ```bash
    ngrok http 8000
    ```
8. Configure your Twilio phone number to use the ngrok URL for incoming calls, pointing to the `/voice` endpoint of your application.

## Usage

-   Make a call to the Twilio number configured in your account.
-   The voice agent will greet you and start processing your speech in real-time.
-   Speak naturally, and the agent will respond based on the conversation context.
