# Voice HR Agent

A voice-based HR agent built with LangGraph that conducts interviews for Data Engineering positions.

## Features

- Voice-based conversation using ElevenLabs TTS
- Speech-to-text transcription using Groq's Whisper
- Structured data extraction for user details and tech stack
- Interactive interview flow with multiple stages
- Streamlit dashboard to view collected data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the voice directory with:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

3. Get API Keys:
- ElevenLabs: Sign up at https://elevenlabs.io/ and get your API key
- Groq: Sign up at https://groq.com/ and get your API key

## Usage

### Run the Voice Agent
```bash
cd voice
python audio.py
```

### View Results Dashboard
```bash
cd voice
streamlit run streamlit_app.py
```

## File Structure

- `audio.py` - Main voice agent application
- `streamlit_app.py` - Dashboard to view collected data
- `tech_required.txt` - Required technologies for the position
- `user_tech_stack.json` - Generated file with user responses
- `requirements.txt` - Python dependencies

## Interview Flow

1. **Greeting**: Agent introduces itself and asks about job interest
2. **Personal Details**: Collects name, email, and phone number
3. **Tech Stack Assessment**: Asks about experience with required technologies
4. **Goodbye**: Thanks the user and ends the interview

## Requirements

- Python 3.8+
- Microphone access
- Speakers/headphones
- Internet connection for API calls 