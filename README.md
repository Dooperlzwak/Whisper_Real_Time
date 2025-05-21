# Whisper_Real_Time

**Whisper_Real_Time** is a real-time speech transcription and translation desktop application powered by OpenAI's Whisper model. It features a user-friendly GUI for live audio transcription, language translation, and text-to-speech playback.

## Features

- **Live Speech Transcription**: Transcribes microphone input in real time using Whisper.
- **Translation**: Supports automatic translation of transcribed text to multiple languages.
- **Text-to-Speech**: Listen to transcriptions and translations using built-in TTS engines.
- **Customizable Audio Devices**: Select input (microphone) and output (speaker) devices.
- **GUI Interface**: Built with CustomTkinter for a modern, intuitive experience.
- **Logging**: Saves logs of transcriptions and events for debugging or review.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dooperlzwak/Whisper_Real_Time.git
   cd Whisper_Real_Time
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python main.py
```

- Use the GUI to select your microphone and preferred language.
- Start live transcription and translation.
- Listen to spoken feedback via your chosen speakers.

## Dependencies

Main dependencies (see `requirements.txt` for the full list):

- `openai-whisper`
- `customtkinter`
- `sounddevice`
- `numpy`
- `googletrans`
- `pyttsx3`
- `torch`
- `pydub`
- `pywin32`
- `gTTS`
- `playsound`
- `soundfile`

## Example

Once the app is running, speak into your selected microphone. The transcription (and translation, if enabled) will appear in the GUI and can be played back using the TTS feature.

---

*For more details, see the source code and [requirements.txt](requirements.txt).*
