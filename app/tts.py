import tempfile
import os
import logging
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
from app.constants import TRANSLATION_LANGUAGE_CODES

logger = logging.getLogger("TranscriptionApp")

def speak_text(text, target_language):
    try:
        lang_code = TRANSLATION_LANGUAGE_CODES.get(target_language, 'en')
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            temp_filename = tf.name
        tts.save(temp_filename)
        audio = AudioSegment.from_mp3(temp_filename)
        play(audio)
        os.remove(temp_filename)
        logger.info(f"Played TTS audio using gTTS for {target_language}.")
    except Exception as e:
        logger.error(f"Error during Text-to-Speech: {e}", exc_info=True)

def speak_text_immediate(tts_engine, text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        logger.error(f"Error during immediate TTS: {e}", exc_info=True)