import threading
import time
import queue
import numpy as np
import torch
import whisper
import logging
from app.constants import SAMPLE_RATE, TRANSCRIPTION_CHUNK_DURATION, SOURCE_LANGUAGE_CODES

logger = logging.getLogger("TranscriptionApp")

class Transcriber:
    def __init__(self, model_name, gui_queue):
        self.model_name = model_name
        self.gui_queue = gui_queue
        self.audio_buffer = np.zeros((0, 1), dtype=np.float32)
        self.transcribing = False
        self.audio_queue = queue.Queue()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Whisper model '{model_name}' on {device}.")
        self.model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper model '{model_name}' loaded successfully on {device}.")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        if self.transcribing:
            self.audio_queue.put(indata.copy())

    def start(self):
        self.transcribing = True
        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()
        logger.info("Transcription thread started.")

    def stop(self):
        self.transcribing = False

    def transcribe_audio(self):
        while self.transcribing:
            try:
                data = self.audio_queue.get(timeout=1)
                self.audio_buffer = np.concatenate((self.audio_buffer, data), axis=0)
                if len(self.audio_buffer) >= SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:
                    audio_chunk = self.audio_buffer[:SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION]
                    self.audio_buffer = self.audio_buffer[SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:]
                    audio_flat = audio_chunk.flatten()
                    self.gui_queue.put({'transcription': "Transcribing...\n"})
                    transcription_start_time = time.time()
                    # Adjust source language as needed
                    result = self.model.transcribe(audio_flat)
                    transcription_end_time = time.time()
                    transcription_duration = transcription_end_time - transcription_start_time
                    transcription = result.get('text', '').strip()
                    if transcription:
                        self.gui_queue.put({
                            'transcription': f"You said: {transcription}\n",
                            'transcription_time': f"Transcription Time: {transcription_duration:.2f} seconds",
                            'transcription_update': True
                        })
                        logger.info(f"Transcription completed in {transcription_duration:.2f} seconds")
            except queue.Empty:
                continue
            except Exception as e:
                self.gui_queue.put({'error': f"Error during transcription process: {e}\n"})
                logger.error(f"Error during transcription process: {e}", exc_info=True)