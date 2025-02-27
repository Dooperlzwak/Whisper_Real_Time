import customtkinter as ctk
import threading
import numpy as np
import sounddevice as sd
import pyttsx3
import queue
import time
import os
from tkinter import messagebox, colorchooser
import logging

from app.logger import setup_logging
from app.devices import list_microphones, get_device_id
from app.preferences import load_preferences, save_preferences, apply_preferences
from app.tts import speak_text, speak_text_immediate
from app.transcriber import Transcriber
from app.constants import (AVAILABLE_MODELS, AVAILABLE_TRANSLATIONS,
                           AVAILABLE_SOURCE_LANGUAGES, TRANSLATION_LANGUAGE_CODES, SOURCE_LANGUAGE_CODES, SAMPLE_RATE, BLOCK_SIZE)

logger = setup_logging()

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription with Translation")
        self.root.geometry("1400x1200")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.current_model_name = ctk.StringVar(value="base")
        self.microphone_var = ctk.StringVar()
        self.translation_var = ctk.StringVar(value="None")
        self.source_language_var = ctk.StringVar(value="Auto")
        self.output_device_var = ctk.StringVar()
        self.transcribing = False
        self.gui_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.tts_engine = pyttsx3.init()
        self.preferences = load_preferences()
        apply_preferences(self.preferences)
        self.create_widgets()
        self.populate_microphones()
        self.populate_output_speakers()
        self.transcriber = Transcriber(self.current_model_name.get(), self.gui_queue)
        self.process_queue()
        # Start TTS thread
        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()

    def create_widgets(self):
        # Create your header, selection menus, text areas, and control buttons.
        # (For brevity, include your widget code here similar to your original program.)
        pass

    def populate_microphones(self):
        input_devices, _ = list_microphones()
        if not input_devices:
            messagebox.showerror("Error", "No input devices found.")
            self.root.destroy()
            return
        self.microphone_names = [f"{device['name']} (ID: {device['index']})" for device in input_devices]
        # Set up your microphone dropdown widget here...
        logger.info(f"Detected Microphones: {self.microphone_names}")

    def populate_output_speakers(self):
        _, output_devices = list_microphones()
        if not output_devices:
            messagebox.showerror("Error", "No output devices found.")
            self.root.destroy()
            return
        self.output_device_names = [f"{device['name']} (ID: {device['index']})" for device in output_devices]
        # Set up your speaker dropdown widget here...
        logger.info(f"Detected Output Devices: {self.output_device_names}")

    def start_transcription(self):
        selected_mic_name = self.microphone_var.get()
        device_id = get_device_id(selected_mic_name, input_device=True)
        if device_id is None:
            messagebox.showerror("Device Error", f"Selected device '{selected_mic_name}' not found.")
            return
        try:
            self.stream = sd.InputStream(
                callback=self.transcriber.audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=device_id,
                dtype='float32'
            )
            self.stream.start()
            self.transcribing = True
            self.transcriber.start()
        except Exception as e:
            messagebox.showerror("Audio Stream Error", f"Failed to start audio stream.\nError: {e}")

    def stop_transcription(self):
        self.transcribing = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.transcriber.stop()

    def process_queue(self):
        try:
            while True:
                message = self.gui_queue.get_nowait()
                # Update your transcription text area and labels based on message content.
                # Also, if a translated text is received, add it to the TTS queue.
                if 'translated_text' in message:
                    self.tts_queue.put(message['translated_text'])
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def process_tts_queue(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            speak_text(text, self.translation_var.get())
            self.tts_queue.task_done()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.transcribing:
                self.stop_transcription()
            self.tts_queue.put(None)
            self.tts_thread.join()
            self.tts_engine.stop()
            self.root.destroy()