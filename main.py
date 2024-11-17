import customtkinter as ctk  
import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
from googletrans import Translator
import time
import pyttsx3
import soundfile as sf
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import logging
import datetime
import torch
from tkinter import messagebox  
import json  

def setup_logging():
    """
    Configures the logging for the application.
    Creates a new log file each time the app starts, named with the current date and time.
    """
    logger = logging.getLogger("TranscriptionApp")
    logger.setLevel(logging.INFO)  

    log_directory = "./logs"
    os.makedirs(log_directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}.log"  
    log_path = os.path.join(log_directory, log_filename)

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Logging is set up successfully. Logs are being saved to {log_path}")
    return logger

logger = setup_logging()

SAMPLE_RATE = 16000  
BLOCK_SIZE = 1024
CHANNELS = 1  
TRANSCRIPTION_CHUNK_DURATION = 5  

audio_queue = queue.Queue()

AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

AVAILABLE_TRANSLATIONS = ['None', 'German', 'Mandarin', 'Spanish', 'French', 'Japanese']

TRANSLATION_LANGUAGE_CODES = {
    'None': None,
    'German': 'de',
    'Mandarin': 'zh-cn',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja'
}

AVAILABLE_SOURCE_LANGUAGES = [
    'Auto', 'English', 'German', 'Mandarin', 'Spanish',
    'French', 'Japanese', 'Korean', 'Italian', 'Russian',
    'Portuguese', 'Dutch', 'Arabic', 'Hindi', 'Bengali'
]

SOURCE_LANGUAGE_CODES = {
    'Auto': None,
    'English': 'en',
    'German': 'de',
    'Mandarin': 'zh',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Italian': 'it',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn'
}

def list_microphones():
    """
    Lists available input and output audio devices.
    Returns two lists: input_devices and output_devices.
    """
    try:
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        output_devices = [device for device in devices if device['max_output_channels'] > 0]
        logger.info(f"Total Input Devices Found: {len(input_devices)}")
        logger.info(f"Total Output Devices Found: {len(output_devices)}")
        return input_devices, output_devices
    except Exception as e:
        logger.error(f"Error querying devices: {e}", exc_info=True)
        return [], []

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription with Translation")
        self.root.geometry("1400x1200")  

        ctk.set_appearance_mode("System")  
        ctk.set_default_color_theme("blue")  

        self.model = None
        self.current_model_name = ctk.StringVar(value="base")  
        self.microphone_var = ctk.StringVar()
        self.translation_var = ctk.StringVar(value="None")  
        self.source_language_var = ctk.StringVar(value="Auto")  
        self.output_device_var = ctk.StringVar()  
        self.transcribing = False
        self.audio_buffer = np.zeros((0, CHANNELS), dtype=np.float32)
        self.stream = None
        self.transcription_thread = None
        self.translator = Translator()

        self.gui_queue = queue.Queue()

        try:
            self.tts_engine = pyttsx3.init()
            logger.info("pyttsx3 TTS engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 TTS engine: {e}", exc_info=True)
            messagebox.showerror("TTS Initialization Error", f"Failed to initialize TTS engine.\nError: {e}")
            self.root.destroy()

        self.preferences = self.load_preferences()

        self.apply_preferences()

        self.create_widgets()

        self.populate_microphones()
        self.populate_output_speakers()

        self.load_model(self.current_model_name.get())

        self.process_queue()

    def create_widgets(self):
        """
        Creates and places all GUI widgets using CustomTkinter.
        """

        header_frame = ctk.CTkFrame(self.root)
        header_frame.pack(padx=10, pady=10, fill="x")

        title_label = ctk.CTkLabel(header_frame, text="Live Speech Transcription", font=("Roboto Medium", 20))
        title_label.pack(side=ctk.LEFT, padx=5, pady=5)

        header_spacer = ctk.CTkLabel(header_frame, text="", width=1000)  
        header_spacer.pack(side=ctk.LEFT, padx=5, pady=5)

        settings_button = ctk.CTkButton(header_frame, text="⚙️", command=self.open_settings, width=40, height=40)
        settings_button.pack(side=ctk.RIGHT, padx=5, pady=5)

        model_frame = ctk.CTkFrame(self.root)
        model_frame.pack(padx=10, pady=10, fill="x")

        model_label = ctk.CTkLabel(model_frame, text="Select Whisper Model:", font=("Roboto Medium", 16))
        model_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            variable=self.current_model_name,
            values=AVAILABLE_MODELS,
            command=self.on_model_change,
            width=200
        )
        self.model_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)

        device_frame = ctk.CTkFrame(self.root)
        device_frame.pack(padx=10, pady=10, fill="x")

        mic_frame = ctk.CTkFrame(device_frame)
        mic_frame.pack(side=ctk.LEFT, padx=5, pady=5, expand=True, fill="x")

        mic_label = ctk.CTkLabel(mic_frame, text="Select Microphone:", font=("Roboto Medium", 16))
        mic_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.microphone_dropdown = ctk.CTkOptionMenu(
            mic_frame,
            variable=self.microphone_var,
            values=[],  
            width=400
        )
        self.microphone_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)

        speaker_frame = ctk.CTkFrame(device_frame)
        speaker_frame.pack(side=ctk.LEFT, padx=5, pady=5, expand=True, fill="x")

        speaker_label = ctk.CTkLabel(speaker_frame, text="Select Speaker Output:", font=("Roboto Medium", 16))
        speaker_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.speaker_dropdown = ctk.CTkOptionMenu(
            speaker_frame,
            variable=self.output_device_var,
            values=[],  
            width=400
        )
        self.speaker_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)

        language_frame = ctk.CTkFrame(self.root)
        language_frame.pack(padx=10, pady=10, fill="x")

        source_lang_frame = ctk.CTkFrame(language_frame)
        source_lang_frame.pack(side=ctk.LEFT, padx=5, pady=5, expand=True, fill="x")

        source_lang_label = ctk.CTkLabel(source_lang_frame, text="Select Source Language:", font=("Roboto Medium", 16))
        source_lang_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.source_language_dropdown = ctk.CTkOptionMenu(
            source_lang_frame,
            variable=self.source_language_var,
            values=AVAILABLE_SOURCE_LANGUAGES,
            width=200
        )
        self.source_language_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)

        translation_frame = ctk.CTkFrame(language_frame)
        translation_frame.pack(side=ctk.LEFT, padx=5, pady=5, expand=True, fill="x")

        translation_label = ctk.CTkLabel(translation_frame, text="Select Target Language:", font=("Roboto Medium", 16))
        translation_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.translation_dropdown = ctk.CTkOptionMenu(
            translation_frame,
            variable=self.translation_var,
            values=AVAILABLE_TRANSLATIONS,
            width=200
        )
        self.translation_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)

        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(padx=10, pady=10, fill="x")

        self.start_button = ctk.CTkButton(control_frame, text="Start Transcription", command=self.start_transcription, width=150)
        self.start_button.pack(side=ctk.LEFT, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(control_frame, text="Stop Transcription", command=self.stop_transcription, state="disabled", width=150)
        self.stop_button.pack(side=ctk.LEFT, padx=5, pady=5)

        self.save_button = ctk.CTkButton(control_frame, text="Save Transcription", command=self.save_transcription, width=150)
        self.save_button.pack(side=ctk.LEFT, padx=5, pady=5)

        status_frame = ctk.CTkFrame(self.root)
        status_frame.pack(padx=10, pady=10, fill="x")

        self.transcription_time_label = ctk.CTkLabel(status_frame, text="Transcription Time: N/A", font=("Roboto Medium", 14))
        self.transcription_time_label.pack(side=ctk.LEFT, padx=10, pady=5)

        self.translation_time_label = ctk.CTkLabel(status_frame, text="Translation Time: N/A", font=("Roboto Medium", 14))
        self.translation_time_label.pack(side=ctk.LEFT, padx=10, pady=5)

        transcription_label = ctk.CTkLabel(self.root, text="Transcription:", font=("Roboto Medium", 16))
        transcription_label.pack(padx=10, pady=(10, 0), anchor="w")

        self.transcription_area = ctk.CTkTextbox(self.root, wrap='word', width=1400, height=600)
        self.transcription_area.pack(padx=10, pady=5, fill="both", expand=True)
        self.transcription_area.configure(state='disabled')

    def open_settings(self):
        """
        Opens the settings window where users can adjust appearance settings.
        """
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x400")
        settings_window.grab_set()  

        mode_frame = ctk.CTkFrame(settings_window)
        mode_frame.pack(padx=10, pady=10, fill="x")

        mode_label = ctk.CTkLabel(mode_frame, text="Appearance Mode:", font=("Roboto Medium", 16))
        mode_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.mode_switch = ctk.CTkSwitch(mode_frame, text="", command=self.toggle_appearance_mode)
        self.mode_switch.pack(side=ctk.LEFT, padx=5, pady=5)

        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            self.mode_switch.select()
        else:
            self.mode_switch.deselect()

        color_frame = ctk.CTkFrame(settings_window)
        color_frame.pack(padx=10, pady=10, fill="x")

        color_label = ctk.CTkLabel(color_frame, text="Primary Color:", font=("Roboto Medium", 16))
        color_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.color_button = ctk.CTkButton(color_frame, text="Choose Color", command=self.pick_primary_color, width=120)
        self.color_button.pack(side=ctk.LEFT, padx=5, pady=5)

        preset_frame = ctk.CTkFrame(settings_window)
        preset_frame.pack(padx=10, pady=10, fill="x")

        preset_label = ctk.CTkLabel(preset_frame, text="Preset Color Schemes:", font=("Roboto Medium", 16))
        preset_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.preset_colors = {
            "Blue": "blue",
            "Green": "green",
            "Dark-Blue": "dark-blue",
            "Purple": "purple",
            "Dark": "dark"
        }

        self.preset_dropdown = ctk.CTkOptionMenu(
            preset_frame,
            values=list(self.preset_colors.keys()),
            command=self.apply_preset_color_scheme,
            width=150
        )
        self.preset_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)
        self.preset_dropdown.set("Select Scheme")

        save_button = ctk.CTkButton(settings_window, text="Save Preferences", command=self.save_preferences, width=150)
        save_button.pack(padx=10, pady=20)

    def toggle_appearance_mode(self):
        """
        Toggles between light and dark mode.
        """
        current_mode = ctk.get_appearance_mode()
        new_mode = "Dark" if current_mode == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
        logger.info(f"Appearance mode changed to {new_mode}")
        self.preferences['appearance_mode'] = new_mode

    def pick_primary_color(self):
        """
        Opens a color chooser dialog to select the primary color.
        """
        color_code = ctk.CTkColorPicker(title="Pick Primary Color").get_color()
        if color_code:

            ctk.set_default_color_theme("blue")  

            logger.info(f"Selected Primary Color: {color_code}")
            self.preferences['primary_color'] = color_code

    def apply_preset_color_scheme(self, scheme_name):
        """
        Applies a preset color scheme based on the user's selection.
        """
        scheme = self.preset_colors.get(scheme_name)
        if scheme:
            ctk.set_default_color_theme(scheme)
            logger.info(f"Applied preset color scheme: {scheme_name}")
            self.preferences['color_scheme'] = scheme_name

    def save_preferences(self):
        """
        Saves user preferences to a JSON file.
        """
        prefs = {
            'appearance_mode': ctk.get_appearance_mode(),
            'primary_color': self.preferences.get('primary_color', 'blue'),
            'color_scheme': self.preferences.get('color_scheme', 'blue')
        }
        try:
            with open("preferences.json", "w") as f:
                json.dump(prefs, f)
            messagebox.showinfo("Settings", "Preferences saved successfully!")
            logger.info("User preferences saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preferences.\nError: {e}")
            logger.error(f"Failed to save preferences: {e}", exc_info=True)

    def load_preferences(self):
        """
        Loads user preferences from a JSON file.
        """
        if os.path.exists("preferences.json"):
            try:
                with open("preferences.json", "r") as f:
                    prefs = json.load(f)
                logger.info("User preferences loaded.")
                return prefs
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}", exc_info=True)
                return {}
        else:
            logger.info("No preferences file found. Using default settings.")
            return {}

    def apply_preferences(self):
        """
        Applies the loaded user preferences to the application.
        """

        mode = self.preferences.get('appearance_mode')
        if mode:
            ctk.set_appearance_mode(mode)
            logger.info(f"Applied appearance mode: {mode}")

        scheme = self.preferences.get('color_scheme')
        if scheme and scheme in ['blue', 'green', 'dark-blue', 'purple', 'dark']:
            ctk.set_default_color_theme(scheme)
            logger.info(f"Applied color scheme: {scheme}")

        primary_color = self.preferences.get('primary_color')
        if primary_color:
            logger.info(f"User selected primary color: {primary_color}")

    def populate_microphones(self):
        """
        Populates the microphone dropdown with available input devices.
        """
        input_devices, _ = list_microphones()
        if not input_devices:
            error_msg = "No input devices found."
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.root.destroy()
            return
        self.microphone_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(input_devices)]
        self.microphone_dropdown.configure(values=self.microphone_names)
        self.microphone_dropdown.set(self.microphone_names[0])  
        logger.info(f"Detected Microphones: {self.microphone_names}")

    def populate_output_speakers(self):
        """
        Populates the speaker dropdown with available output devices.
        """
        _, output_devices = list_microphones()
        if not output_devices:
            error_msg = "No output devices found."
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.root.destroy()
            return
        self.output_device_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(output_devices)]
        self.speaker_dropdown.configure(values=self.output_device_names)
        self.speaker_dropdown.set(self.output_device_names[0])  
        logger.info(f"Detected Output Devices: {self.output_device_names}")

    def load_model(self, model_name):
        """
        Loads the specified Whisper model with error handling.
        """
        try:
            self.append_transcription(f"Loading Whisper model '{model_name}'...\n")
            logger.info(f"Loading Whisper model '{model_name}'.")
            self.model = whisper.load_model(model_name, device='cpu')  
            self.append_transcription(f"Model '{model_name}' loaded successfully.\n")
            logger.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
            messagebox.showerror("Model Loading Error", f"Failed to load model '{model_name}'.\nError: {e}")
            self.append_transcription(f"Error loading model '{model_name}': {e}\n")

    def on_model_change(self, selected_model):
        """
        Handler for when the Whisper model selection changes.
        """
        if self.transcribing:
            messagebox.showwarning("Transcription Active", "Please stop transcription before changing the model.")

            self.model_dropdown.set(self.model.name if self.model else "base")
            return
        self.load_model(selected_model)

    def start_transcription(self):
        """
        Starts the transcription process with error handling.
        """
        logger.info("Start Transcription button clicked.")  
        selected_input_index = self.microphone_dropdown.current()
        if selected_input_index == -1:
            messagebox.showwarning("Warning", "Please select a microphone.")
            logger.warning("Start Transcription: No microphone selected.")
            return
        selected_output_index = self.speaker_dropdown.current()
        if selected_output_index == -1:
            messagebox.showwarning("Warning", "Please select a speaker output.")
            logger.warning("Start Transcription: No speaker output selected.")
            return
        input_devices, output_devices = list_microphones()
        if selected_input_index >= len(input_devices) or selected_output_index >= len(output_devices):
            messagebox.showerror("Selection Error", "Selected device index out of range.")
            logger.error("Start Transcription: Selected device index out of range.")
            return
        device_id = input_devices[selected_input_index]['index']
        output_device_id = output_devices[selected_output_index]['index']

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.model_dropdown.configure(state="disabled")
        self.microphone_dropdown.configure(state="disabled")
        self.speaker_dropdown.configure(state="disabled")
        self.source_language_dropdown.configure(state="disabled")
        self.translation_dropdown.configure(state="disabled")

        self.transcription_time_label.configure(text="Transcription Time: N/A")
        self.translation_time_label.configure(text="Translation Time: N/A")

        self.transcribing = True
        logger.info("Transcription started.")

        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()

        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=device_id,
                dtype='float32'
            )
            self.stream.start()
            self.append_transcription("Transcription started...\n")
            logger.info("Audio stream started successfully.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}", exc_info=True)
            messagebox.showerror("Audio Stream Error", f"Failed to start audio stream.\nError: {e}")
            self.append_transcription(f"Error starting audio stream: {e}\n")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.model_dropdown.configure(state="readonly")
            self.microphone_dropdown.configure(state="readonly")
            self.speaker_dropdown.configure(state="readonly")
            self.source_language_dropdown.configure(state="readonly")
            self.translation_dropdown.configure(state="readonly")
            self.transcribing = False

    def stop_transcription(self):
        """
        Stops the transcription process with error handling.
        """
        logger.info("Stop Transcription button clicked.")  
        self.transcribing = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}", exc_info=True)
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.model_dropdown.configure(state="readonly")
        self.microphone_dropdown.configure(state="readonly")
        self.speaker_dropdown.configure(state="readonly")
        self.source_language_dropdown.configure(state="readonly")
        self.translation_dropdown.configure(state="readonly")
        self.append_transcription("\nTranscription stopped.\n")
        logger.info("Transcription stopped.")

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio stream. Adds audio data to the queue.
        """
        if status:
            logger.warning(f"Audio status: {status}")
            print(f"Audio status: {status}")
        if self.transcribing:
            audio_queue.put(indata.copy())

    def transcribe_audio(self):
        """
        Processes audio data from the queue and performs transcription and translation.
        """
        logger.info("Transcription thread started.")  
        while self.transcribing:
            try:
                data = audio_queue.get(timeout=1)
                self.audio_buffer = np.concatenate((self.audio_buffer, data), axis=0)
                if len(self.audio_buffer) >= SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:

                    audio_chunk = self.audio_buffer[:SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION]
                    self.audio_buffer = self.audio_buffer[SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:]

                    audio_flat = audio_chunk.flatten()

                    source_language = self.source_language_var.get()
                    source_lang_code = SOURCE_LANGUAGE_CODES.get(source_language)

                    self.gui_queue.put({'transcription': "Transcribing...\n"})
                    logger.info("Transcribing audio chunk.")

                    transcription_start_time = time.time()

                    try:
                        if source_lang_code:
                            result = self.model.transcribe(audio_flat, language=source_lang_code)
                        else:

                            result = self.model.transcribe(audio_flat)
                    except Exception as e:
                        logger.error(f"Error during transcription: {e}", exc_info=True)
                        self.gui_queue.put({'error': f"Error during transcription: {e}\n"})
                        continue  

                    transcription_end_time = time.time()

                    transcription_duration = transcription_end_time - transcription_start_time
                    transcription_duration_formatted = f"{transcription_duration:.2f} seconds"

                    transcription = result.get('text', '').strip()

                    if transcription:

                        self.gui_queue.put({
                            'transcription': f"You said: {transcription}\n",
                            'transcription_time': f"Transcription Time: {transcription_duration_formatted}",
                            'transcription_update': True
                        })
                        logger.info(f"Transcription completed in {transcription_duration_formatted}")

                        target_language = self.translation_var.get()
                        if target_language != 'None':

                            translation_start_time = time.time()

                            translated_text = self.translate_text(transcription, target_language)

                            translation_end_time = time.time()

                            translation_duration = translation_end_time - translation_start_time
                            translation_duration_formatted = f"{translation_duration:.2f} seconds"

                            if translated_text:

                                self.gui_queue.put({
                                    'translation': f"Translation ({target_language}): {translated_text}\n",
                                    'translation_time': f"Translation Time: {translation_duration_formatted}",
                                    'translation_update': True,
                                    'translated_text': translated_text  
                                })
                                logger.info(f"Translation completed in {translation_duration_formatted}")
            except queue.Empty:
                continue
            except Exception as e:

                error_message = f"Error during transcription process: {e}\n"
                self.gui_queue.put({'error': error_message})
                logger.error(f"Error during transcription process: {e}", exc_info=True)

    def translate_text(self, text, target_language):
        """
        Translates the given text into the target language with error handling.
        """
        try:
            lang_code = TRANSLATION_LANGUAGE_CODES.get(target_language)
            if not lang_code:
                logger.warning(f"No language code found for target language: {target_language}")
                return None
            translation = self.translator.translate(text, dest=lang_code)
            if translation and hasattr(translation, 'text'):
                logger.info(f"Translated text to {target_language}: {translation.text}")
                return translation.text
            else:
                logger.error("Translation failed or returned None.")
                return None
        except Exception as e:
            logger.error(f"Error during translation: {e}", exc_info=True)
            self.gui_queue.put({'error': f"Error during translation: {e}\n"})
            return None

    def append_transcription(self, text, tag='default'):
        """
        Appends text to the transcription area in a thread-safe manner.
        """

        self.transcription_area.configure(state='normal')
        self.transcription_area.insert(ctk.END, text, tag)
        self.transcription_area.see(ctk.END)
        self.transcription_area.configure(state='disabled')

    def process_queue(self):
        """
        Processes messages from the GUI queue and updates the GUI accordingly.
        """
        try:
            while True:
                message = self.gui_queue.get_nowait()
                if 'transcription' in message:
                    self.append_transcription(message['transcription'], 'transcription')
                if 'transcription_time' in message and message.get('transcription_update'):
                    self.transcription_time_label.configure(text=message['transcription_time'])
                if 'translation' in message:
                    self.append_transcription(message['translation'], 'translation')
                if 'translation_time' in message and message.get('translation_update'):
                    self.translation_time_label.configure(text=message['translation_time'])
                if 'translated_text' in message:

                    threading.Thread(target=self.speak_text, args=(message['translated_text'],), daemon=True).start()
                if 'error' in message:
                    self.append_transcription(message['error'])
        except queue.Empty:
            pass
        finally:

            self.root.after(100, self.process_queue)

    def speak_text(self, text):
        """
        Converts text to speech with error handling.
        """
        try:

            target_language = self.translation_var.get()
            if target_language == 'None':

                voice_id = self.get_default_voice()
                if voice_id:
                    self.tts_engine.setProperty('voice', voice_id)
                else:
                    logger.warning("No default voice found. Skipping TTS.")
                    return

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                    temp_filename = tf.name
                self.tts_engine.save_to_file(text, temp_filename)
                self.tts_engine.runAndWait()

                selected_output_index = self.speaker_dropdown.current()
                _, output_devices = list_microphones()
                if selected_output_index >= len(output_devices):
                    logger.error("Selected output device index out of range.")
                    self.gui_queue.put({'error': "Selected output device index out of range.\n"})
                    return
                output_device_id = output_devices[selected_output_index]['index']

                data, fs = sf.read(temp_filename, dtype='float32')

                sd.play(data, fs, device=output_device_id)
                sd.wait()  

                os.remove(temp_filename)
                logger.info("Played TTS audio using pyttsx3.")
            elif target_language == 'Mandarin':

                tts = gTTS(text=text, lang='zh-cn')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
                    temp_filename = tf.name
                tts.save(temp_filename)

                playsound(temp_filename)

                os.remove(temp_filename)
                logger.info("Played TTS audio using gTTS for Mandarin.")
            else:

                logger.warning(f"TTS for language '{target_language}' is not specifically handled. Using default voice.")
                voice_id = self.get_default_voice()
                if voice_id:
                    self.tts_engine.setProperty('voice', voice_id)
                else:
                    logger.warning("No default voice found. Skipping TTS.")
                    return

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                    temp_filename = tf.name
                self.tts_engine.save_to_file(text, temp_filename)
                self.tts_engine.runAndWait()

                selected_output_index = self.speaker_dropdown.current()
                _, output_devices = list_microphones()
                if selected_output_index >= len(output_devices):
                    logger.error("Selected output device index out of range.")
                    self.gui_queue.put({'error': "Selected output device index out of range.\n"})
                    return
                output_device_id = output_devices[selected_output_index]['index']

                data, fs = sf.read(temp_filename, dtype='float32')

                sd.play(data, fs, device=output_device_id)
                sd.wait()  

                os.remove(temp_filename)
                logger.info(f"Played TTS audio using pyttsx3 for {target_language}.")
        except Exception as e:
            error_message = f"Error during Text-to-Speech: {e}\n"
            self.gui_queue.put({'error': error_message})
            logger.error(f"Error during Text-to-Speech: {e}", exc_info=True)

    def get_default_voice(self):
        """
        Returns the default voice ID from pyttsx3.
        """
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                return voices[0].id
            logger.warning("No voices available in pyttsx3.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving voices from pyttsx3: {e}", exc_info=True)
            return None

    def save_transcription(self):
        """
        Saves the current transcription to a text file with error handling.
        """
        logger.info("Save Transcription button clicked.")  
        transcription_text = self.transcription_area.get("0.0", ctk.END).strip()
        if transcription_text:
            try:
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(transcription_text)
                messagebox.showinfo("Saved", "Transcription saved to transcription.txt")
                logger.info("Transcription saved to transcription.txt")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save transcription.\nError: {e}")
                logger.error(f"Failed to save transcription. Error: {e}", exc_info=True)
        else:
            messagebox.showwarning("Warning", "No transcription to save.")
            logger.warning("Save Transcription attempted with no content.")

    def open_settings(self):
        """
        Opens the settings window where users can adjust appearance settings.
        """
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x400")
        settings_window.grab_set()  

        mode_frame = ctk.CTkFrame(settings_window)
        mode_frame.pack(padx=10, pady=10, fill="x")

        mode_label = ctk.CTkLabel(mode_frame, text="Appearance Mode:", font=("Roboto Medium", 16))
        mode_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.mode_switch = ctk.CTkSwitch(mode_frame, text="", command=self.toggle_appearance_mode)
        self.mode_switch.pack(side=ctk.LEFT, padx=5, pady=5)

        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            self.mode_switch.select()
        else:
            self.mode_switch.deselect()

        color_frame = ctk.CTkFrame(settings_window)
        color_frame.pack(padx=10, pady=10, fill="x")

        color_label = ctk.CTkLabel(color_frame, text="Primary Color:", font=("Roboto Medium", 16))
        color_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.color_button = ctk.CTkButton(color_frame, text="Choose Color", command=self.pick_primary_color, width=120)
        self.color_button.pack(side=ctk.LEFT, padx=5, pady=5)

        preset_frame = ctk.CTkFrame(settings_window)
        preset_frame.pack(padx=10, pady=10, fill="x")

        preset_label = ctk.CTkLabel(preset_frame, text="Preset Color Schemes:", font=("Roboto Medium", 16))
        preset_label.pack(side=ctk.LEFT, padx=5, pady=5)

        self.preset_colors = {
            "Blue": "blue",
            "Green": "green",
            "Dark-Blue": "dark-blue",
            "Purple": "purple",
            "Dark": "dark"
        }

        self.preset_dropdown = ctk.CTkOptionMenu(
            preset_frame,
            values=list(self.preset_colors.keys()),
            command=self.apply_preset_color_scheme,
            width=150
        )
        self.preset_dropdown.pack(side=ctk.LEFT, padx=5, pady=5)
        self.preset_dropdown.set("Select Scheme")

        save_button = ctk.CTkButton(settings_window, text="Save Preferences", command=self.save_preferences, width=150)
        save_button.pack(padx=10, pady=20)

    def toggle_appearance_mode(self):
        """
        Toggles between light and dark mode.
        """
        current_mode = ctk.get_appearance_mode()
        new_mode = "Dark" if current_mode == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
        logger.info(f"Appearance mode changed to {new_mode}")
        self.preferences['appearance_mode'] = new_mode

    def pick_primary_color(self):
        """
        Opens a color chooser dialog to select the primary color.
        """

        from tkinter import colorchooser
        color_code = colorchooser.askcolor(title="Choose Primary Color")[1]
        if color_code:

            logger.info(f"Selected Primary Color: {color_code}")
            messagebox.showinfo("Primary Color", f"Selected Primary Color: {color_code}")
            self.preferences['primary_color'] = color_code

    def apply_preset_color_scheme(self, scheme_name):
        """
        Applies a preset color scheme based on the user's selection.
        """
        scheme = self.preset_colors.get(scheme_name)
        if scheme:
            ctk.set_default_color_theme(scheme)
            logger.info(f"Applied preset color scheme: {scheme_name}")
            self.preferences['color_scheme'] = scheme_name

    def save_preferences(self):
        """
        Saves user preferences to a JSON file.
        """
        prefs = {
            'appearance_mode': ctk.get_appearance_mode(),
            'primary_color': self.preferences.get('primary_color', 'blue'),
            'color_scheme': self.preferences.get('color_scheme', 'blue')
        }
        try:
            with open("preferences.json", "w") as f:
                json.dump(prefs, f)
            messagebox.showinfo("Settings", "Preferences saved successfully!")
            logger.info("User preferences saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preferences.\nError: {e}")
            logger.error(f"Failed to save preferences: {e}", exc_info=True)

    def load_preferences(self):
        """
        Loads user preferences from a JSON file.
        """
        if os.path.exists("preferences.json"):
            try:
                with open("preferences.json", "r") as f:
                    prefs = json.load(f)
                logger.info("User preferences loaded.")
                return prefs
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}", exc_info=True)
                return {}
        else:
            logger.info("No preferences file found. Using default settings.")
            return {}

    def apply_preferences(self):
        """
        Applies the loaded user preferences to the application.
        """

        mode = self.preferences.get('appearance_mode')
        if mode:
            ctk.set_appearance_mode(mode)
            logger.info(f"Applied appearance mode: {mode}")

        scheme = self.preferences.get('color_scheme')
        if scheme and scheme in ['blue', 'green', 'dark-blue', 'purple', 'dark']:
            ctk.set_default_color_theme(scheme)
            logger.info(f"Applied color scheme: {scheme}")

        primary_color = self.preferences.get('primary_color')
        if primary_color:
            logger.info(f"User selected primary color: {primary_color}")

    def on_closing(self, app, root):
        """
        Handles the application closure with graceful shutdown.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if app.transcribing:
                app.stop_transcription()

            try:
                app.tts_engine.stop()
                logger.info("TTS engine stopped gracefully.")
            except Exception as e:
                logger.error(f"Error stopping TTS engine: {e}", exc_info=True)
            logger.info("Application closed.")
            root.destroy()

def main():
    root = ctk.CTk()  
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: app.on_closing(app, root))
    root.mainloop()

if __name__ == "__main__":
    main()
