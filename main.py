import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from googletrans import Translator
import time
import pyttsx3
import soundfile as sf
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription_times.log"),
        logging.StreamHandler()
    ]
)

# Audio parameters
SAMPLE_RATE = 16000  # Whisper expects 16000 Hz
BLOCK_SIZE = 1024
CHANNELS = 1  # Mono audio
TRANSCRIPTION_CHUNK_DURATION = 5  # seconds

# Queues for audio data and GUI updates
audio_queue = queue.Queue()

# Available Whisper models
AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Available target languages for translation
AVAILABLE_TRANSLATIONS = ['None', 'German', 'Mandarin', 'Spanish', 'French', 'Japanese']

# Mapping for googletrans language codes
TRANSLATION_LANGUAGE_CODES = {
    'None': None,
    'German': 'de',
    'Mandarin': 'zh-cn',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja'
}

# Available source languages for transcription
AVAILABLE_SOURCE_LANGUAGES = [
    'Auto', 'English', 'German', 'Mandarin', 'Spanish',
    'French', 'Japanese', 'Korean', 'Italian', 'Russian',
    'Portuguese', 'Dutch', 'Arabic', 'Hindi', 'Bengali'
]

# Mapping for Whisper language codes
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

# Function to list available input and output microphones
def list_microphones():
    try:
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        output_devices = [device for device in devices if device['max_output_channels'] > 0]
        logging.info(f"Total Input Devices Found: {len(input_devices)}")
        logging.info(f"Total Output Devices Found: {len(output_devices)}")
        return input_devices, output_devices
    except Exception as e:
        logging.error(f"Error querying devices: {e}")
        return [], []

# GUI Application Class
class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription with Translation")
        self.root.geometry("1400x1200")  # Increased window size for better layout

        # Initialize variables
        self.model = None
        self.current_model_name = tk.StringVar(value="base")  # Default model
        self.microphone_var = tk.StringVar()
        self.translation_var = tk.StringVar(value="None")  # Default no translation
        self.source_language_var = tk.StringVar(value="Auto")  # Default auto-detect
        self.output_device_var = tk.StringVar()  # For speaker output
        self.transcribing = False
        self.audio_buffer = np.zeros((0, CHANNELS), dtype=np.float32)
        self.stream = None
        self.transcription_thread = None
        self.translator = Translator()

        # Initialize GUI queue for thread-safe updates
        self.gui_queue = queue.Queue()

        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        # Optionally, set properties like voice, rate, volume here
        # Example:
        # voices = self.tts_engine.getProperty('voices')
        # self.tts_engine.setProperty('voice', voices[0].id)  # Select a voice
        # self.tts_engine.setProperty('rate', 150)  # Set speech rate

        # Log application start
        logging.info("Application initialized.")

        # Create GUI components first
        self.create_widgets()

        # Then load the default model
        self.load_model(self.current_model_name.get())

        # Start processing the GUI queue
        self.process_queue()

    def create_widgets(self):
        # Frame for Model Selection
        model_frame = ttk.LabelFrame(self.root, text="Model Selection")
        model_frame.pack(padx=10, pady=10, fill="x")

        model_label = ttk.Label(model_frame, text="Select Whisper Model:")
        model_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.current_model_name,
            values=AVAILABLE_MODELS,
            state="readonly",
            width=20
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)

        # Frame for Microphone Selection
        mic_frame = ttk.LabelFrame(self.root, text="Microphone Selection")
        mic_frame.pack(padx=10, pady=10, fill="x")

        mic_label = ttk.Label(mic_frame, text="Select Microphone:")
        mic_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.microphone_dropdown = ttk.Combobox(
            mic_frame,
            textvariable=self.microphone_var,
            state="readonly",
            width=50
        )
        self.microphone_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        self.populate_microphones()

        # Frame for Speaker Output Selection
        speaker_frame = ttk.LabelFrame(self.root, text="Speaker Output Selection")
        speaker_frame.pack(padx=10, pady=10, fill="x")

        speaker_label = ttk.Label(speaker_frame, text="Select Speaker Output:")
        speaker_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.speaker_dropdown = ttk.Combobox(
            speaker_frame,
            textvariable=self.output_device_var,
            state="readonly",
            width=50
        )
        self.speaker_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        self.populate_output_speakers()

        # Frame for Source Language Selection
        source_lang_frame = ttk.LabelFrame(self.root, text="Source Language Selection")
        source_lang_frame.pack(padx=10, pady=10, fill="x")

        source_lang_label = ttk.Label(source_lang_frame, text="Select Source Language:")
        source_lang_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.source_language_dropdown = ttk.Combobox(
            source_lang_frame,
            textvariable=self.source_language_var,
            values=AVAILABLE_SOURCE_LANGUAGES,
            state="readonly",
            width=20
        )
        self.source_language_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame for Translation Selection
        translation_frame = ttk.LabelFrame(self.root, text="Translation")
        translation_frame.pack(padx=10, pady=10, fill="x")

        translation_label = ttk.Label(translation_frame, text="Select Target Language:")
        translation_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.translation_dropdown = ttk.Combobox(
            translation_frame,
            textvariable=self.translation_var,
            values=AVAILABLE_TRANSLATIONS,
            state="readonly",
            width=20
        )
        self.translation_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame for Control Buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=10, fill="x")

        self.start_button = ttk.Button(control_frame, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Transcription", command=self.stop_transcription, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = ttk.Button(control_frame, text="Save Transcription", command=self.save_transcription)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame for Process Durations (New Box)
        status_frame = ttk.LabelFrame(self.root, text="Process Durations")
        status_frame.pack(padx=10, pady=10, fill="x")

        # Transcription Time Label
        self.transcription_time_label = ttk.Label(status_frame, text="Transcription Time: N/A")
        self.transcription_time_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Translation Time Label
        self.translation_time_label = ttk.Label(status_frame, text="Translation Time: N/A")
        self.translation_time_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Scrolled Text for Transcription Output
        transcription_label = ttk.Label(self.root, text="Transcription:")
        transcription_label.pack(padx=10, pady=(10, 0), anchor="w")

        self.transcription_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=140, height=40, state='disabled')
        self.transcription_area.pack(padx=10, pady=5, fill="both", expand=True)

    def populate_microphones(self):
        input_devices, _ = list_microphones()
        if not input_devices:
            error_msg = "No input devices found."
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
            self.root.destroy()
            return
        self.microphone_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(input_devices)]
        self.microphone_dropdown['values'] = self.microphone_names
        self.microphone_dropdown.current(0)  # Select the first microphone by default
        logging.info(f"Detected Microphones: {self.microphone_names}")

    def populate_output_speakers(self):
        _, output_devices = list_microphones()
        if not output_devices:
            error_msg = "No output devices found."
            messagebox.showerror("Error", error_msg)
            logging.error(error_msg)
            self.root.destroy()
            return
        self.output_device_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(output_devices)]
        self.speaker_dropdown['values'] = self.output_device_names
        self.speaker_dropdown.current(0)  # Select the first speaker by default
        logging.info(f"Detected Output Devices: {self.output_device_names}")

    def load_model(self, model_name):
        try:
            self.append_transcription(f"Loading Whisper model '{model_name}'...\n")
            self.model = whisper.load_model(model_name)
            self.append_transcription(f"Model '{model_name}' loaded successfully.\n")
            logging.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model '{model_name}'.\nError: {e}")
            self.append_transcription(f"Error loading model '{model_name}': {e}\n")
            logging.error(f"Failed to load model '{model_name}'. Error: {e}")

    def on_model_change(self, event):
        selected_model = self.current_model_name.get()
        if self.transcribing:
            messagebox.showwarning("Transcription Active", "Please stop transcription before changing the model.")
            # Revert to the previous model selection
            self.model_dropdown.set(self.model.name if self.model else "base")
            return
        self.load_model(selected_model)

    def start_transcription(self):
        selected_input_index = self.microphone_dropdown.current()
        if selected_input_index == -1:
            messagebox.showwarning("Warning", "Please select a microphone.")
            return
        selected_output_index = self.speaker_dropdown.current()
        if selected_output_index == -1:
            messagebox.showwarning("Warning", "Please select a speaker output.")
            return
        input_devices, output_devices = list_microphones()
        device_id = input_devices[selected_input_index]['index']
        output_device_id = output_devices[selected_output_index]['index']

        # Disable start button and enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.model_dropdown.config(state="disabled")
        self.microphone_dropdown.config(state="disabled")
        self.speaker_dropdown.config(state="disabled")
        self.source_language_dropdown.config(state="disabled")
        self.translation_dropdown.config(state="disabled")

        # Reset duration labels
        self.transcription_time_label.config(text="Transcription Time: N/A")
        self.translation_time_label.config(text="Translation Time: N/A")

        self.transcribing = True

        # Start the transcription thread
        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()

        # Start the audio stream
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
            logging.info("Transcription started.")
        except Exception as e:
            messagebox.showerror("Audio Stream Error", f"Failed to start audio stream.\nError: {e}")
            self.append_transcription(f"Error starting audio stream: {e}\n")
            logging.error(f"Failed to start audio stream. Error: {e}")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.model_dropdown.config(state="readonly")
            self.microphone_dropdown.config(state="readonly")
            self.speaker_dropdown.config(state="readonly")
            self.source_language_dropdown.config(state="readonly")
            self.translation_dropdown.config(state="readonly")
            self.transcribing = False

    def stop_transcription(self):
        self.transcribing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.model_dropdown.config(state="readonly")
        self.microphone_dropdown.config(state="readonly")
        self.speaker_dropdown.config(state="readonly")
        self.source_language_dropdown.config(state="readonly")
        self.translation_dropdown.config(state="readonly")
        self.append_transcription("\nTranscription stopped.\n")
        logging.info("Transcription stopped.")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
            logging.warning(f"Audio status: {status}")
        if self.transcribing:
            audio_queue.put(indata.copy())

    def transcribe_audio(self):
        while self.transcribing:
            try:
                data = audio_queue.get(timeout=1)
                self.audio_buffer = np.concatenate((self.audio_buffer, data), axis=0)
                if len(self.audio_buffer) >= SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:
                    # Prepare audio chunk
                    audio_chunk = self.audio_buffer[:SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION]
                    self.audio_buffer = self.audio_buffer[SAMPLE_RATE * TRANSCRIPTION_CHUNK_DURATION:]

                    # Flatten the audio data
                    audio_flat = audio_chunk.flatten()

                    # Get the selected source language
                    source_language = self.source_language_var.get()
                    source_lang_code = SOURCE_LANGUAGE_CODES.get(source_language)

                    # Transcribe the audio chunk
                    self.gui_queue.put({'transcription': "Transcribing...\n"})

                    # Start time for transcription
                    transcription_start_time = time.time()

                    if source_lang_code:
                        result = self.model.transcribe(audio_flat, language=source_lang_code)
                    else:
                        # If 'Auto' is selected, let Whisper detect the language
                        result = self.model.transcribe(audio_flat)
                    transcription_end_time = time.time()

                    transcription_duration = transcription_end_time - transcription_start_time
                    transcription_duration_formatted = f"{transcription_duration:.2f} seconds"

                    transcription = result['text'].strip()

                    if transcription:
                        # Put transcription and duration into the queue
                        self.gui_queue.put({
                            'transcription': f"You said: {transcription}\n",
                            'transcription_time': f"Transcription Time: {transcription_duration_formatted}",
                            'transcription_update': True
                        })
                        logging.info(f"Transcription completed in {transcription_duration_formatted}")

                        # Translate if a target language is selected
                        target_language = self.translation_var.get()
                        if target_language != 'None':
                            # Start time for translation
                            translation_start_time = time.time()

                            translated_text = self.translate_text(transcription, target_language)

                            translation_end_time = time.time()

                            translation_duration = translation_end_time - translation_start_time
                            translation_duration_formatted = f"{translation_duration:.2f} seconds"

                            if translated_text:
                                # Put translation and duration into the queue
                                self.gui_queue.put({
                                    'translation': f"Translation ({target_language}): {translated_text}\n",
                                    'translation_time': f"Translation Time: {translation_duration_formatted}",
                                    'translation_update': True,
                                    'translated_text': translated_text  # Include translated text for TTS
                                })
                                logging.info(f"Translation completed in {translation_duration_formatted}")
            except queue.Empty:
                continue
            except Exception as e:
                # Log the error and put it into the queue
                error_message = f"Error during transcription: {e}\n"
                self.gui_queue.put({'error': error_message})
                logging.error(f"Error during transcription: {e}")

    def translate_text(self, text, target_language):
        try:
            lang_code = TRANSLATION_LANGUAGE_CODES.get(target_language)
            if not lang_code:
                return None
            translation = self.translator.translate(text, dest=lang_code)
            return translation.text
        except Exception as e:
            self.gui_queue.put({'error': f"Error during translation: {e}\n"})
            logging.error(f"Error during translation: {e}")
            return None

    def append_transcription(self, text, tag='default'):
        # Ensure thread-safe GUI updates
        self.transcription_area.configure(state='normal')
        self.transcription_area.insert(tk.END, text, tag)
        self.transcription_area.see(tk.END)
        self.transcription_area.configure(state='disabled')

    def process_queue(self):
        try:
            while True:
                message = self.gui_queue.get_nowait()
                if 'transcription' in message:
                    self.append_transcription(message['transcription'], 'transcription')
                if 'transcription_time' in message and message.get('transcription_update'):
                    self.transcription_time_label.config(text=message['transcription_time'])
                if 'translation' in message:
                    self.append_transcription(message['translation'], 'translation')
                if 'translation_time' in message and message.get('translation_update'):
                    self.translation_time_label.config(text=message['translation_time'])
                if 'translated_text' in message:
                    # Speak the translated text in a separate thread
                    threading.Thread(target=self.speak_text, args=(message['translated_text'],), daemon=True).start()
                if 'error' in message:
                    self.append_transcription(message['error'])
        except queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.process_queue)

    def speak_text(self, text):
        try:
            # Determine the target language
            target_language = self.translation_var.get()
            if target_language == 'None':
                # Use default voice (pyttsx3)
                self.tts_engine.setProperty('voice', self.get_default_voice())
                # Save the TTS output to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                    temp_filename = tf.name
                self.tts_engine.save_to_file(text, temp_filename)
                self.tts_engine.runAndWait()

                # Play the temporary file through the selected output device
                selected_output_index = self.speaker_dropdown.current()
                _, output_devices = list_microphones()
                output_device_id = output_devices[selected_output_index]['index']

                # Read the WAV file
                data, fs = sf.read(temp_filename, dtype='float32')

                # Play the audio through the selected output device
                sd.play(data, fs, device=output_device_id)
                sd.wait()  # Wait until playback is finished

                # Remove the temporary file
                os.remove(temp_filename)
            elif target_language == 'Mandarin':
                # Use gTTS for Mandarin
                tts = gTTS(text=text, lang='zh-cn')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
                    temp_filename = tf.name
                tts.save(temp_filename)

                # Play the MP3 file through the selected output device
                playsound(temp_filename)

                # Remove the temporary file
                os.remove(temp_filename)
            else:
                # Handle other languages as needed
                self.tts_engine.setProperty('voice', self.get_default_voice())
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                    temp_filename = tf.name
                self.tts_engine.save_to_file(text, temp_filename)
                self.tts_engine.runAndWait()

                data, fs = sf.read(temp_filename, dtype='float32')

                sd.play(data, fs, device=output_device_id)
                sd.wait()

                os.remove(temp_filename)
        except Exception as e:
            error_message = f"Error during Text-to-Speech: {e}\n"
            self.gui_queue.put({'error': error_message})
            logging.error(f"Error during Text-to-Speech: {e}")

    def get_default_voice(self):
        """Returns the default voice."""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            return voices[0].id
        return None

    def save_transcription(self):
        transcription_text = self.transcription_area.get('1.0', tk.END).strip()
        if transcription_text:
            try:
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(transcription_text)
                messagebox.showinfo("Saved", "Transcription saved to transcription.txt")
                logging.info("Transcription saved to transcription.txt")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save transcription.\nError: {e}")
                logging.error(f"Failed to save transcription. Error: {e}")
        else:
            messagebox.showwarning("Warning", "No transcription to save.")
            logging.warning("Save Transcription attempted with no content.")

# Main function to run the GUI
def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(app, root))
    root.mainloop()

def on_closing(app, root):
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        if app.transcribing:
            app.stop_transcription()
        # Stop the TTS engine gracefully
        try:
            app.tts_engine.stop()
        except:
            pass
        logging.info("Application closed.")
        root.destroy()

if __name__ == "__main__":
    main()
