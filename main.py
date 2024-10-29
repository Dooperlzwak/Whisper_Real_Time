import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from googletrans import Translator
import time

# Audio parameters
SAMPLE_RATE = 16000  # Whisper expects 16000 Hz
BLOCK_SIZE = 1024
CHANNELS = 1  # Mono audio
TRANSCRIPTION_CHUNK_DURATION = 5  # seconds

# Queue to hold audio data
audio_queue = queue.Queue()

# Available Whisper models
AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Available target languages for translation
AVAILABLE_TRANSLATIONS = ['None', 'German', 'Mandarin']

# Mapping for googletrans language codes
TRANSLATION_LANGUAGE_CODES = {
    'None': None,
    'German': 'de',
    'Mandarin': 'zh-cn'
}

# Available source languages for transcription
AVAILABLE_SOURCE_LANGUAGES = [
    'Auto', 'English', 'German', 'Mandarin', 'Spanish',
    'French', 'Japanese', 'Korean', 'Italian', 'Russian'
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
    'Russian': 'ru'
}

# Function to list available microphones
def list_microphones():
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    return input_devices

# GUI Application Class
class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription with Translation")
        self.root.geometry("1300x1000")  # Increased window size for better layout

        # Initialize variables
        self.model = None
        self.current_model_name = tk.StringVar(value="base")  # Default model
        self.microphone_var = tk.StringVar()
        self.translation_var = tk.StringVar(value="None")  # Default no translation
        self.source_language_var = tk.StringVar(value="Auto")  # Default auto-detect
        self.transcribing = False
        self.audio_buffer = np.zeros((0, CHANNELS), dtype=np.float32)
        self.stream = None
        self.transcription_thread = None
        self.translator = Translator()

        # Initialize GUI queue for thread-safe updates
        self.gui_queue = queue.Queue()

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
        devices = list_microphones()
        if not devices:
            messagebox.showerror("Error", "No input devices found.")
            self.root.destroy()
            return
        self.microphone_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(devices)]
        self.microphone_dropdown['values'] = self.microphone_names
        self.microphone_dropdown.current(0)  # Select the first microphone by default

    def load_model(self, model_name):
        try:
            self.append_transcription(f"Loading Whisper model '{model_name}'...\n")
            self.model = whisper.load_model(model_name)
            self.append_transcription(f"Model '{model_name}' loaded successfully.\n")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model '{model_name}'.\nError: {e}")
            self.append_transcription(f"Error loading model '{model_name}': {e}\n")

    def on_model_change(self, event):
        selected_model = self.current_model_name.get()
        if self.transcribing:
            messagebox.showwarning("Transcription Active", "Please stop transcription before changing the model.")
            # Revert to the previous model selection
            self.model_dropdown.set(self.model.name if self.model else "base")
            return
        self.load_model(selected_model)

    def start_transcription(self):
        selected_index = self.microphone_dropdown.current()
        if selected_index == -1:
            messagebox.showwarning("Warning", "Please select a microphone.")
            return
        device_id = list_microphones()[selected_index]['index']
        
        # Disable start button and enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.model_dropdown.config(state="disabled")
        self.microphone_dropdown.config(state="disabled")
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
        except Exception as e:
            messagebox.showerror("Audio Stream Error", f"Failed to start audio stream.\nError: {e}")
            self.append_transcription(f"Error starting audio stream: {e}\n")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.model_dropdown.config(state="readonly")
            self.microphone_dropdown.config(state="readonly")
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
        self.source_language_dropdown.config(state="readonly")
        self.translation_dropdown.config(state="readonly")
        self.append_transcription("\nTranscription stopped.\n")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
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
                                    'translation_update': True
                                })
            except queue.Empty:
                continue
            except Exception as e:
                # Log the error and put it into the queue
                error_message = f"Error during transcription: {e}\n"
                self.gui_queue.put({'error': error_message})

    def translate_text(self, text, target_language):
        try:
            lang_code = TRANSLATION_LANGUAGE_CODES.get(target_language)
            if not lang_code:
                return None
            translation = self.translator.translate(text, dest=lang_code)
            return translation.text
        except Exception as e:
            self.gui_queue.put({'error': f"Error during translation: {e}\n"})
            return None

    def append_transcription(self, text):
        # Ensure thread-safe GUI updates
        self.transcription_area.configure(state='normal')
        self.transcription_area.insert(tk.END, text)
        self.transcription_area.see(tk.END)
        self.transcription_area.configure(state='disabled')

    def process_queue(self):
        try:
            while True:
                message = self.gui_queue.get_nowait()
                if 'transcription' in message:
                    self.append_transcription(message['transcription'])
                if 'transcription_time' in message and message.get('transcription_update'):
                    self.transcription_time_label.config(text=message['transcription_time'])
                if 'translation' in message:
                    self.append_transcription(message['translation'])
                if 'translation_time' in message and message.get('translation_update'):
                    self.translation_time_label.config(text=message['translation_time'])
                if 'error' in message:
                    self.append_transcription(message['error'])
        except queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.process_queue)

    def save_transcription(self):
        transcription_text = self.transcription_area.get('1.0', tk.END).strip()
        if transcription_text:
            try:
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(transcription_text)
                messagebox.showinfo("Saved", "Transcription saved to transcription.txt")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save transcription.\nError: {e}")
        else:
            messagebox.showwarning("Warning", "No transcription to save.")

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
        root.destroy()

if __name__ == "__main__":
    main()
