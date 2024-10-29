import tkinter as tk
from tkinter import ttk
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import pyttsx3

# Function to get list of available input devices
def get_input_devices():
    devices = sd.query_devices()
    input_devices = []
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((idx, device['name']))
    return input_devices

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Speech Transcription with Whisper")
        
        # Initialize audio parameters
        self.sampling_rate = 16000  # Whisper models expect 16000 Hz audio
        
        # Queues and threading events
        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        
        # Create GUI elements
        self.create_widgets()
        
        # Transcription and TTS threads
        self.transcription_thread = None
        self.tts_thread = None
        
        # Flag to check if transcription is running
        self.is_running = False

    def create_widgets(self):
        # Device selection
        devices = get_input_devices()
        self.device_var = tk.StringVar()
        self.device_var.set(f"{devices[0][0]}: {devices[0][1]}")  # Default to first input device

        device_frame = tk.Frame(self.root)
        device_frame.pack(pady=5)

        device_label = tk.Label(device_frame, text="Select Input Device:")
        device_label.pack(side=tk.LEFT)

        self.device_menu = ttk.Combobox(device_frame, textvariable=self.device_var, values=[f"{idx}: {name}" for idx, name in devices], state="readonly", width=50)
        self.device_menu.current(0)
        self.device_menu.pack(side=tk.LEFT)

        # Model size selection
        model_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        self.model_var = tk.StringVar()
        self.model_var.set('base')  # Default model size

        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)

        model_label = tk.Label(model_frame, text="Select Model Size:")
        model_label.pack(side=tk.LEFT)

        self.model_menu = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_sizes, state="readonly", width=10)
        self.model_menu.current(1)  # Set 'base' as default
        self.model_menu.pack(side=tk.LEFT)

        # Start/Stop button
        self.start_button = tk.Button(self.root, text="Start Transcription", command=self.toggle_transcription)
        self.start_button.pack(pady=10)

        # Transcription display
        self.transcription_text = tk.Text(self.root, height=15, width=80)
        self.transcription_text.pack(pady=10)

    def toggle_transcription(self):
        if not self.is_running:
            self.start_transcription()
            self.start_button.config(text="Stop Transcription")
            self.is_running = True
        else:
            self.stop_transcription()
            self.start_button.config(text="Start Transcription")
            self.is_running = False

    def start_transcription(self):
        # Get selected device index
        selected_device = int(self.device_menu.get().split(":")[0])
        self.device_index = selected_device

        # Get selected model size
        selected_model_size = self.model_var.get()
        self.load_model(selected_model_size)

        # Start the transcription and TTS threads
        self.stop_event.clear()
        self.transcription_thread = threading.Thread(target=self.transcribe_audio)
        self.transcription_thread.start()

        self.tts_thread = threading.Thread(target=self.speak_transcription)
        self.tts_thread.start()

        # Start audio stream in a separate thread to avoid blocking the GUI
        self.audio_thread = threading.Thread(target=self.audio_stream)
        self.audio_thread.start()

    def stop_transcription(self):
        # Stop the threads
        self.stop_event.set()
        self.transcription_thread.join()
        self.tts_thread.join()
        self.audio_thread.join()

    def load_model(self, model_size):
        # Load the Whisper model
        self.model = whisper.load_model(model_size)
        print(f"Loaded Whisper model '{model_size}'.")

    def audio_stream(self):
        # Start recording from the microphone
        try:
            with sd.InputStream(device=self.device_index, channels=1, samplerate=self.sampling_rate, callback=self.audio_callback):
                while not self.stop_event.is_set():
                    sd.sleep(100)
        except Exception as e:
            print(f"Error in audio stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_queue.put(indata.copy())

    def transcribe_audio(self):
        audio_data = np.zeros((0, 1))
        while not self.stop_event.is_set():
            try:
                # Get audio data from the queue
                data = self.audio_queue.get(timeout=0.1)
                audio_data = np.concatenate((audio_data, data), axis=0)

                # Transcribe when we have enough audio (e.g., every 5 seconds)
                if len(audio_data) >= self.sampling_rate * 5:
                    # Preprocess the audio to match Whisper's requirements
                    audio = whisper.pad_or_trim(audio_data.flatten())

                    # Create the log-Mel spectrogram
                    mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

                    # Detect the spoken language (optional)
                    _, probs = self.model.detect_language(mel)
                    language = max(probs, key=probs.get)
                    print(f"\nDetected language: {language}")

                    # Perform the transcription
                    options = whisper.DecodingOptions(language=language)
                    result = whisper.decode(self.model, mel, options)

                    # Update the GUI with the transcription
                    self.root.after(0, self.update_transcription, result.text)

                    # Put the transcription text into the TTS queue
                    self.tts_queue.put(result.text)

                    # Reset audio data
                    audio_data = np.zeros((0, 1))
            except queue.Empty:
                continue

    def update_transcription(self, text):
        self.transcription_text.insert(tk.END, text + '\n')
        self.transcription_text.see(tk.END)

    def speak_transcription(self):
        while not self.stop_event.is_set():
            try:
                # Get the text to speak from the queue
                text = self.tts_queue.get(timeout=0.1)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except queue.Empty:
                continue

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
