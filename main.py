import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Initialize the Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")  # Options: 'tiny', 'small', 'medium', 'large'
print("Model loaded.")

# Audio parameters
SAMPLE_RATE = 16000  # Whisper expects 16000 Hz
BLOCK_SIZE = 1024
CHANNELS = 1  # Mono audio
TRANSCRIPTION_CHUNK_DURATION = 5  # seconds

# Queue to hold audio data
audio_queue = queue.Queue()

# Function to list available microphones
def list_microphones():
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    return input_devices

# GUI Application Class
class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Speech Transcription")
        self.root.geometry("800x600")
        
        # Microphone Selection
        self.microphone_label = ttk.Label(root, text="Select Microphone:")
        self.microphone_label.pack(pady=5)
        
        self.microphone_var = tk.StringVar()
        self.microphone_dropdown = ttk.Combobox(root, textvariable=self.microphone_var, state="readonly", width=50)
        self.microphone_dropdown.pack(pady=5)
        
        self.populate_microphones()
        
        # Start Button
        self.start_button = ttk.Button(root, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(pady=10)
        
        # Stop Button
        self.stop_button = ttk.Button(root, text="Stop Transcription", command=self.stop_transcription, state="disabled")
        self.stop_button.pack(pady=5)
        
        # Scrolled Text for Transcription Output
        self.transcription_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30, state='disabled')
        self.transcription_area.pack(padx=10, pady=10)
        
        # Transcription Control
        self.transcribing = False
        self.audio_buffer = np.zeros((0, CHANNELS), dtype=np.float32)
        self.stream = None
        
    def populate_microphones(self):
        devices = list_microphones()
        if not devices:
            messagebox.showerror("Error", "No input devices found.")
            self.root.destroy()
            return
        self.microphone_names = [f"{device['name']} (ID: {idx})" for idx, device in enumerate(devices)]
        self.microphone_dropdown['values'] = self.microphone_names
        self.microphone_dropdown.current(0)  # Select the first microphone by default
    
    def start_transcription(self):
        selected_index = self.microphone_dropdown.current()
        if selected_index == -1:
            messagebox.showwarning("Warning", "Please select a microphone.")
            return
        device_id = list_microphones()[selected_index]['index']
        
        # Disable start button and enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.transcribing = True
        
        # Start the transcription thread
        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()
        
        # Start the audio stream
        self.stream = sd.InputStream(callback=self.audio_callback,
                                     channels=CHANNELS,
                                     samplerate=SAMPLE_RATE,
                                     blocksize=BLOCK_SIZE,
                                     device=device_id,
                                     dtype='float32')
        self.stream.start()
        
        self.append_transcription("Transcription started...\n")
    
    def stop_transcription(self):
        self.transcribing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.append_transcription("\nTranscription stopped.")
    
    def audio_callback(self, indata, frames, time, status):
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
                    
                    # Transcribe the audio chunk
                    result = model.transcribe(audio_flat, language='en')
                    transcription = result['text'].strip()
                    
                    if transcription:
                        self.append_transcription(f"You said: {transcription}\n")
            except queue.Empty:
                continue
            except Exception as e:
                self.append_transcription(f"Error during transcription: {e}\n")
    
    def append_transcription(self, text):
        # Ensure thread-safe GUI updates
        self.transcription_area.configure(state='normal')
        self.transcription_area.insert(tk.END, text)
        self.transcription_area.see(tk.END)
        self.transcription_area.configure(state='disabled')

# Main function to run the GUI
def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(app, root))
    root.mainloop()

def on_closing(app, root):
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.stop_transcription()
        root.destroy()

if __name__ == "__main__":
    main()