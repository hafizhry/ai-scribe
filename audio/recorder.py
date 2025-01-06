import os
import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
import streamlit as st

class AudioRecorder:
    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024, format=pyaudio.paInt16):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.thread = None
        self.start_time = None
        self.current_level = 0

    def record_audio(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer
        )
        
        self.frames = []
        self.start_time = datetime.now()
        while self.is_recording:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.current_level = np.clip(np.abs(audio_data).mean() / 5000, 0, 1)
        
        self.stream.stop_stream()
        self.stream.close()
        self.start_time = None

    def start_recording(self):
        self.is_recording = True
        self.thread = threading.Thread(target=self.record_audio)
        self.thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def save_recording(self, filename=None):
        if not self.frames:
            st.warning("No recording to save!")
            return None
            
        if filename is None:
            os.makedirs('recordings', exist_ok=True)
            filename = os.path.join('recordings', f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            
        if not filename.endswith('.wav'):
            filename += '.wav'
            
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            
        return filename
        
    def __del__(self):
        self.audio.terminate()
