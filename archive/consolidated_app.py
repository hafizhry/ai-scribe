import streamlit as st
import os
import pyaudio
from pydub import AudioSegment
import wave
import threading
import numpy as np
from datetime import datetime, timedelta
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI

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
            self.current_level = np.clip(np.abs(audio_data).mean() / 5000, 0, 1)  # Normalize level
        
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

class AudioTranscriber:
    def __init__(self, api_key):
        self.api_key = api_key
        self.MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB in bytes

    def get_wav_duration(self, wav_path):
        """Get duration of WAV file in milliseconds"""
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = (frames / float(rate)) * 1000
            return duration

    def get_file_size(self, file_path):
        """Get file size in bytes"""
        return os.path.getsize(file_path)

    def split_audio(self, wav_path):
        """Split audio file into segments smaller than 20MB"""
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio)
        segments = []
        
        # Get file size per millisecond
        size_per_ms = self.get_file_size(wav_path) / total_duration
        
        # Calculate segment duration to stay under 25MB
        segment_duration = int((self.MAX_FILE_SIZE / size_per_ms) * 0.95)  # 95% of max size to be safe
        
        # Split audio into segments
        for start in range(0, total_duration, segment_duration):
            end = min(start + segment_duration, total_duration)
            segment = audio[start:end]
            
            # Create temporary file for segment
            segment_path = f"{wav_path[:-4]}_segment_{start}.wav"
            segment.export(segment_path, format="wav")
            segments.append(segment_path)
        
        return segments

    def transcribe_and_tag(self, audio_path):
        client = OpenAI(api_key=self.api_key)
        file_size = self.get_file_size(audio_path)
        
        if file_size > self.MAX_FILE_SIZE:
            # Split audio into segments
            segments = self.split_audio(audio_path)
            full_transcription = ""
            
            # Transcribe each segment
            for segment_path in segments:
                with open(segment_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                full_transcription += transcription.text + " "
                
                # Clean up segment file
                os.remove(segment_path)
                
            transcription_result = full_transcription.strip()
        else:
            # Process single file as before
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcription_result = transcription.text

        prompt = (
            "I want you to tag the given transcription into doctor and patient. "
            "Remove any name and other personally identifiable information from the conversation, make the generated conversation anonymous. "
            "Make the conversation in Indonesian. "
            "DO NOT ADD OTHER INFORMATION THAT IS NOT ON THE TRANSCRIPT. "
            "JUST RETURN THE TAGGED TRANSCRIPTION WITH EXAMPLE AS FOLLOWS: "
            "**Pasien:** ..... **Dokter:** ..... "
            "Here is the transcription: " + transcription_result + ". "
            "If there is no conversation, please write 'No conversation found.'"
        )

        tag = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical scribe that helps to tag what part of the conversation said by doctor, patient, or etc"},
                {"role": "user", "content": prompt}
            ]
        )

        tag_result = tag.choices[0].message.content
        return tag_result

class Summarize:
    def __init__(self, api_key):
        self.models = {
            "gpt": ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        }
        self.current_model = self.models["gpt"]

    def summarize_complaint(self, complaint):
        prompt = """
        Please summarize the patient's complaint by identifying the following key points:

        1. Main Issue/Concern: What is the primary problem or symptom the patient is experiencing?
        2. Duration: How long has the patient been experiencing this issue?
        3. Severity: How severe is the issue, and has it worsened or improved over time?
        4. Location: Where is the issue located, if applicable?
        5. Associated Symptoms: Are there any other symptoms that accompany the main issue?
        6. Aggravating or Relieving Factors: Are there any factors that make the problem better or worse?
        7. Previous Treatments: Has the patient tried anything to manage the issue?

        Summarize these key points in Bahasa Indonesia.
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": complaint}
        ]

        ai_msg = self.current_model.invoke(messages)
        return ai_msg.content if hasattr(ai_msg, 'content') else ai_msg

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return str(timedelta(seconds=int(seconds))).split(':')[1:]

import streamlit as st
import time
from datetime import datetime, timedelta

# Initialize session state variables
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = 'ready'  # ready, recording, processing
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = False

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def stop_recording_callback():
    st.session_state.stop_recording = True

def main():
    st.title("Audio Recording and Transcription")
    
    # Sidebar to display session state
    with st.sidebar:
        st.header("Session State")
        st.write("Recording Status:", st.session_state.recording_status)
        st.write("Start Time:", st.session_state.start_time)
        st.write("Has Transcription:", 
                 "Yes" if st.session_state.transcription else "No")
        st.write("Has Summary:", 
                 "Yes" if st.session_state.summary else "No")
    
    # Main interface
    if st.session_state.recording_status == 'ready':
        # Create two columns for the buttons
        col1, col2 = st.columns(2)
        
        # Recording option
        with col1:
            if st.button("Start Recording"):
                st.session_state.recorder = AudioRecorder()
                st.session_state.recorder.start_recording()
                st.session_state.recording_status = 'recording'
                st.session_state.start_time = datetime.now()
                st.session_state.stop_recording = False
                st.rerun()
        
        # Import option
        with col2:
            uploaded_file = st.file_uploader("Import Audio", type=['wav', 'mp3'])
            if uploaded_file is not None:
                # Create temporary file to save the uploaded audio
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Convert to WAV if MP3
                if uploaded_file.name.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(temp_path)
                    wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
                    audio.export(wav_path, format='wav')
                    os.remove(temp_path)  # Remove the original MP3
                    temp_path = wav_path
                
                # Process the audio file
                try:
                    # Initialize transcriber and summarizer
                    transcriber = AudioTranscriber(st.secrets["OPENAI_API_KEY"])
                    summarizer = Summarize(st.secrets["OPENAI_API_KEY"])
                    
                    with st.spinner("Transcribing audio..."):
                        st.session_state.transcription = transcriber.transcribe_and_tag(temp_path)
                    
                    with st.spinner("Generating summary..."):
                        st.session_state.summary = summarizer.summarize_complaint(
                            st.session_state.transcription)
                    
                    st.session_state.recording_status = 'processing'
                    os.remove(temp_path)  # Clean up temporary file
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    os.remove(temp_path)  # Clean up temporary file on error
    
    elif st.session_state.recording_status == 'recording':
        # Create the layout first
        timer_placeholder = st.empty()
        level_placeholder = st.empty()
        level_text_placeholder = st.empty()
        
        # Create stop button outside the recording loop
        if st.button("Stop Recording", key="stop_button", on_click=stop_recording_callback):
            pass
        
        # Update timer and level in real-time
        while st.session_state.recorder and st.session_state.recorder.is_recording:
            if st.session_state.stop_recording:
                # Stop and save the recording
                st.session_state.recorder.stop_recording()
                audio_file = st.session_state.recorder.save_recording()
                
                if audio_file:
                    # Initialize transcriber and summarizer
                    transcriber = AudioTranscriber(st.secrets["OPENAI_API_KEY"])
                    summarizer = Summarize(st.secrets["OPENAI_API_KEY"])
                    
                    with st.spinner("Transcribing audio..."):
                        st.session_state.transcription = transcriber.transcribe_and_tag(audio_file)
                    
                    with st.spinner("Generating summary..."):
                        st.session_state.summary = summarizer.summarize_complaint(
                            st.session_state.transcription)
                    
                    st.session_state.recording_status = 'processing'
                    st.rerun()
                break
            
            # Update timer
            current_time = time.time()
            elapsed_time = current_time - st.session_state.start_time.timestamp()
            timer_placeholder.markdown(f"### Recording Time: {format_time(elapsed_time)}")
            
            # Update volume level
            level = st.session_state.recorder.current_level
            level_placeholder.progress(float(level))
            level_text_placeholder.text(f"Volume Level: {int(level * 100)}%")
            
            # Short sleep to prevent overwhelming the UI
            time.sleep(0.1)
    
    # Display results
    if st.session_state.recording_status == 'processing':
        st.header("Transcription")
        st.markdown(st.session_state.transcription)
        
        st.header("Summary")
        st.markdown(st.session_state.summary)
        
        if st.button("Process New Audio"):
            # Reset all states for new processing
            st.session_state.recorder = None
            st.session_state.recording_status = 'ready'
            st.session_state.start_time = None
            st.session_state.transcription = None
            st.session_state.summary = None
            st.session_state.stop_recording = False
            st.rerun()

if __name__ == "__main__":
    main()