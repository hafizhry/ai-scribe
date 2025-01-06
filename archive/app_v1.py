import streamlit as st
import os
import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
import time
from openai import OpenAI
from langchain.chat_models import ChatOpenAI

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

    def transcribe_and_tag(self, audio_path):
        client = OpenAI(api_key=self.api_key)

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
            "gpt": ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
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

def main():
    st.title("AI Scribe App")

    # Debugging info in sidebar
    st.sidebar.title("Debugging Info")
    st.sidebar.write("Session State:", st.session_state)

    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = False
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'show_new_recording_button' not in st.session_state:
        st.session_state.show_new_recording_button = False
    if 'transcription_done' not in st.session_state:
        st.session_state.transcription_done = False

    transcriber = AudioTranscriber(api_key=st.secrets["OPENAI_API_KEY"])
    summarizer = Summarize(api_key=st.secrets["OPENAI_API_KEY"])

    col1, col2 = st.columns(2)

    with col1:
        # "Start Recording" button if not recording and no transcription done
        if not st.session_state.recording_state and (not st.session_state.transcription_done):
            if st.button("Start Recording"):
                st.session_state.recording_state = True
                st.session_state.start_time = datetime.now()
                st.session_state.recorder.start_recording()

    with col2:
        # "Stop Recording" button if recording
        if st.session_state.recording_state:
            stop_button = st.button("Stop Recording")
            if stop_button:
                st.session_state.recording_state = False
                st.session_state.recorder.stop_recording()
                filename = st.session_state.recorder.save_recording()
                if filename:
                    st.session_state.audio_file = filename
                    st.session_state.show_new_recording_button = True

    # Show recording progress if in recording_state
    if st.session_state.recording_state:
        timer_placeholder = st.empty()
        progress_placeholder = st.empty()

        while st.session_state.recording_state:
            elapsed_time = datetime.now() - st.session_state.start_time
            timer_placeholder.warning(f"Recording in progress... {str(elapsed_time).split('.')[0]}")

            audio_level = st.session_state.recorder.current_level
            progress_placeholder.progress(min(audio_level, 1.0))

            time.sleep(0.1)

    # Once we have an audio file and it exists
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.audio(st.session_state.audio_file)

        # If transcription hasn't been done yet
        if not st.session_state.get("transcription_result"):
            st.write("### Transcription")
            with st.spinner("Transcribing audio..."):
                transcription = transcriber.transcribe_and_tag(st.session_state.audio_file)
                st.session_state["transcription_result"] = transcription
                st.session_state.transcription_done = True
        
        # Show the transcription result
        if st.session_state.get("transcription_result"):
            st.markdown(st.session_state["transcription_result"])

            # Summarize
            st.write("### Summarized Complaint")
            with st.spinner("Summarizing complaint..."):
                summary = summarizer.summarize_complaint(st.session_state["transcription_result"])
                st.markdown(summary)

        # Download button
        with open(st.session_state.audio_file, 'rb') as f:
            st.download_button(
                label="Download Recording",
                data=f,
                file_name=os.path.basename(st.session_state.audio_file),
                mime="audio/wav"
            )

    # "Start New Recording" button after finishing
    if st.session_state.show_new_recording_button:
        st.markdown("---")
        if st.button("Start New Recording", key="start_new_recording_bottom"):
            st.session_state.recording_state = False
            st.session_state.audio_file = None
            st.session_state.start_time = None
            st.session_state.show_new_recording_button = False
            st.session_state.transcription_done = False
            st.session_state.pop("transcription_result", None)
            st.rerun()

if __name__ == "__main__":
    main()
