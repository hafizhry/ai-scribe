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

# -------------------
# Utility Classes
# -------------------
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
        """ Continuously read audio from the microphone and store frames. """
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
            # Calculate a crude "audio level" to display in the progress bar
            self.current_level = np.clip(np.abs(audio_data).mean() / 5000, 0, 1)
        
        self.stream.stop_stream()
        self.stream.close()
        self.start_time = None

    def start_recording(self):
        """ Launch a new thread to record audio. """
        self.is_recording = True
        self.thread = threading.Thread(target=self.record_audio)
        self.thread.start()

    def stop_recording(self):
        """ Stop recording audio. """
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def save_recording(self, filename=None):
        """ Save the recorded frames to a WAV file. """
        if not self.frames:
            st.warning("No recording to save!")
            return None

        if filename is None:
            os.makedirs('recordings', exist_ok=True)
            filename = os.path.join(
                'recordings',
                f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )

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
    """ Wrapper class to handle OpenAI Whisper transcriptions and conversation tagging. """
    def __init__(self, api_key):
        self.api_key = api_key

    def transcribe_and_tag(self, audio_path):
        """ Transcribe and tag speaker roles (Doctor/Patient) using OpenAI. """
        client = OpenAI(api_key=self.api_key)

        # Whisper transcription
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcription_result = transcription.text

        # Prompt for tagging
        prompt = (
            "I want you to tag the given transcription into doctor and patient. "
            "Remove any name and other personally identifiable information from the conversation, "
            "make the generated conversation anonymous, and in Indonesian. "
            "DO NOT ADD OTHER INFORMATION THAT IS NOT ON THE TRANSCRIPT. "
            "JUST RETURN THE TAGGED TRANSCRIPTION. For example:\n\n"
            "**Pasien:** ... **Dokter:** ... \n\n"
            "Here is the transcription: " + transcription_result + ". "
            "If there is no conversation, please write 'No conversation found.'"
        )

        # Tag roles using ChatCompletion
        tag = client.chat.completions.create(
            model="gpt-4o-mini",  # This model name is for illustration; adjust as needed
            messages=[
                {"role": "system", "content": "You are a helpful medical scribe that tags conversation."},
                {"role": "user", "content": prompt}
            ]
        )

        tag_result = tag.choices[0].message.content
        return tag_result

class Summarize:
    """ Wrapper class using LangChain ChatOpenAI to summarize the complaint. """
    def __init__(self, api_key):
        self.models = {
            "gpt": ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
        }
        self.current_model = self.models["gpt"]

    def summarize_complaint(self, complaint):
        """ Summarize the patient's complaint using a custom prompt. """
        prompt = """
        Please summarize the patient's complaint by identifying the following key points:

        1. Main Issue/Concern
        2. Duration
        3. Severity
        4. Location
        5. Associated Symptoms
        6. Aggravating or Relieving Factors
        7. Previous Treatments

        Summarize these key points in Bahasa Indonesia.
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": complaint}
        ]

        ai_msg = self.current_model.invoke(messages)
        return ai_msg.content if hasattr(ai_msg, 'content') else ai_msg

# -------------------
# Streamlit App
# -------------------
def main():
    st.set_page_config(page_title="Audio Recording Demo", layout="centered")

    # Initialize session state
    if "recorder" not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "recording_started" not in st.session_state:
        st.session_state.recording_started = False
    if "recording_stopped" not in st.session_state:
        st.session_state.recording_stopped = False
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = ""
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = ""
    if "level" not in st.session_state:
        st.session_state.level = 0.0

    # Instantiate the classes (You would replace 'YOUR_API_KEY' with an actual API key)
    # Do not change the logic inside these classes
    audio_transcriber = AudioTranscriber(api_key=st.secrets["OPENAI_API_KEY"])
    summarizer = Summarize(api_key=st.secrets["OPENAI_API_KEY"])

    st.title("Audio Recording Demo")

    # Buttons and logic
    if not st.session_state.recording_started and not st.session_state.recording_stopped:
        # Show 'Start Recording' button
        if st.button("Start Recording"):
            st.session_state.recording_started = True
            st.session_state.start_time = time.time()
            st.session_state.recorder.start_recording()

    if st.session_state.recording_started and not st.session_state.recording_stopped:
        # Show 'Stop Recording' button
        if st.button("Stop Recording"):
            st.session_state.recording_stopped = True
            st.session_state.recording_started = False
            st.session_state.recorder.stop_recording()
            # Save the recording
            st.session_state.audio_path = st.session_state.recorder.save_recording()

        # Show recording visualization
        st.markdown("### Recording in progress...")
        elapsed_time = int(time.time() - st.session_state.start_time)
        st.write(f"Time Elapsed: {elapsed_time} seconds")

        # Fetch current audio level from the recorder
        st.session_state.level = st.session_state.recorder.current_level
        # A simple progress bar to visualize audio level
        st.progress(st.session_state.level)

    # After recording is stopped
    if st.session_state.recording_stopped:
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            # Transcribe
            st.session_state.transcription_result = audio_transcriber.transcribe_and_tag(
                st.session_state.audio_path
            )
            # Summarize
            st.session_state.summary_result = summarizer.summarize_complaint(
                st.session_state.transcription_result
            )

            st.subheader("Transcription Result")
            st.write(st.session_state.transcription_result)

            st.subheader("Summary of Complaint")
            st.write(st.session_state.summary_result)

            # Download recording button
            with open(st.session_state.audio_path, "rb") as file:
                btn = st.download_button(
                    label="Download Recording",
                    data=file,
                    file_name=os.path.basename(st.session_state.audio_path),
                    mime="audio/wav"
                )

        # Start new encounter button
        if st.button("Start New Encounter"):
            reset_app()

def reset_app():
    # Reset session state
    st.session_state.recording_started = False
    st.session_state.recording_stopped = False
    st.session_state.start_time = None
    st.session_state.audio_path = None
    st.session_state.transcription_result = ""
    st.session_state.summary_result = ""
    st.session_state.level = 0.0
    st.rerun()

if __name__ == "__main__":
    main()