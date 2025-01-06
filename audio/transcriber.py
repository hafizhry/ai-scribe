import os
from pydub import AudioSegment
import wave
from openai import OpenAI

class AudioTranscriber:
    def __init__(self, api_key):
        self.api_key = api_key
        self.MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB in bytes

    def get_wav_duration(self, wav_path):
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = (frames / float(rate)) * 1000
            return duration

    def get_file_size(self, file_path):
        return os.path.getsize(file_path)

    def split_audio(self, wav_path):
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio)
        segments = []
        
        size_per_ms = self.get_file_size(wav_path) / total_duration
        segment_duration = int((self.MAX_FILE_SIZE / size_per_ms) * 0.95)
        
        for start in range(0, total_duration, segment_duration):
            end = min(start + segment_duration, total_duration)
            segment = audio[start:end]
            segment_path = f"{wav_path[:-4]}_segment_{start}.wav"
            segment.export(segment_path, format="wav")
            segments.append(segment_path)
        
        return segments

    def transcribe_and_tag(self, audio_path):
        client = OpenAI(api_key=self.api_key)
        file_size = self.get_file_size(audio_path)
        
        if file_size > self.MAX_FILE_SIZE:
            segments = self.split_audio(audio_path)
            full_transcription = ""
            
            for segment_path in segments:
                with open(segment_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                full_transcription += transcription.text + " "
                os.remove(segment_path)
                
            transcription_result = full_transcription.strip()
        else:
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

        return tag.choices[0].message.content