import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RECORDINGS_DIR = "recordings"
    TEMP_UPLOADS_DIR = "temp_uploads"
    
    @staticmethod
    def ensure_directories():
        for directory in [Config.RECORDINGS_DIR, Config.TEMP_UPLOADS_DIR]:
            os.makedirs(directory, exist_ok=True)