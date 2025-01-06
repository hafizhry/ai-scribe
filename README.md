# MedAI Scribe

An AI-powered medical transcription and summarization tool built with Streamlit and OpenAI.

## ⚠️ Important Disclaimer

**DO NOT INCLUDE ANY PERSONAL IDENTIFIABLE INFORMATION (PII) IN RECORDINGS OR UPLOADS**
- No patient names
- No dates of birth
- No addresses
- No phone numbers
- No other identifying details

## 🚀 Features

- Live audio recording
- Audio file upload support (WAV, MP3)
- Automatic transcription using OpenAI Whisper
- Customizable summary templates:
  - SOAP Format
  - APSO Format
  - Initial Visit
  - Multi-Section
  - Custom Templates
- PDF export functionality
- Simple and straightforward interface

## 🛠️ Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/medai-scribe.git
cd medai-scribe
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your environment variables
```bash
# Create a .env file or add to your environment
OPENAI_API_KEY=your_api_key_here
```

## 📦 Dependencies

- streamlit
- openai
- pydub
- fpdf
- python-dotenv
- [Other dependencies listed in requirements.txt]

## 🚀 Usage

1. Start the application
```bash
streamlit run app.py
```

2. Enter your OpenAI API key in the sidebar configuration
3. Choose your preferred summary template
4. Start recording or upload an audio file
5. View generated transcription and summary
6. Export to PDF if needed

## 📁 Project Structure

```
medai-scribe/
├── app.py
├── audio/
│   ├── recorder.py
│   └── transcriber.py
├── processors/
│   └── summarizer.py
├── utils/
│   ├── time_utils.py
│   ├── openai_utils.py
│   └── pdf_utils.py
└── config.py
```

## 🔑 API Key

You'll need an OpenAI API key to use this application. Get one at [OpenAI Platform](https://platform.openai.com/api-keys).

## 🙏 Acknowledgements

- OpenAI for their Whisper and GPT models
- Streamlit for the fantastic web framework

## 📧 Contact

Hafizh Yusuf
Master of Health Informatics
University of Michigan

Email: hafizhry@umich.edu
