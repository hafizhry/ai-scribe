import streamlit as st
import time
from datetime import datetime
import os
from audio.recorder import AudioRecorder
from audio.transcriber import AudioTranscriber
from processors.summarizer import Summarize, TemplateType
from utils.time_utils import format_time
from utils.openai_utils import get_api_key
from utils.pdf_utils import export_summary_to_pdf, get_pdf_download_button
from config import Config
from pydub import AudioSegment
from fpdf import FPDF
from openai import OpenAI
import openai

# Set page configuration
st.set_page_config(
    page_title="MedAI Scribe",
    page_icon="🎙️",
    layout="wide"
)

# Initialize configuration
Config.ensure_directories()

# Initialize session state variables
def init_session_state():
    if 'recorder' not in st.session_state:
        st.session_state.recorder = None
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = 'ready'
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'stop_recording' not in st.session_state:
        st.session_state.stop_recording = False
    if 'template_type' not in st.session_state:
        st.session_state.template_type = TemplateType.SOAP.value
    if 'custom_sections' not in st.session_state:
        st.session_state.custom_sections = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None

def stop_recording_callback():
    st.session_state.stop_recording = True

def process_audio_file(file_path: str, template_type: str, custom_sections: list = None):
    """Process audio file and generate transcription and summary."""
    try:
        transcriber = AudioTranscriber(st.session_state.api_key)
        summarizer = st.session_state.summarizer

        with st.spinner("Processing audio..."):
            st.session_state.transcription = transcriber.transcribe_and_tag(file_path)

        with st.spinner("Creating summary..."):
            if template_type == TemplateType.CUSTOM.value and custom_sections:
                summarizer.create_custom_template(custom_sections)
            st.session_state.summary = summarizer.summarize(
                st.session_state.transcription,
                TemplateType(template_type)
            )

        st.session_state.recording_status = 'processing'
        return True
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return False

def render_recording_interface():
    """Render the recording interface with timer and volume level."""
    timer_placeholder = st.empty()
    level_placeholder = st.empty()
    level_text_placeholder = st.empty()
    
    st.button("Stop Recording", key="stop_button", on_click=stop_recording_callback)
    
    while st.session_state.recorder and st.session_state.recorder.is_recording:
        if st.session_state.stop_recording:
            st.session_state.recorder.stop_recording()
            audio_file = st.session_state.recorder.save_recording()
            
            if audio_file and process_audio_file(
                audio_file,
                st.session_state.template_type,
                st.session_state.custom_sections
            ):
                st.rerun()
            break
        
        current_time = time.time()
        elapsed_time = current_time - st.session_state.start_time.timestamp()
        timer_placeholder.markdown(f"### Recording Duration: {format_time(elapsed_time)}")
        
        level = st.session_state.recorder.current_level
        level_placeholder.progress(float(level))
        level_text_placeholder.text(f"Volume Level: {int(level * 100)}%")
        
        time.sleep(0.1)

def render_results():
    """Render transcription and summary results."""    
    st.header("Summary ")
    st.markdown(st.session_state.summary)

    with st.expander("View Full Transcription"):
        st.markdown(st.session_state.transcription)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Summary as PDF"):
            
            pdf_path = export_summary_to_pdf(
                transcription=st.session_state.transcription,
                summary=st.session_state.summary,
                template_type=st.session_state.template_type
            )
            
            if pdf_path and os.path.exists(pdf_path):
                if get_pdf_download_button(pdf_path):
                    st.success("PDF exported successfully!")
    
    with col2:
        if st.button("Start New Conversation"):
            for key in ['recorder', 'transcription', 'summary', 'stop_recording']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.recording_status = 'ready'
            st.rerun()

def handle_file_upload():
    """Handle audio file upload and processing."""
    st.markdown("### Import Audio")
    uploaded_file = st.file_uploader(
        "", 
        type=['wav', 'mp3'], 
        key="audio_file_uploader",
        label_visibility="collapsed"
    )
    
    # Custom styling for the upload area
    st.markdown("""
        <style>
        .uploadedFile {
            display: none;
        }
        .stFileUploader > div {
            background-color: #2A2B2E;
            padding: 20px;
            border-radius: 10px;
            border: none;
        }
        .stFileUploader > div > div {
            background-color: transparent;
            border: none;
        }
        .stFileUploader > div:hover {
            background-color: #363739;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Add explanatory text below uploader
    st.markdown("""
        <div style="
            text-align: center;
            color: #808080;
            font-size: 0.8em;
            margin-top: -10px;
            padding: 10px;
        ">
            Limit 200MB per file • WAV, MP3
        </div>
    """, unsafe_allow_html=True)
    if uploaded_file is not None:
        temp_dir = Config.TEMP_UPLOADS_DIR
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.mp3'):
                audio = AudioSegment.from_mp3(temp_path)
                wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_path, format='wav')
                os.remove(temp_path)
                temp_path = wav_path
            
            if process_audio_file(
                temp_path,
                st.session_state.template_type,
                st.session_state.custom_sections
            ):
                st.rerun()
        except Exception as e:
            st.error(f"Audio Import Error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def render_sidebar_configuration():
    """Render sidebar configuration with API key input and template selection."""
    
    st.header("Quick Guide")

    with st.expander("How to Use MedAI Scribe", expanded=False):
        st.markdown("""
        ### How to Use MedAI Scribe
        
        1. **Setup**
        - Enter OpenAI API key in sidebar Configuration
        - Choose summary template (SOAP, APSO, etc.)
        
        2. **Record or Upload**
        - Click 'Start Recording' to record live
        - OR drag & drop audio file (WAV/MP3)
        
        3. **Get Results**
        - View generated summary
        - Export to PDF if needed
        - Start new session
        
        """)
    
    st.header("Configuration")
    
    # API Key input
    with st.expander("API Key Configuration", expanded=False):
        st.markdown("""
        #### API Key Guide
        - Enter your OpenAI API Key
        - Get your API key at [OpenAI Platform](https://platform.openai.com/api-keys)
        """)
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter API key",
            key="api_key_input",
            value=st.session_state.get('api_key', '')
        )
        
        if api_key:
            with st.spinner("Verifying API Key..."):
                actual_key, is_valid = get_api_key(api_key)
                
                if is_valid:
                    if st.session_state.api_key != actual_key:
                        st.session_state.api_key = actual_key
                        st.session_state.summarizer = Summarize(actual_key)
                        st.success("✅ API Key validated and configured successfully!")
                else:
                    st.error("❌ Invalid API Key. Please check and try again.")
                    st.session_state.api_key = None
                    st.session_state.summarizer = None

    # Template selection
    if st.session_state.summarizer:
        template_options = st.session_state.summarizer.get_available_templates()
        
        with st.expander("Summary Configuration", expanded=False):
            st.markdown("""
            #### Summary Template Guide
            Select a template for generating the summary
            - **SOAP Format**: Outlines subjective, objective, assessment, and plan
            - **APSO Format**: Outlines assessment, plan, subjective, and objective
            - **Initial Visit**: Outlines patient history and initial visit details
            - **Multi-Section Format**: Allows you to define multiple sections for the summary
            - **Custom**: Define your own sections for the summary
            """)

            template_type = st.selectbox(
                "Select Summary Template",
                options=[t.value for t in TemplateType],
                format_func=lambda x: {
                    'intake': 'Initial Visit',
                    'soap': 'SOAP Format',
                    'apso': 'APSO Format',
                    'multiple_section': 'Multi-Section Format',
                    'custom': 'Custom'
                }.get(x, template_options[x]),
                key="template_selector"
            )
        
        custom_sections = []
        if template_type == TemplateType.CUSTOM.value:
            with st.expander("Custom Template Sections", expanded=True):
                st.info("Enter section names one per line. These will become headers in your summary.")
                sections_text = st.text_area(
                    "Custom Sections",
                    placeholder="Enter section names here...\nExample:\nTrauma History\nSocial Support\nCoping Mechanisms",
                    height=100
                )
                if sections_text:
                    custom_sections = [s.strip() for s in sections_text.split('\n') if s.strip()]
                    if not custom_sections:
                        st.warning("Please enter at least one section name.")
        
        return template_type, custom_sections
    else:
        st.warning("Please enter an API Key to continue.")
        return None, None

def main():
    st.title("MedAI Scribe")
    st.markdown("""#### AI powered medical scribe that generates summaries from patient and doctor conversations.""")
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**
    
    DO NOT USE REAL DATA THAT INCLUDES ANY PERSONAL IDENTIFIABLE INFORMATION (PII) SUCH AS:
    - PATIENT NAMES
    - DATES OF BIRTH
    - ADDRESSES
    - PHONE NUMBERS
    - ANY OTHER IDENTIFYING DETAILS
    """)
    
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        config_result = render_sidebar_configuration()
        if config_result:
            template_type, custom_sections = config_result
            st.session_state.template_type = template_type
            st.session_state.custom_sections = custom_sections
    
    # Main interface
    if not st.session_state.summarizer:
        st.info("Please enter your API Key in the configuration panel to begin.")
        return
        
    if st.session_state.recording_status == 'ready':
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Record Audio")
            # Create a card-like container for recording
            with st.container():
                st.markdown("""
                    <div style="
                        background-color: #1E1F23;
                        padding: 40px 20px;
                        border-radius: 12px;
                        text-align: center;
                        margin-bottom: 10px;
                        height: 165px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                    ">
                        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                <line x1="12" y1="19" x2="12" y2="23"></line>
                                <line x1="8" y1="23" x2="16" y2="23"></line>
                            </svg>
                            <div style="text-align: left;">
                                <p style="color: #E0E0E0; margin: 0; font-size: 1.1em;">Click button to start recording</p>
                                <p style="color: #808080; font-size: 0.85em; margin: 0;">No time limit • WAV format</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("Start Recording", use_container_width=True):
                    st.session_state.recorder = AudioRecorder()
                    st.session_state.recorder.start_recording()
                    st.session_state.recording_status = 'recording'
                    st.session_state.start_time = datetime.now()
                    st.session_state.stop_recording = False
                    st.rerun()
        
        with col2:
            handle_file_upload()
            
    elif st.session_state.recording_status == 'recording':
        timer_placeholder = st.empty()
        level_placeholder = st.empty()
        level_text_placeholder = st.empty()
        
        st.button("Stop Recording", key="stop_button", on_click=stop_recording_callback)
        
        while st.session_state.recorder and st.session_state.recorder.is_recording:
            if st.session_state.stop_recording:
                st.session_state.recorder.stop_recording()
                audio_file = st.session_state.recorder.save_recording()
                
                if audio_file and process_audio_file(
                    audio_file,
                    st.session_state.template_type,
                    st.session_state.custom_sections
                ):
                    st.rerun()
                break
            
            current_time = time.time()
            elapsed_time = current_time - st.session_state.start_time.timestamp()
            timer_placeholder.markdown(f"### Recording Time: {format_time(elapsed_time)}")
            
            level = st.session_state.recorder.current_level
            level_placeholder.progress(float(level))
            level_text_placeholder.text(f"Volume Level: {int(level * 100)}%")
            
            time.sleep(0.1)
    
    elif st.session_state.recording_status == 'recording':
        render_recording_interface()
    
    elif st.session_state.recording_status == 'processing':
        render_results()

if __name__ == "__main__":
    main()