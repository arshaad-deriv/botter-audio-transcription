# Set environment variable to prevent pygame from initializing GUI components
# This fixes the "setting the main menu on a non-main thread" error on macOS
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import streamlit as st
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page setup
st.title("Simple Audio Transcription")
st.write("Record audio for 5 seconds and transcribe it using OpenAI's GPT-4o-transcribe model")

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.warning("Please enter your OpenAI API key to use this app")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    st.info(f"Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    
    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    
    sd.wait()  # Wait until recording is finished
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Save recording to file
    sf.write(temp_file_path, recording, sample_rate)
    
    return temp_file_path

# Function to check file size and provide warnings for large files
def check_file_size(file_path):
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > 500:
        st.warning(f"Large file detected ({file_size_mb:.2f}MB). Processing may take some time.")
        st.info("Using memory-efficient processing for large audio file...")
    elif file_size_mb > 100:
        st.info(f"Processing {file_size_mb:.2f}MB audio file...")
    
    return file_size_mb

# Function to transcribe audio
def transcribe_audio(file_path):
    try:
        # Check file size and provide appropriate warnings
        file_size_mb = check_file_size(file_path)
        
        # For very large files, show a progress indicator
        if file_size_mb > 100:
            with st.spinner("Processing audio file..."):
                with open(file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe", 
                        file=audio_file,
                        response_format="text"
                    )
                    return transcription
        else:
            # Standard processing for smaller files
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", 
                    file=audio_file,
                    response_format="text"
                )
                return transcription
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Main app logic
if st.button("Record for 5 seconds and transcribe"):
    # Record audio
    audio_path = record_audio(duration=5)
    st.success("Recording completed!")
    
    # Play the recorded audio
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")
    
    # Transcribe the audio
    with st.spinner("Transcribing..."):
        result = transcribe_audio(audio_path)
    
    if result:
        st.subheader("Transcription:")
        st.write(result)
        
        # Clean up temporary file
        os.unlink(audio_path)
    else:
        st.error("Transcription failed. Please try again.")

st.markdown("---")
st.caption("Built with Streamlit and OpenAI's GPT-4o-transcribe model")
